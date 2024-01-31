import os
import sys
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
from tqdm import tqdm
import argparse
from rouge import Rouge
from datasets import Dataset

# Please add the path of main folder into system path
ROOT = "/huawei-data/BD/project/BiTA"
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))
from src.modeling_llama_if import LlamaForCausalLM, _make_causal_mask, _expand_mask
from src.modeling_falcon_if import FalconForCausalLM
from atten_m import make_tree_attention

RANDOM_SEED = 2023


def multiply(L):
    assert isinstance(L, list)
    assert len(L) > 0
    product = 1
    for l in L:
        product *= l
    return product


def decode_alg_hf(batch_inputs, res_dict):
    ## use greedy decoding from HF
    infer_time = time.perf_counter()
    batch_response_ids = model.generate(**batch_inputs, **GENERATION_CONFIG)
    infer_time = time.perf_counter() - infer_time
    # batch_response_ids = [q[i:-1] for i, q in zip(batch_inputs["attention_mask"].sum(axis=-1), batch_response_ids)]
    batch_responses = tokenizer.batch_decode(batch_response_ids[:, batch_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"Generate greedy decoding time {infer_time:.3f}: {batch_responses}")
    res_dict["infer_time_hf"].append(infer_time)

    return res_dict


def decode_alg_direct(batch_inputs, res_dict, do_sample=False):
    eos_token_id, pad_token_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    input_ids = batch_inputs["input_ids"]
    infer_time = time.perf_counter()
    past_key_values = None
    attention_mask = batch_inputs["attention_mask"]
    prompt_len = input_ids.shape[-1]

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

    next_tokens = torch.empty(1)
    while True:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        if past_key_values:
            next_tokens_ids = next_tokens[:, None]
            position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            next_tokens_ids = input_ids

        with torch.no_grad():
            outputs = model(next_tokens_ids, attention_mask=attention_mask,  position_ids=position_ids,
                            past_key_values=past_key_values,
                            return_dict=True, use_cache=USE_CACHE)
        logits = torch.softmax(outputs.logits, dim=-1)
        next_token_logits = logits[:, -1, :]

        past_key_values = outputs.past_key_values
        if do_sample:
            torch.random.manual_seed(RANDOM_SEED)
            next_tokens = torch.multinomial(next_token_logits, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        next_tokens =  next_tokens * unfinished_sequences + pad_token_id * (1-unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )
        if unfinished_sequences.max() == 0 or input_ids.shape[-1]-prompt_len>=MAX_NEW_TOKENS:
            break

    infer_time = time.perf_counter() - infer_time
    responses = tokenizer.batch_decode(input_ids[:, batch_inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    res_dict["infer_time_direct"].append(infer_time)
    res_dict["content_direct"].append(responses[0])
    print(f"Direct greedy decoding time {infer_time:.3f} speed in tokens/s {(input_ids.shape[-1]-batch_inputs['input_ids'].shape[-1])/infer_time:.3f}: \n{responses[0]}")
    print("\n")
    # assert responses == batch_responses
    return res_dict


def decode_alg_mask_space(model_type, batch_inputs, res_dict, token_dict, data_idx, do_sample=False, infer_dtype=torch.float16, save_data=True):

    eos_token_id, pad_token_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    input_ids = batch_inputs["input_ids"]
    infer_time = time.perf_counter()
    attention_mask = batch_inputs["attention_mask"]

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    if args.mask_diff.lower() == 'false':
        Lm = MASK_ID * torch.ones((batch_size, MASK_NUM), dtype=input_ids.dtype, device=input_ids.device)
    else:
        Lm = MASK_ID + torch.arange(0, MASK_NUM, dtype=input_ids.dtype, device=input_ids.device).view(batch_size, -1)
    Lc = torch.tensor([MASK_ID for _ in range(MASK_NUM)], dtype=input_ids.dtype, device=input_ids.device).repeat(batch_size, 1)
    Pc = torch.tensor([torch.finfo(infer_dtype).max for _ in range(MASK_NUM)], dtype=infer_dtype, device=input_ids.device).repeat(batch_size, 1)

    past_key_values = None
    new_generate_token = torch.empty(1)
    while True:
        input_ids_idx = input_ids.shape[-1]

        # create input token ids
        tmp = torch.hstack([torch.hstack([Lc[:, i: i + 1], Lm]) for i in range(MASK_NUM)])
        input_ids_extend = torch.hstack([input_ids, Lm, tmp])

        # create attention mask
        combined_attention_mask = _make_causal_mask(
            input_ids_extend.shape,
            infer_dtype,
            device=input_ids_extend.device,
            past_key_values_length=0
        )
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            torch.cat([attention_mask, attention_mask.new_ones((batch_size, MASK_NUM * (MASK_NUM + 2)))], dim=-1),
            infer_dtype,
            tgt_len=input_ids_extend.shape[-1]
        ).to(input_ids_extend.device)
        for idx in range(input_ids_idx, expanded_attn_mask.shape[-1], MASK_NUM + 1):
            expanded_attn_mask[:, :, idx + MASK_NUM:, idx: idx + MASK_NUM] = torch.finfo(infer_dtype).min
        attention_mask_extend = expanded_attn_mask + combined_attention_mask

        # create position ids
        position_ids = (attention_mask_extend == 0).sum(axis=-1).squeeze(0) - 1

        # run LLM
        if past_key_values is not None:
            kv_cache_idx = torch.tensor([input_ids_idx-new_generate_token+i*(MASK_NUM+1)-1 for i in range(1, new_generate_token)], dtype=int, device=device)
            kv_cache_idx = torch.hstack([torch.arange(0, input_ids_idx-new_generate_token, dtype=int, device=device), kv_cache_idx])
            past_key_values = [(kv_cache[0][:, :, kv_cache_idx, :], kv_cache[1][:, :, kv_cache_idx, :]) for kv_cache in past_key_values]

            input_ids_extend = input_ids_extend[:, input_ids_idx - 1:]
            position_ids = position_ids[:, input_ids_idx - 1:]
            attention_mask_extend = attention_mask_extend[:, :, input_ids_idx - 1:, :]
            input_ids_idx = 1

        if "falcon" in model_type:
            attention_mask_extend = attention_mask_extend != 0

        with torch.no_grad():
            outputs = model(
                input_ids_extend,
                attention_mask=attention_mask_extend,
                position_ids=position_ids,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=USE_CACHE,
                infer_with_prefix=True,
                mask_num=MASK_NUM,
                decoding_method="space"
            )
        past_key_values = outputs.past_key_values

        logits = torch.softmax(outputs.logits, dim=-1)  # normalized logits

        if save_data:
            token_logits_candidate = logits[:, input_ids_idx: input_ids_idx + MASK_NUM, :]
            value, indice = torch.topk(token_logits_candidate, k=10, dim=-1)
            for i in range(MASK_NUM):
                token_dict[f"mask_idx_{i}"].append(json.dumps(indice[0, i, :].tolist()))
                token_dict[f"mask_val_{i}"].append(json.dumps(value[0, i, :].tolist()))

        new_generate_token = 0
        select_idx = input_ids_idx
        next_token_logit = logits[:, input_ids_idx - 1, :]
        for idx in range(MASK_NUM):
            if do_sample:
                condition = np.random.uniform() <= next_token_logit[:, Lc[:, idx]]/Pc[:, idx]
            else:
                condition = torch.argmax(next_token_logit, dim=-1)==Lc[:, idx]

            if condition:
                next_tokens = Lc[:, idx]
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                new_generate_token += 1
                unfinished_sequences = unfinished_sequences.mul(next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
                if unfinished_sequences.max() == 0:
                    break
                next_token_logit = logits[:, input_ids_idx - 1 + (idx + 1) * (MASK_NUM + 1), :]
                select_idx += MASK_NUM + 1
            else:
                break

        if do_sample:
            torch.random.manual_seed(RANDOM_SEED)
            next_tokens = torch.multinomial(next_token_logit, num_samples=1).squeeze(1)
            Lc, Pc = [], []
            for bs in range(batch_size):
                candidate_tokens = torch.multinomial(logits[bs, select_idx: select_idx + MASK_NUM, :], num_samples=1)
                Lc.append(candidate_tokens.reshape(1, -1))
                Pc.append(torch.tensor([logits[bs, select_idx+i, k] for i, k in enumerate(candidate_tokens)]).reshape(1, -1))
            Lc = torch.cat(Lc).to(device)
            Pc = torch.cat(Pc).to(device)
        else:
            next_tokens = torch.argmax(next_token_logit, dim=-1)
            # generate new candidate tokens
            Pc, Lc = torch.max(logits[:, select_idx: select_idx + MASK_NUM, :], dim=-1)

        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        new_generate_token += 1

        if save_data:
            for i in range(MASK_NUM):
                token_dict[f"Lc_{i}"].append(Lc[0, i].item())
            token_dict["token"].append(json.dumps(input_ids[0, -new_generate_token:].cpu().numpy().tolist()))
            token_dict["idx"].append(data_idx)

        unfinished_sequences = unfinished_sequences.mul(next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
        if unfinished_sequences.max() == 0 or input_ids.shape[-1] - prompt_len >= MAX_NEW_TOKENS:
            break

    infer_time = time.perf_counter() - infer_time
    responses = tokenizer.batch_decode(input_ids[:, batch_inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    print(f"No.{data_idx}: Mask greedy decoding time {infer_time:.3f} speed in tokens/s {(input_ids.shape[-1]-batch_inputs['input_ids'].shape[-1])/infer_time:.3f}: \n{responses[0]}")
    print("\n")
    res_dict["infer_time_mask"].append(infer_time)
    res_dict["content_mask"].append(responses[0])
    return token_dict, res_dict


def decode_alg_mask_tree1(model_type, batch_inputs, res_dict, token_dict, data_idx, do_sample=False, infer_dtype=torch.float16, num_candidate=3, tree_type='full'):

    assert not do_sample, "do_sample is not supported currently :("
    assert not USE_CACHE, "use_cache is not supported currently :("

    eos_token_id, pad_token_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    input_ids = batch_inputs["input_ids"]
    infer_time = time.perf_counter()
    attention_mask = batch_inputs["attention_mask"]

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    if args.mask_diff.lower() == 'false':
        Lm = MASK_ID * torch.ones((batch_size, MASK_NUM), dtype=input_ids.dtype, device=input_ids.device)
    else:
        Lm = MASK_ID + torch.arange(0, MASK_NUM, dtype=input_ids.dtype, device=input_ids.device).view(batch_size, -1)
    
    num_candidates = []
    Lcs = []
    Lcs_num_accumulate = [0]
    Lcs_num = 0
    if tree_type == "full":
        for m in range(1, MASK_NUM + 1):
            Lc = torch.tensor([MASK_ID for _ in range(num_candidate ** m)], dtype=input_ids.dtype, device=input_ids.device).repeat(batch_size, 1)
            Lcs.append(Lc)
            Lcs_num += Lc.shape[-1]
            Lcs_num_accumulate.append(Lcs_num)
            num_candidates.append(num_candidate)
    elif tree_type == "half":
        for _ in range(MASK_NUM):
            Lc = torch.tensor([MASK_ID for _ in range(num_candidate)], dtype=input_ids.dtype, device=input_ids.device).repeat(batch_size, 1)
            Lcs.append(Lc)
            Lcs_num += Lc.shape[-1]
            Lcs_num_accumulate.append(Lcs_num)
            num_candidates.append(num_candidate)
    else:
        raise ValueError
        
    mask_c = make_tree_attention(num_candidate, tree_type, dtype=infer_dtype, device=input_ids.device, mask_num=MASK_NUM)

    past_key_values = None
    new_generate_token = torch.empty(1)
    while True:
        input_ids_idx = input_ids.shape[-1]

        # create input token ids
        tmp = torch.hstack(Lcs + [Lm for _ in range(MASK_NUM + 1)]) 
        input_ids_extend = torch.hstack([input_ids, tmp])

        # create attention mask
        combined_attention_mask = _make_causal_mask(
            input_ids_extend.shape,
            infer_dtype,
            device=input_ids_extend.device,
            past_key_values_length=0
        )
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            torch.cat([attention_mask, attention_mask.new_ones((batch_size, Lcs_num + MASK_NUM * (MASK_NUM + 1)))], dim=-1),
            infer_dtype,
            tgt_len=input_ids_extend.shape[-1]
        ).to(input_ids_extend.device)
        expanded_attn_mask[:, :, input_ids_idx : input_ids_idx + Lcs_num, input_ids_idx : input_ids_idx + Lcs_num] = mask_c
        expanded_attn_mask[:, :, input_ids_idx + Lcs_num:, input_ids_idx : input_ids_idx + Lcs_num] = torch.finfo(infer_dtype).min
        for i in range(MASK_NUM):
            expanded_attn_mask[:, :, input_ids_idx + Lcs_num + (i + 1) * MASK_NUM:, input_ids_idx + Lcs_num_accumulate[i]] = 0
            expanded_attn_mask[:, :, input_ids_idx + Lcs_num + (i + 1) * MASK_NUM:, input_ids_idx + Lcs_num + i * MASK_NUM : input_ids_idx + Lcs_num + (i + 1) * MASK_NUM] = torch.finfo(infer_dtype).min
        attention_mask_extend = expanded_attn_mask + combined_attention_mask

        # create position ids
        position_ids = (attention_mask_extend==0).sum(axis=-1).squeeze(0) - 1

        if "falcon" in model_type:
            attention_mask_extend = attention_mask_extend != 0

        # run LLM
        with torch.no_grad():
            outputs = model(
                input_ids_extend,
                attention_mask=attention_mask_extend,
                position_ids=position_ids,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=USE_CACHE,
                infer_with_prefix=True,
                mask_num_all=MASK_NUM*(MASK_NUM+1),
                decoding_method="tree1"
            )

        logits = torch.softmax(outputs.logits, dim=-1)  # normalized logits

        new_generate_token = 0
        select_idx = input_ids_idx + Lcs_num
        select_candidates = []
        next_token_logit = logits[:, input_ids_idx - 1, :]

        for idx in range(MASK_NUM):
            next_token_idx = torch.argmax(next_token_logit, dim=-1)
            condition = False
            for c in range(num_candidates[idx]):
                if next_token_idx == Lcs[idx][:, c]:
                    condition = True
                    break

            if condition:
                next_tokens = next_token_idx
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                new_generate_token += 1
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
                if unfinished_sequences.max() == 0:
                    break

                if tree_type == "full":
                    next_shift = 0
                    for s in range(len(select_candidates)):
                        next_shift += select_candidates[s] * num_candidate ** (len(select_candidates) - s)
                    next_token_logit = logits[:, input_ids_idx + Lcs_num_accumulate[idx] + next_shift + c, :]
                    select_idx += MASK_NUM
                    select_candidates.append(c)
                elif tree_type == "half":
                    next_token_logit = logits[:, input_ids_idx + Lcs_num_accumulate[idx] + c, :]
                    select_idx += MASK_NUM
                    if c > 0:
                        break
            else:
                break

        next_tokens = torch.argmax(next_token_logit, dim=-1)
        
        # generate new candidate tokens
        logits_candidate = logits[:, select_idx: select_idx + MASK_NUM, :]
        _, indice = torch.topk(logits_candidate, k=max(num_candidates), dim=-1)  # indice: [1, MASK_NUM, num_candidate]
        for idx in range(MASK_NUM):
            if tree_type == "full":
                Lcs[idx] = indice[:, idx].repeat(1, num_candidate**idx)
            elif tree_type == "half":
                Lcs[idx] = indice[:, idx, :num_candidates[idx]]

        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        new_generate_token += 1

        unfinished_sequences = unfinished_sequences.mul(next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
        if unfinished_sequences.max() == 0 or input_ids.shape[-1]-prompt_len>=MAX_NEW_TOKENS:
            break

    infer_time = time.perf_counter() - infer_time
    responses = tokenizer.batch_decode(input_ids[:, batch_inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    print(f"No.{data_idx}: Mask greedy decoding time {infer_time:.3f} speed in tokens/s {(input_ids.shape[-1]-batch_inputs['input_ids'].shape[-1])/infer_time:.3f}: \n{responses[0]}")
    print("\n")
    res_dict["infer_time_mask"].append(infer_time)
    res_dict["content_mask"].append(responses[0])
    return token_dict, res_dict


def decode_alg_mask_tree2(model_type, batch_inputs, res_dict, token_dict, data_idx, do_sample=False, infer_dtype=torch.float16, num_candidate=3, tree_type='full'):

    assert not do_sample, "do_sample is not supported currently :("
    assert not USE_CACHE, "use_cache is not supported currently :("

    eos_token_id, pad_token_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    input_ids = batch_inputs["input_ids"]
    infer_time = time.perf_counter()
    attention_mask = batch_inputs["attention_mask"]

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    if args.mask_diff.lower() == 'false':
        Lm = MASK_ID * torch.ones((batch_size, MASK_NUM), dtype=input_ids.dtype, device=input_ids.device)
    else:
        Lm = MASK_ID + torch.arange(0, MASK_NUM, dtype=input_ids.dtype, device=input_ids.device).view(batch_size, -1)
    
    num_candidates = []
    Lcs = []
    Lcs_num_accumulate = [0]
    Lcs_num = 0
    if tree_type == "full":
        for m in range(1, MASK_NUM + 1):
            Lc = torch.tensor([MASK_ID for _ in range(num_candidate ** m)], dtype=input_ids.dtype, device=input_ids.device).repeat(batch_size, 1)
            Lcs.append(Lc)
            Lcs_num += Lc.shape[-1]
            Lcs_num_accumulate.append(Lcs_num)
            num_candidates.append(num_candidate)
    elif tree_type == "half":
        for _ in range(MASK_NUM):
            Lc = torch.tensor([MASK_ID for _ in range(num_candidate)], dtype=input_ids.dtype, device=input_ids.device).repeat(batch_size, 1)
            Lcs.append(Lc)
            Lcs_num += Lc.shape[-1]
            Lcs_num_accumulate.append(Lcs_num)
            num_candidates.append(num_candidate)
    elif tree_type == "upper-triangle":
        assert num_candidate >= MASK_NUM
        for m in range(MASK_NUM):
            Lc = torch.tensor([MASK_ID for _ in range(num_candidate - m)], dtype=input_ids.dtype, device=input_ids.device).repeat(batch_size, 1)
            Lcs.append(Lc)
            Lcs_num += Lc.shape[-1]
            Lcs_num_accumulate.append(Lcs_num)
            num_candidates.append(num_candidate - m)
    else:
        raise ValueError
        
    mask_c, activate_idx = make_tree_attention(num_candidate, tree_type, dtype=infer_dtype, device=input_ids.device, mask_num=MASK_NUM, return_index=True)

    past_key_values = None
    new_generate_token = torch.empty(1)
    while True:
        input_ids_idx = input_ids.shape[-1]

        # create input token ids
        tmp = torch.hstack(Lcs + [Lm for _ in range(Lcs_num + 1)]) 
        input_ids_extend = torch.hstack([input_ids, tmp])

        # create attention mask
        combined_attention_mask = _make_causal_mask(
            input_ids_extend.shape,
            infer_dtype,
            device=input_ids_extend.device,
            past_key_values_length=0
        )
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            torch.cat([attention_mask, attention_mask.new_ones((batch_size, Lcs_num + MASK_NUM * (Lcs_num + 1)))], dim=-1),
            infer_dtype,
            tgt_len=input_ids_extend.shape[-1]
        ).to(input_ids_extend.device)
        expanded_attn_mask[:, :, input_ids_idx : input_ids_idx + Lcs_num, input_ids_idx : input_ids_idx + Lcs_num] = mask_c
        expanded_attn_mask[:, :, input_ids_idx + Lcs_num:, input_ids_idx : input_ids_idx + Lcs_num] = torch.finfo(infer_dtype).min
        for i in range(1, Lcs_num + 1):
            for _idx in activate_idx[i - 1]:
                expanded_attn_mask[:, :, input_ids_idx + Lcs_num + i * MASK_NUM : input_ids_idx + Lcs_num + (i + 1) * MASK_NUM, input_ids_idx + _idx] = 0
            expanded_attn_mask[:, :, input_ids_idx + Lcs_num + i * MASK_NUM:, input_ids_idx + Lcs_num + (i - 1) * MASK_NUM : input_ids_idx + Lcs_num + i * MASK_NUM] = torch.finfo(infer_dtype).min
        attention_mask_extend = expanded_attn_mask + combined_attention_mask

        # create position ids
        position_ids = (attention_mask_extend==0).sum(axis=-1).squeeze(0) - 1

        if "falcon" in model_type:
            attention_mask_extend = attention_mask_extend != 0

        # run LLM
        with torch.no_grad():
            outputs = model(
                input_ids_extend,
                attention_mask=attention_mask_extend,
                position_ids=position_ids,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=USE_CACHE,
                infer_with_prefix=True,
                mask_num_all=MASK_NUM*(Lcs_num+1),
                decoding_method="tree2"
            )

        logits = torch.softmax(outputs.logits, dim=-1)  # normalized logits

        new_generate_token = 0
        select_idx = input_ids_idx + Lcs_num
        select_candidates = []
        next_token_logit = logits[:, input_ids_idx - 1, :]

        for idx in range(MASK_NUM):
            next_token_idx = torch.argmax(next_token_logit, dim=-1)
            condition = False
            for c in range(num_candidates[idx]):
                if next_token_idx == Lcs[idx][:, c]:
                    condition = True
                    break

            if condition:
                next_tokens = next_token_idx
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                new_generate_token += 1
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
                if unfinished_sequences.max() == 0:
                    break

                if tree_type == "full":
                    next_shift = 0
                    for s in range(len(select_candidates)):
                        next_shift += select_candidates[s] * num_candidate ** (len(select_candidates) - s)
                    next_token_logit = logits[:, input_ids_idx + Lcs_num_accumulate[idx] + next_shift + c, :]
                    select_idx = input_ids_idx + Lcs_num + MASK_NUM + (Lcs_num_accumulate[idx] + next_shift + c) * MASK_NUM
                    select_candidates.append(c)
                elif tree_type in ("half", "upper-triangle"):
                    next_token_logit = logits[:, input_ids_idx + Lcs_num_accumulate[idx] + c, :]
                    select_idx = input_ids_idx + Lcs_num + MASK_NUM + (Lcs_num_accumulate[idx] + c) * MASK_NUM
                    if c > 0:
                        break
            else:
                break

        next_tokens = torch.argmax(next_token_logit, dim=-1)
        
        # generate new candidate tokens
        logits_candidate = logits[:, select_idx: select_idx + MASK_NUM, :]
        _, indice = torch.topk(logits_candidate, k=max(num_candidates), dim=-1)  # indice: [1, MASK_NUM, num_candidate]
        for idx in range(MASK_NUM):
            if tree_type == "full":
                Lcs[idx] = indice[:, idx].repeat(1, num_candidate**idx)
            elif tree_type in ("half", "upper-triangle"):
                Lcs[idx] = indice[:, idx, :num_candidates[idx]]

        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        new_generate_token += 1

        unfinished_sequences = unfinished_sequences.mul(next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
        if unfinished_sequences.max() == 0 or input_ids.shape[-1]-prompt_len>=MAX_NEW_TOKENS:
            break

    infer_time = time.perf_counter() - infer_time
    responses = tokenizer.batch_decode(input_ids[:, batch_inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    print(f"No.{data_idx}: Mask greedy decoding time {infer_time:.3f} speed in tokens/s {(input_ids.shape[-1]-batch_inputs['input_ids'].shape[-1])/infer_time:.3f}: \n{responses[0]}")
    print("\n")
    res_dict["infer_time_mask"].append(infer_time)
    res_dict["content_mask"].append(responses[0])
    return token_dict, res_dict


def get_performance(out_text, df_res, df_token):
    for column in df_res.columns:
        if "time" in column:
            out_text += f"Infer time {column}: {df_res[column].mean():.3f}\n"
    time_percentage = df_res['infer_time_mask'].mean() / df_res['infer_time_direct'].mean()
    out_text += f"infer time mask {df_res['infer_time_mask'].mean():.3f} direct {df_res['infer_time_direct'].mean():.3f} percentage {time_percentage:.3f} {1 / time_percentage:.3f}\n"

    if df_token is not None:
        num_infer = len(df_token)
        num_token = df_token['token'].apply(lambda x: len(json.loads(x))).sum()
        out_text += f"num infer {num_infer} num tokens {num_token} percentage {num_infer / num_token:.3f} {num_token / num_infer:.3f}\n"

        df_token['token_len'] = df_token['token'].apply(lambda x: len(json.loads(x)))
        for idx, value in df_token['token_len'].value_counts().items():
            out_text += f"{idx} count {value} percentage {value / len(df_token):.3f}\n"
        out_text += f"avg num of tokens {df_token['token_len'].mean():.3f}\n"

    if 'answer' in df_res.columns:
        rouger = Rouge()
        scores = rouger.get_scores(df_res['content_mask'].values, df_res['answer'].values)[0]
        for level, data_dict in scores.items():
            for k, v in data_dict.items():
                out_text += f"{level}: {k}: {v:.3f}\n"

    if 'category' in df_res.columns:
        for category, sub_df in df_res.groupby('category'):
            time_percentage = sub_df['infer_time_mask'].mean() / sub_df['infer_time_direct'].mean()
            out_text += f"{category} infer time mask {sub_df['infer_time_mask'].mean():.3f} direct {sub_df['infer_time_direct'].mean():.3f} percentage {time_percentage:.3f} {1 / time_percentage:.3f}\n"
    return out_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="eval model")
    parser.add_argument("--llm_dir", type=str, default='./output/llama-2-7b-chat-prompt-16-mask-3_12-30-11-30')
    parser.add_argument("--dataset", type=str, default="xsum", help="xsum, mt-bench, cip, human_eval")
    parser.add_argument("--mask_id", type=int, default=32000)
    parser.add_argument("--mask_num", type=int, default=3)
    parser.add_argument("--mask_diff", type=str, default='true')
    parser.add_argument("--do_sample", type=str, default='false')
    parser.add_argument("--use_cache", type=str, default='false')
    parser.add_argument("--model_type", type=str, default='llama')
    parser.add_argument("--save_data", type=str, default='false')
    parser.add_argument("--template_type", type=str, default='full', help="full, short")
    parser.add_argument("--decoding", type=str, default='tree2', help="SPACE, tree1, or tree2")
    parser.add_argument("--tree_type", type=str, default='upper-triangle', help="full, half, or upper-triangle, only valid in decoding method tree1/tree2")
    parser.add_argument("--num_candidate", type=int, default=3, help="only valid in decoding method tree1/tree2")
    args = parser.parse_args()
    if args.llm_dir[-1] != '/':
        args.llm_dir += '/'

    llm_dir = args.llm_dir
    MASK_ID = args.mask_id
    MASK_NUM = args.mask_num
    do_sample = args.do_sample.lower() == 'true'
    USE_CACHE = args.use_cache.lower() == 'true'

    decoding_method = args.decoding.lower()
    tree_type = args.tree_type.lower()
    num_candidate = int(args.num_candidate)
    assert decoding_method in ("space", "tree1", "tree2")
    assert num_candidate > 0
    
    print(f"Mask num {MASK_NUM} ID {MASK_ID} Do sample {do_sample} Use KVCache {USE_CACHE} Dir {llm_dir}")
    print(f"Decoding method {decoding_method}, tree type {tree_type}, number of candidates {num_candidate}")

    data_list = []
    if args.dataset == 'xsum':
        # MAX_SAMPLE = 100
        test_file_name = "xsum-test"
        with open(os.path.join(ROOT, 'testsets/xsum-test.jsonl'), 'r') as f:
            for d in f.readlines():
                data_dict = json.loads(d)
                data_list.append({'input_text': [f"Document: {data_dict['document']}\nBased on the previous text, provide a brief single summary:"], 'answer': data_dict['summary']})
                # if len(data_list) == MAX_SAMPLE:
                #     break
        additional_col = ['answer']
        MAX_NEW_TOKENS = 256
    elif args.dataset == 'mt-bench':
        test_file_name = "mt-bench"
        with open(os.path.join(ROOT, 'testsets/mt-bench-question.jsonl'), 'r') as f:
            for d in f.readlines():
                data_dict = json.loads(d)
                data_list.append({"input_text": data_dict['turns'], "category": data_dict['category'], "id": data_dict['question_id']})
        additional_col = ["category", "id"]
        MAX_NEW_TOKENS = 512
    elif args.dataset == 'cip':
        # MAX_SAMPLE = 100
        test_file_name = "cip"
        with open(os.path.join(ROOT, 'testsets/chatbot_instruction_prompts_test.jsonl'), 'r') as f:
            for d in f.readlines():
                data_dict = json.loads(d)
                data_list.append({"input_text": [data_dict['prompt']]})
                # if len(data_list) == MAX_SAMPLE:
                #     break
        additional_col = []
        MAX_NEW_TOKENS = 512
    elif args.dataset == 'human_eval':
        test_file_name = "human_eval"
        with open(os.path.join(ROOT, 'testsets/humaneval-x-python.jsonl'), 'r') as f:
            for d in f.readlines():
                data_dict = json.loads(d)
                data_list.append({"input_text": ["Complete the following python code\n"+data_dict['prompt']],
                                  "task_id": data_dict['task_id'],
                                  "declaration": data_dict['declaration'],
                                  "canonical_solution": data_dict['canonical_solution'],
                                  "test": data_dict['test'],
                                  "example_test": data_dict['example_test'] })
        additional_col = ["task_id", "declaration", "canonical_solution", "test", "example_test"]
        MAX_NEW_TOKENS = 1024
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    
    dataset = Dataset.from_list(data_list)
    GENERATION_CONFIG = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": 1,
        "do_sample": do_sample,
        "temperature": 0.01,
        "repetition_penalty": 1.0,
        "top_p": 0.8,
        "use_cache": USE_CACHE
    }

    if 'llama' in args.model_type.lower():
        model = LlamaForCausalLM.from_pretrained(llm_dir, device_map="auto").half()
        system_prompt = ""
    elif "vicuna" in args.model_type.lower():
        model = LlamaForCausalLM.from_pretrained(llm_dir, device_map="auto").half()
        system_prompt = ""
    elif "falcon" in args.model_type.lower():
        model = FalconForCausalLM.from_pretrained(llm_dir, device_map="auto").half()
        system_prompt = ""
    else:
        model = AutoModelForCausalLM.from_pretrained(llm_dir, device_map="auto", trust_remote_code=True).half()
        system_prompt = ""

    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True, padding_side='left', add_bos_token=True)
    
    test_file_name += f"_mask_num_{MASK_NUM}_id_{MASK_ID}_do_sample_{do_sample}_use_cache_{USE_CACHE}"
    test_file_name += f"_candidate_{num_candidate}_tree_{tree_type}_type_{decoding_method}"

    token_dict = {f"mask_idx_{i}": [] for i in range(MASK_NUM)}
    token_dict.update({f"mask_val_{i}": [] for i in range(MASK_NUM)})
    token_dict.update({"token": [], "idx": []})
    token_dict.update({f"Lc_{i}": [] for i in range(MASK_NUM)})
    res_dict = {
        "infer_time_hf": [],
        "infer_time_direct": [],
        "infer_time_mask": [],
        "input_text": [],
        "content_hf": [],
        "content_direct": [],
        "content_mask": []
    }
    res_dict.update({col: [] for col in additional_col})

    if "llama" in args.model_type.lower():
        # case 1 - default template
        if args.template_type.lower() == "full":
            prompt = "[INST] {{query}} [/INST] "
            prefix = "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
            system = (
                "You are a helpful, respectful and honest assistant. "
                "Always answer as helpfully as possible, while being safe.  "
                "Your answers should not include any harmful, unethical, "
                "racist, sexist, toxic, dangerous, or illegal content. "
                "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                "If a question does not make any sense, or is not factually coherent, "
                "explain why instead of answering something not correct. "
                "If you don't know the answer to a question, please don't share false information."
            )
            query = prefix.replace("{{system}}", system, 1) + "{}"
            query = prompt.replace("{{query}}", query, 1)
            template_first = query
            template_others = "[INST] {} [/INST] "

        # case 2, non-system-info template (succeed for llama2)
        elif args.template_type.lower() == "short":
            prompt = "[INST] {{query}} [/INST] "
            query = prompt.replace("{{query}}", "{}", 1)
            template_first = query
            template_others = "[INST] {} [/INST] "
        
        else:
            raise ValueError(f"invalid template: {args.template_type}")

    elif "vicuna" in args.model_type.lower():
        # case 1 - default template
        if args.template_type.lower() == "full":
            prompt = "USER: {{query}} ASSISTANT:"
            prefix = "{{system}}"
            system = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
            )
            query = prefix.replace("{{system}}", system, 1)
            query = " ".join([query, prompt.replace("{{query}}", "{}", 1)])
            template_first = query
            template_others = "USER: {} ASSISTANT:"
        
        # case 2, non-system-info template
        elif args.template_type.lower() == "short":
            prompt = "USER: {{query}} ASSISTANT:"
            query = prompt.replace("{{query}}", "{}", 1)
            template_first = query
            template_others = "USER: {} ASSISTANT:"
        
        else:
            raise ValueError(f"invalid template: {args.template_type}")
    
    elif "falcon" in args.model_type.lower():
        template = "User: {}\nAssistant:"
        print(f"using template {template}")

    else:
        raise ValueError(f"invalid model_type: {args.model_type}")

    data_idx = 0
    input_texts_list = []
    for _, data_dict in tqdm(enumerate(dataset)):
        history = system_prompt
        for input_idx, input_texts in enumerate(data_dict['input_text']):

            if "llama" in args.model_type.lower():
                if input_idx == 0:
                    input_texts = template_first.format(input_texts)
                else:
                    input_texts = template_others.format(input_texts)
                model_input_texts = ''.join([history, input_texts])
            elif "vicuna" in args.model_type.lower():
                if input_idx == 0:
                    input_texts = template_first.format(input_texts)
                else:
                    input_texts = template_others.format(input_texts)
                model_input_texts = ''.join([history, input_texts])
            elif "falcon" in args.model_type.lower():
                input_texts = template.format(input_texts)
                model_input_texts = ''.join([history, input_texts])

            input_tokens = tokenizer(model_input_texts, return_tensors='pt', add_special_tokens=True).to(model.device)
            print(f"model input text:\n{model_input_texts}")
            print("\n")

            res_dict['input_text'].append(model_input_texts)
            # res_dict = decode_alg_hf(input_tokens, res_dict)
            res_dict = decode_alg_direct(input_tokens, res_dict, do_sample=do_sample)
            if decoding_method == "space":  # fully tree-like decoding with 1 draft candidate for each mask token (k = 1), as described in Table 4 in our paper
                token_dict, res_dict = decode_alg_mask_space(args.model_type.lower(), input_tokens, res_dict, token_dict, data_idx, do_sample=do_sample, save_data=(args.save_data.lower()=='true'))
            elif decoding_method == "tree1":  # tree-like decoding where the draft candidatas derivated from the same mask tokens share a universal group of mask tokens for future predictions, deprecated due to performance reasons
                decode_alg_mask_tree1(args.model_type.lower(), input_tokens, res_dict, token_dict, data_idx, do_sample=do_sample, num_candidate=num_candidate, tree_type=tree_type)
            elif decoding_method == "tree2":  # tree-like decoding where each draft candidata is equiped with an indivisual group of mask tokens for future predictions, employed in our method
                decode_alg_mask_tree2(args.model_type.lower(), input_tokens, res_dict, token_dict, data_idx, do_sample=do_sample, num_candidate=num_candidate, tree_type=tree_type)

            if "llama" in args.model_type.lower():
                history = model_input_texts + res_dict['content_mask'][-1] + "</s><s>"
            elif "vicuna" in args.model_type.lower():
                history = model_input_texts + res_dict['content_mask'][-1] + "</s><s>"
            elif "falcon" in args.model_type.lower():
                history = model_input_texts + res_dict['content_mask'][-1] + "<|endoftext|>\n"

            for col in additional_col:
                res_dict[col].append(data_dict[col])
            data_idx += 1

    if args.save_data.lower() == 'true':
        df_token = pd.DataFrame(token_dict)
        df_token.to_csv(f"{llm_dir + test_file_name}_token_dict.csv", index=False)
    else:
        df_token = None

    df_res = pd.DataFrame({k: v for k, v in res_dict.items() if len(v) > 0})
    df_res.to_csv(f"{llm_dir + test_file_name}_res_dict.csv", index=False)

    # show performance
    out_text = ""
    out_text = get_performance(out_text, df_res, df_token)
    out_text += '\n' + '-' * 50 + '\n'
    out_text += f"{args.dataset} Mask num {MASK_NUM} ID {MASK_ID} Do sample {do_sample} Use KVCache {USE_CACHE} Dir {llm_dir}\n"
    out_text += '\n' + '#' * 50 + '\n'

    print(out_text)

    with open(f"{llm_dir + test_file_name}.txt", 'w') as f:
        f.writelines(out_text)