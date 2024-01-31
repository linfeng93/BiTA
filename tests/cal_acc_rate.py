import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import json

GENERATION_CONFIG = {"max_new_tokens": 512, \
                         "num_beams": 1, \
                         "do_sample": False, \
                         "temperature": 0.01,\
                         "repetition_penalty": 1.0,\
                         "top_p":0.8}

def decode_alg(input_texts, idx, res_dict):
    # input_texts = ['Human:保持健康的三个提示。\nAssistant:']
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    batch_input_ids = tokenizer(input_texts, padding=True, return_tensors='pt', add_special_tokens=True).to("cuda")[
        'input_ids']
    # print(f"Inputs are {input_texts} input_ids are {batch_input_ids}")

    ## use greedy decoding from HF
    # infer_time = time.perf_counter()
    # batch_response_ids = model.generate(batch_input_ids, **GENERATION_CONFIG)
    # infer_time = time.perf_counter() - infer_time
    # batch_response_ids = [q[len(i):-1] for i, q in zip(batch_input_ids, batch_response_ids)]
    # batch_responses = tokenizer.batch_decode(batch_response_ids, skip_special_tokens=True)
    # print(f"Generate greedy decoding time {infer_time:.3f}: {batch_responses}")

    eos_token_id = tokenizer.eos_token_id
    # input_ids = batch_input_ids
    # infer_time = time.perf_counter()
    # past_key_values = None
    # attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    # while True:
    #     position_ids = attention_mask.long().cumsum(-1) - 1
    #     position_ids.masked_fill_(attention_mask == 0, 1)
    #     if past_key_values:
    #         position_ids = position_ids[:, -1].unsqueeze(-1)
    #
    #     outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, return_dict=True, use_cache=False)
    #     next_token_logits = outputs.logits[:, -1, :]
    #     # past_key_values = outputs.past_key_values
    #
    #     next_token = torch.argmax(next_token_logits, dim=-1)
    #
    #     input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
    #     attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    #
    #     if next_token[0] == eos_token_id:
    #         break
    #
    # infer_time = time.perf_counter() - infer_time
    # responses = tokenizer.batch_decode([q[len(i):-1] for i, q in zip(batch_input_ids, input_ids)], skip_special_tokens=True)
    # print(f"Direct greedy decoding time {infer_time:.3f}: {responses}")
    # assert responses == batch_responses

    input_ids = batch_input_ids
    infer_time = time.perf_counter()

    while True:
        input_ids_extend = torch.cat([input_ids, MASK_ID * input_ids.new_ones((input_ids.shape[0], MASK_NUM))], dim=-1)

        outputs = model(input_ids_extend, return_dict=True, use_cache=False)
        next_token_logits = outputs.logits[:, -MASK_NUM - 1, :]

        next_token = torch.argmax(next_token_logits, dim=-1)

        input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)

        res_dict["token"].append(next_token[0].item())
        res_dict["idx"].append(idx)
        for i in range(MASK_NUM):
            token_logits = outputs.logits[:, -MASK_NUM + i, :]
            token = torch.argmax(token_logits, dim=-1)
            res_dict[f"mask_{i}"].append(token[0].item())

        if next_token[0] == eos_token_id:
            break

    infer_time = time.perf_counter() - infer_time
    responses = tokenizer.batch_decode([q[len(i):-1] for i, q in zip(batch_input_ids, input_ids)],
                                       skip_special_tokens=True)
    print(f"{idx} Direct greedy decoding time {infer_time:.3f}, input {input_texts[0]} output {responses}")
    return res_dict

if __name__ == '__main__':
    llm_dir = '/data/FM/yihanling/sft/output/llama-2-7b-mask-ablation-8_10-22-14-51/'

    MASK_ID = 32002 if 'mask' in llm_dir else 0
    MASK_NUM = 8
    print(f"Mask num {MASK_NUM} ID {MASK_ID} Dir {llm_dir}")

    tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True, padding_side='left', add_bos_token=True)
    model = AutoModelForCausalLM.from_pretrained(llm_dir, trust_remote_code=True, device_map="auto").half()
    model = model.eval()

    # get data
    with open("../data/alpaca_data_zh_51k.json", "r") as f:
        data_list = json.load(f)

    res_dict = {f"mask_{i}": [] for i in range(MASK_NUM)}
    res_dict.update({"token": [], "idx": []})
    for idx, data_dict in enumerate(data_list[:10]):
        input_texts = '\n'.join([data_dict['instruction'], data_dict['input']]) if len(data_dict['input']) > 0 else data_dict['instruction']
        input_texts = f"Human:{input_texts}\nAssistant:"
        res_dict = decode_alg(input_texts, idx, res_dict)

    # input_texts = ['Human:保持健康的三个提示。\nAssistant:']
    # res_dict = decode_alg(input_texts, 0, res_dict)

    # cal accept rate
    df = pd.DataFrame(res_dict)

    accept_res = {f"mask_acc_all_{idx}": 0 for idx in range(MASK_NUM)}
    accept_res.update({f"mask_acc_{idx}": 0 for idx in range(MASK_NUM)})
    # accept_res.update({f"mask_con_all_{idx}": 0 for idx in range(MASK_NUM)})
    # accept_res.update({f"mask_con_{idx}": 0 for idx in range(MASK_NUM)})

    for idx, sub_df in df.groupby("idx"):
        last_match_res = [True] * len(sub_df)
        for i in range(MASK_NUM):
            match_res = sub_df['token'].shift(-i-1) == sub_df[f'mask_{i}']
            # print(f"{idx} mask {i} accept rate {match_res[:-i-1].mean():.3f}")

            condition_match_res = match_res & last_match_res
            # if sum(last_match_res) > 0:
            #     condition_accept_rate = condition_match_res[:-i-1].sum()/sum(last_match_res[:-i-1])
            #     accept_res[f"mask_con_all_{i}"] += sum(last_match_res[:-i-1])
            #     accept_res[f"mask_con_{i}"] += condition_match_res[:-i-1].sum()
            # else:
            #     condition_accept_rate = 0

            accept_res[f"mask_acc_all_{i}"] += len(match_res[:-i - 1])
            accept_res[f"mask_acc_{i}"] += condition_match_res[:-i - 1].sum()
            # print(f"{idx} mask {i} condition accept rate {condition_accept_rate:.3f}")
            last_match_res = condition_match_res

    for i in range(MASK_NUM):
        accept = accept_res[f"mask_acc_{i}"] / accept_res[f"mask_acc_all_{i}"]
        print(f"mask {i} accept rate {accept:.3f}")

        # if accept_res[f"mask_con_all_{i}"] == 0:
        #     accept_con = 0
        # else:
        #     accept_con = accept_res[f"mask_con_{i}"] / accept_res[f"mask_con_all_{i}"]
        # print(f"mask {i} condition accept rate {accept_con:.3f}")

    print(accept_res)



