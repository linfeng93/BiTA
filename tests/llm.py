import os
import time
from typing import List, Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#GENERATION_CONFIG = {"max_new_tokens": 32, "num_beams": 1, "do_sample": False, "temperature": 0.01, "repetition_penalty": 1.1}
GENERATION_CONFIG = {"max_new_tokens": 32, \
                         "num_beams": 1, \
                         "do_sample": False, \
                         "temperature": 0.01,\
                         "repetition_penalty": 1.1,\
                         "top_p":0.8}

def make_chat(model_config: Dict):
    llm_dir = model_config['llm_dir']
    tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True, padding_side='left',add_bos_token=True)
    tokenizer.pad_token = tokenizer.unk_token

#    if not hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
#        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(llm_dir, trust_remote_code=True,device_map="auto").half()
    model = model.eval()

    def get_choice_ids(tokenizer, choices):
        choice_ids = []
        for i, choice in enumerate(choices):
            cur_choice_ids = tokenizer.encode(choice, add_special_tokens=False)
            assert len(cur_choice_ids) == 1
            cur_choice_ids = cur_choice_ids[0]
            choice_ids.append(cur_choice_ids)
        return choice_ids

    def get_choice_probs(input_texts: List[str], choices: List[str]) -> List[str]:
        choice_ids = get_choice_ids(tokenizer, choices)
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to("cuda")

        infer_time = time.perf_counter()
        results = model(**inputs)
        infer_time = time.perf_counter() - infer_time

        logits = results.logits
        if tokenizer.padding_side == "left":
            choice_probs = logits[:, -1, choice_ids]
        elif tokenizer.padding_side == "right":
            choice_probs = []
            for i in range(len(logits)):
                start_idx = inputs["attention_mask"][i].sum() - 1
                choice_probs.append(logits[i, start_idx, choice_ids])
            choice_probs = torch.stack(choice_probs, dim=0)

        choice_probs = choice_probs.softmax(dim=-1)
        pred_choice_ids = torch.argmax(choice_probs, dim=-1)
        return [choices[idx] for idx in pred_choice_ids], infer_time

    def chat(input_texts: List[str], params: Optional[Dict] = None) -> Dict:
        params = params or {}
        params = {**GENERATION_CONFIG, **params}
        temp = []
        for inp in input_texts:
            temp.append('<|User|>'+inp+'<|Assistant|>')
        input_texts = temp
        choices = params.pop("choices", None)
        if choices:
            with torch.no_grad():
                results, infer_time = get_choice_probs(input_texts, choices)
                return {'output_texts': results, 'infer_time': infer_time}
        #batch_input_ids = tokenizer(input_texts, padding=True, return_tensors='pt').to("cuda")
        batch_input_ids = tokenizer(input_texts, padding=True,return_tensors='pt', add_special_tokens=True).to("cuda")
        batch_input_ids.pop("token_type_ids")

        infer_time = time.perf_counter()
        batch_response_ids = model.generate(**batch_input_ids, **params)
        infer_time = time.perf_counter() - infer_time

        batch_response_ids=[q[len(i):-1] for i,q in zip(batch_input_ids['input_ids'],batch_response_ids)]
        batch_responses = tokenizer.batch_decode(batch_response_ids, skip_special_tokens=True)

        return {'output_texts': batch_responses, 'infer_time': infer_time}

    return chat


def print_result(result):
    for output_text in result['output_texts']:
        print(f'output: {output_text}')


if __name__ == '__main__':
    llm_dir = os.path.dirname(os.path.abspath(__file__))
    print('*'*8)
    print(llm_dir)
    config = { "llm_dir": llm_dir}
    chat = make_chat(config)

    result = chat(['问题：乒乓球的起源地 回答：'])
    print_result(result)
    result = chat(['who are you?'])
    print_result(result)
