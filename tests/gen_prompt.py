import os
import pandas as pd
import json
from tqdm import tqdm
import argparse
from datasets import Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate prompt inputs for LLM")
    parser.add_argument("--dataset", type=str, default="assembled", help="assembled, xsum, mt-bench, cip, human_eval")
    parser.add_argument("--model_type", type=str, default='llama')
    parser.add_argument("--output_path", type=str, default='data/alpaca_lima_cip_code_platypus-prompt.jsonl')
    parser.add_argument("--prompt_use_system", action="store_true", default=False)
    args = parser.parse_args()

    data_list = []
    if "assembled" in args.dataset:
        path = 'data/assembled_v2/llama2-7b/alpaca_lima_cip-50k_code_platypus_v2-prompt2-output.jsonl'
    elif "alpaca" == args.dataset:
        path = 'data/alpaca_gpt4_data_en.json'
    elif "lima" == args.dataset:
        path = 'data/lima.json'
    elif "cip" == args.dataset:
        path = 'data/chatbot_instruction_prompts_train_format.jsonl'
    elif "code" == args.dataset:
        path = 'data/code_alpaca_20k.json'
    elif "platypus" == args.dataset:
        path = 'data/open_platypus.jsonl'
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    if os.path.splitext(path)[-1] == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif os.path.splitext(path)[-1] == '.jsonl':
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported filetype {path}")
    
    for _data in data:
        prompt = _data['instruction']
        query = _data['input']
        assert isinstance(prompt, str) and isinstance(query, str)
        if len(prompt) > 0:
            input_text = prompt
            if len(query) > 0:
                input_text = "\n".join([input_text, query])
        else:
            input_text = query
        if len(input_text) > 0:
            data_list.append(
                {"input_text": input_text,
                 "input_raw": {
                    "instruction": prompt,
                    "input": query}})
    
    dataset = Dataset.from_list(data_list)

    system_prompt = ""
    if "llama" in args.model_type.lower():
        # case 1 - default template
        if args.prompt_use_system:
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

        # case 2, non-system-info template
        else:
            prompt = "[INST] {{query}} [/INST] "
            query = prompt.replace("{{query}}", "{}", 1)
            template_first = query
            template_others = "[INST] {} [/INST] "

        print(f"using template 1 {template_first}")
        print(f"using template 2 {template_others}")

    elif "vicuna" in args.model_type.lower():
        # case 1 - default template
        if args.prompt_use_system:
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
        else:
            prompt = "USER: {{query}} ASSISTANT:"
            query = prompt.replace("{{query}}", "{}", 1)
            template_first = query
            template_others = "USER: {} ASSISTANT:"

        print(f"using template 1 {template_first}")
        print(f"using template 2 {template_others}")
    
    elif "falcon" in args.model_type.lower():
        template = "User: {}\nAssistant:"
        print(f"using template {template}")

    else:
        template = "Human:{}\nAssistant:"
        print(f"using template {template}")

    f = open(args.output_path, "w", encoding='utf-8')

    data_idx = 0
    input_texts_list = []
    for data_dict in tqdm(dataset):
        history = system_prompt
        input_texts = data_dict['input_text']
        input_raw = data_dict['input_raw'] if 'input_raw' in data_dict else None

        if "llama" in args.model_type.lower():
            input_texts = template_first.format(input_texts)
            model_input_texts = ''.join([history, input_texts])
            model_input_texts = "<s>" + model_input_texts
        elif "vicuna" in args.model_type.lower():
            input_texts = template_first.format(input_texts)
            model_input_texts = ''.join([history, input_texts])
            model_input_texts = "<s>" + model_input_texts
        elif "falcon" in args.model_type.lower():
            input_texts = template.format(input_texts)
            model_input_texts = ''.join([history, input_texts])
        else:
            input_texts = template.format(input_texts)
            model_input_texts = '\n'.join([history, input_texts]).strip()

        if data_idx % 1000 == 0:
            print(f"model input text:\n{model_input_texts}")
            print("\n")

        line = {}
        line["model_input_texts"] = model_input_texts
        if input_raw is not None:
            line["instruction"] = input_raw["instruction"]
            line["input"] = input_raw["input"]
        line["template_tag"] = "full" if args.prompt_use_system else "short"

        f.write(json.dumps(line, ensure_ascii=False)+'\n')
        data_idx += 1

    f.close()
    print("done.")