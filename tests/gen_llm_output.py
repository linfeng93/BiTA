import os
import sys
import json
import requests
import copy
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool


def multi_process_data(prompts, post_template, url, output_path, num_process, error_logging="mp_error.log"):
     
    def call_back_func_write_failures(output):
        with open(error_logging, 'a+') as f:
            for line in output:
                f.write(str(line)+'\n')

    p = Pool(num_process)
    part_content_len = len(prompts) // num_process
    last_len = len(prompts) % num_process
    for i in range(num_process):
        if i < last_len:
            part_content = prompts[(part_content_len + 1) * i : (part_content_len + 1) * (i + 1)]
        else:
            part_content = prompts[(part_content_len * i + last_len) : (part_content_len * (i + 1) + last_len)]
        p.apply_async(process_data, args=(part_content, post_template, url, output_path, i,), callback=call_back_func_write_failures)
        # process_data(part_content, post_template, url, output_path, i)
    
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocesses done.")


def process_data(part_content, post_template, url, output_path, idx):
      
    failed_list = []
    for content in tqdm(part_content):

        input_info = copy.deepcopy(post_template)
        input_info["inputs"] = content["model_input_texts"]

        try_times = 3
        while try_times > 0:
            flag, output_text = post(url, input_info)
            if flag:
                break
            try_times -= 1

        if flag:
            content["output"] = output_text
            write_to_file(output_path.format(idx), content)
        else:
            failed_list.append(content)

    return failed_list

    
def post(url, input_info):
    response = requests.post(url, json=input_info)
    flag = response.status_code == 200
    output_text = response.json()["generated_text"] if flag else ""
    return flag, output_text
    

def write_to_file(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data, ensure_ascii=False)+'\n')


if __name__ == "__main__":

    input_prompt = sys.argv[1]
    output_dir = sys.argv[2]
    num_process = int(sys.argv[3])
    ip = int(sys.argv[4])
    error_logging = "mp_error.log"

    url = f"http://{ip}/generate"
    post_template = {
        "inputs": None,
        "parameters": {
            "best_of": 1,
            "decoder_input_details": False,
            "details": False,
            "do_sample": False,
            "max_new_tokens": 1600,
            "repetition_penalty": 1.0,
            "return_full_text": False,
            "seed": 2023,
            "stop": [
                "photographer"
            ],
            "temperature": 0.01,
            "top_k": 10,
            "top_p": 0.8,
            "truncate": None,
            "typical_p": 0.95,
            "watermark": False
        }
    }
    
    prompts = []
    with open(input_prompt, "r", encoding='utf-8') as f:
        for line in f:
            prompts.append(json.loads(line))

    if len(prompts) > 0:
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.splitext(os.path.basename(input_prompt))[0] + "-output_{}.jsonl"
        output_path = os.path.join(output_dir, output_path)
        print(f"generating output texts of {input_prompt} by LLM")
        print(f"output path: {output_path}")

        print(f"number of input prompts: {len(prompts)}")
        print("start processing...")
        print("\n")

        multi_process_data(prompts, post_template, url, output_path, num_process)