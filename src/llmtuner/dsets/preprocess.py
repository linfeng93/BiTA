import random

import tiktoken
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union
from itertools import chain

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.template import get_template_and_fix_tokenizer

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer
    from llmtuner.hparams import DataArguments


def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo"],
    finetuning_type: Literal["pt", "pt2"],
) -> Union["Dataset", "IterableDataset"]:
    column_names = list(next(iter(dataset)).keys())
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        for i in range(len(examples["prompt"])):
            query, response = examples["prompt"][i], examples["response"][i]
            query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
            history = examples["history"][i] if "history" in examples else None
            system = examples["system"][i] if "system" in examples else None
            template_tag = examples["template_tag"][i] if "template_tag" in examples else None
            yield query, response, history, system, template_tag

    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_eos_token"): # for LLaMA tokenizer
            setattr(tokenizer, "add_eos_token", True)

        tokenized_examples = tokenizer(examples["prompt"], **kwargs)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    # original implementation for pt/pt2, deprecated currently
    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:

        if finetuning_type in ("pt", "pt2"):
            model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            if finetuning_type == "pt2":
                model_inputs["freeze_num"] = []

            for query, response, history, system, template_tag in construct_example(examples):
                input_ids, labels = [], []
                attention_mask = []
                for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                    tokenizer, query, response, history, system, template_tag=template_tag
                )):
                    total_len = len(source_ids) + len(target_ids)
                    max_source_len = int(data_args.cutoff_len * (len(source_ids) / total_len))
                    max_target_len = int(data_args.cutoff_len * (len(target_ids) / total_len))

                    if len(source_ids) > max_source_len:
                        source_ids = source_ids[:max_source_len]
                    if len(target_ids) > max_target_len:
                        target_ids = target_ids[:max_target_len]

                    if data_args.train_on_prompt:
                        source_mask = source_ids
                    elif turn_idx != 0 and template.efficient_eos:  # In this work, template.efficient_eos always keeps False
                        source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                    else:
                        source_mask = [IGNORE_INDEX] * len(source_ids)

                    mask_num_for_training = data_args.mask_num + 1  # Ensure that there are mask_num mask tokens involved in the loss calculation in the forward function
                    if data_args.mask_num > 0 and len(target_ids) >= mask_num_for_training:
                        start_idx = random.randint(0, len(target_ids) - mask_num_for_training)

                        new_input_mask_ids = [data_args.mask_id + i for i in range(data_args.mask_num)] if data_args.mask_diff else [data_args.mask_id] * data_args.mask_num
                        new_input_mask_ids.append(new_input_mask_ids[-1])  # append a placeholder
                        new_input_ids = target_ids[:start_idx] + new_input_mask_ids
                        new_target_ids = [IGNORE_INDEX] * start_idx + target_ids[start_idx : start_idx + mask_num_for_training]

                        input_ids += source_ids + new_input_ids
                        labels += source_mask + new_target_ids
                        attention_mask.extend([1] * (len(source_ids + new_input_ids) - 1) + [0])  # exclude the impact of the appended placeholder
                        break   # only consider one-turn prompt-response pairs
                    else:
                        input_ids += source_ids
                        labels += source_mask
                        attention_mask.extend([1] * len(source_ids))

                if template.efficient_eos:
                    input_ids += [tokenizer.eos_token_id]
                    labels += [tokenizer.eos_token_id]
                    attention_mask.extend([1] * len([tokenizer.eos_token_id]))

                if len(input_ids) > data_args.cutoff_len:
                    input_ids = input_ids[:data_args.cutoff_len]
                    labels = labels[:data_args.cutoff_len]
                    attention_mask = attention_mask[:data_args.cutoff_len]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append(attention_mask)
                model_inputs["labels"].append(labels)

                if finetuning_type == "pt2":
                    learnable_num = 0
                    for input_id in reversed(input_ids):
                        if not input_id < data_args.mask_id:
                            learnable_num += 1
                            continue
                        break
                    freeze_num = len(input_ids) - learnable_num
                    model_inputs["freeze_num"].append(freeze_num)
        
        else:
            raise TypeError("only finetuning_types pt/pt2 are supported.")

        return model_inputs

    # efficient implementation for pt2
    def preprocess_efficient_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        assert not template.efficient_eos, "template.efficient_eos should be False during efficient training"
        if finetuning_type == "pt2":
            model_inputs = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "freeze_num": [],
                "efficient_groups": []
            }

            for query, response, history, system, template_tag in construct_example(examples):
                input_ids, labels, attention_mask, freeze_num, efficient_groups = [], [], [], [], []

                for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                    tokenizer, query, response, history, system, template_tag=template_tag
                )):

                    # only consider one-turn prompt-response pairs
                    if turn_idx > 0:
                        break

                    max_source_len = min(data_args.cutoff_len, len(source_ids))
                    if len(source_ids) > max_source_len:
                        source_ids = source_ids[:max_target_len]
                    max_target_len = data_args.cutoff_len - len(source_ids)
                    if len(target_ids) > max_target_len:
                        target_ids = target_ids[:max_target_len]
                    
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                    if data_args.mask_num > 0 and len(target_ids) > data_args.mask_num:

                        available_num = len(target_ids) - data_args.mask_num
                        if available_num > data_args.max_efficient_groups:
                            _start_idx = random.randint(0, available_num - data_args.max_efficient_groups)
                            start_idx = list(range(_start_idx, _start_idx + data_args.max_efficient_groups))
                        else:
                            start_idx = [_start_idx for _start_idx in range(available_num)]

                        new_input_mask_ids = [data_args.mask_id + i for i in range(data_args.mask_num)] if data_args.mask_diff else [data_args.mask_id] * data_args.mask_num
                        new_input_ids = target_ids[:start_idx[-1]] + new_input_mask_ids * len(start_idx)
                        new_target_ids_tmp = [target_ids[_start_idx + 1 : _start_idx + 1 + data_args.mask_num] for _start_idx in start_idx]
                        new_target_ids = [IGNORE_INDEX] * start_idx[-1] + [_new_target_id_tmp for _new_target_ids_tmp in new_target_ids_tmp for _new_target_id_tmp in _new_target_ids_tmp]

                        input_ids.extend(source_ids + new_input_ids)
                        labels.extend(source_mask + new_target_ids)
                        attention_mask.extend([1] * len(source_ids + new_input_ids))
                        freeze_num.append(len(source_ids + target_ids[:start_idx[-1]]))
                        efficient_groups.append(len(start_idx))

                    else:
                        input_ids.extend(source_ids)
                        labels.extend(source_mask)
                        attention_mask.extend([1] * len(source_ids))
                        freeze_num.append(len(source_ids))
                        efficient_groups.append(0)
                
                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append(attention_mask)
                model_inputs["labels"].append(labels)
                model_inputs["freeze_num"].append(freeze_num)
                model_inputs["efficient_groups"].append(efficient_groups)

        else:
            raise TypeError("only finetuning_types pt2 are supported with efficient training.")

        return model_inputs

    def preprocess_packed_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        for query, response, history, system, _ in construct_example(examples):
            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
                if data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        total_length = len(input_ids)
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        for i in range(0, total_length, block_size):
            model_inputs["input_ids"].append(input_ids[i: i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i: i + block_size])

        return model_inputs

    def preprocess_unsupervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system, _ in construct_example(examples):
            input_ids, labels = template.encode_oneturn(tokenizer, query, response, history, system)

            if template.efficient_eos:
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
            if len(labels) > data_args.cutoff_len:
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_pairwise_dataset(examples):
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
        for query, response, history, system, _ in construct_example(examples):
            prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
            _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

            if template.efficient_eos:
                chosen_ids += [tokenizer.eos_token_id]
                rejected_ids += [tokenizer.eos_token_id]

            total_len = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
            max_source_len = int(data_args.cutoff_len * (len(prompt_ids) / total_len))
            max_target_len = int(data_args.cutoff_len * (max(len(chosen_ids), len(rejected_ids)) / total_len))

            if len(prompt_ids) > max_source_len:
                prompt_ids = prompt_ids[:max_source_len]
            if len(chosen_ids) > max_target_len:
                chosen_ids = chosen_ids[:max_target_len]
            if len(rejected_ids) > max_target_len:
                rejected_ids = rejected_ids[:max_target_len]

            model_inputs["prompt_ids"].append(prompt_ids)
            model_inputs["chosen_ids"].append(chosen_ids)
            model_inputs["rejected_ids"].append(rejected_ids)
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))

        if finetuning_type in ("pt", "pt2") and data_args.mask_num > 0:
            print("inputs:\n{}".format(tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["input_ids"])), skip_special_tokens=False)))
        else:
            print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)))

    def print_pairwise_dataset_example(example):
        print("prompt_ids:\n{}".format(example["prompt_ids"]))
        print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
        print("chosen_ids:\n{}".format(example["chosen_ids"]))
        print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
        print("rejected_ids:\n{}".format(example["rejected_ids"]))
        print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))

    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if stage == "pt":
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_func = preprocess_pretrain_dataset
        print_function = print_unsupervised_dataset_example
    elif stage == "sft" and not training_args.predict_with_generate:
        dataset = dataset.filter(lambda example: example["prompt"] and example["response"])
        if data_args.sft_packing:
            preprocess_func = preprocess_packed_supervised_dataset
        else:
            if data_args.max_efficient_groups > 1:
                preprocess_func = preprocess_efficient_supervised_dataset
            else:
                preprocess_func = preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    elif stage == "rm":
        dataset = dataset.filter(lambda example: example["prompt"] and len(example["response"]) > 1)
        preprocess_func = preprocess_pairwise_dataset
        print_function = print_pairwise_dataset_example
    else:
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_func = preprocess_unsupervised_dataset
        print_function = print_unsupervised_dataset_example

    with training_args.main_process_first(desc="dataset map pre-processing"):
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset"
            )

        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        try:
            print_function(next(iter(dataset)))
        except StopIteration:
            raise ValueError("Empty dataset!")

        return dataset
