#!/bin/bash

MODEL_TYPE=llama
MASK_ID=32000
TEST_DIR=./output/llama-2-7b-chat-prompt-16-mask-3_01-04-20-14/

for mask_num in 3
do
  for dataset in cip
  do
    echo "evaluating $mask_num $dataset"
    CUDA_VISIBLE_DEVICES=2 python tests/eval_infer.py \
      --llm_dir=$TEST_DIR \
      --dataset=$dataset \
      --do_sample=false \
      --use_cache=false \
      --model_type=$MODEL_TYPE \
      --mask_num=$mask_num \
      --mask_id=$MASK_ID \
      --mask_diff=true \
      --save_data=false \
      --template_type=full \
      --decoding=tree2 \
      --tree_type=upper-triangle \
      --num_candidate=5
    done
done
echo "done"