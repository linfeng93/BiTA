#!/bin/bash

MODEL_TYPE=falcon
MASK_ID=65024
TEST_DIR=./output/falcon-40b-chat-prompt-16-mask-4_01-07-04/

for mask_num in 4
do
  for dataset in cip
  do
    echo "evaluating $mask_num $dataset"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tests/eval_infer.py \
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