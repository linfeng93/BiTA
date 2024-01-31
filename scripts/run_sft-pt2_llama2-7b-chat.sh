#!/bin/bash

ROOT_DIR=.

for mask_num in 3
do
  for prompt_num in 16
  do
    echo "training $mask_num $prompt_num"

    EXPERIMENT_NAME="llama-2-7b-chat-prompt-$prompt_num-mask-$mask_num"
    DATESTR=$(date +"%m-%d-%H-%M")
    EXPERIMENT_NAME=${EXPERIMENT_NAME}_${DATESTR}
    SAVE_PATH=$ROOT_DIR/output/${EXPERIMENT_NAME}
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file accelerate_config_8_nodes.yaml src/train_bash.py \
      --ddp_timeout 18000 \
      --preprocessing_num_workers 16 \
      --do_train \
      --stage sft \
      --finetuning_type pt2 \
      --model_name_or_path $ROOT_DIR/checkpoints/Llama-2-7b-chat-hf \
      --dataset assembled_prompt_gen,assembled_prompt2_gen \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 2 \
      --lr_scheduler_type cosine \
      --learning_rate 3e-2 \
      --save_strategy steps \
      --save_steps 5000 \
      --logging_steps 10 \
      --output_dir $SAVE_PATH \
      --overwrite_cache \
      --overwrite_output_dir \
      --plot_loss \
      --bf16 \
      --remove_unused_columns False \
      --template llama2 \
      --cutoff_len 2048 \
      --mask_num $mask_num \
      --mask_diff True \
      --mask_id 32000 \
      --prompt_num $prompt_num \
      --num_train_epochs 1.0 \
      --resample_times_each_epoch 4.0 \
      --max_efficient_groups 64

    cp $0 $SAVE_PATH

    done
done
echo "done"
