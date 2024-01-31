#!/bin/bash

ROOT_DIR=.

mask_num=4
prompt_num=16
echo "training $mask_num $prompt_num"
EXPERIMENT_NAME="vicuna-33b-chat-prompt-$prompt_num-mask-$mask_num"
DATESTR=$(date +"%m-%d-%H-%M")
EXPERIMENT_NAME=${EXPERIMENT_NAME}_${DATESTR}
SAVE_PATH=$ROOT_DIR/output/${EXPERIMENT_NAME}

config_json="./ds_config.json"
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": 2,
  "train_batch_size": "auto",
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 1e7,
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": {
    "enabled": true
  },
  "fp16": {
    "enabled": false,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "zero_allow_untested_optimizer": true,
  "flops_profiler": {
    "enabled": true,
    "profile_step": 50,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": "flops_profiler.log"
  },
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
EOT

DISTRIBUTED_ARGS="-H hostfile --master_addr $PAI_HOST_IP_exec_0 --master_port $PAI_PORT_LIST_exec_0_master"

export LOGLEVEL=WARNING
ENV='NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=front1 NCCL_DEBUG=INFO'
export $ENV

deepspeed $DISTRIBUTED_ARGS src/train_bash.py \
  --deepspeed ${config_json} \
  --ddp_timeout 18000 \
  --preprocessing_num_workers 16 \
  --do_train \
  --stage sft \
  --finetuning_type pt2 \
  --model_name_or_path $ROOT_DIR/checkpoints/vicuna-33b-v1.3-hf \
  --dataset assembled_prompt_gen-vicuna33b,assembled_prompt2_gen-vicuna33b \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
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
  --template vicuna \
  --cutoff_len 2048 \
  --mask_num $mask_num \
  --mask_diff True \
  --mask_id 32000 \
  --prompt_num $prompt_num \
  --num_train_epochs 1.0 \
  --resample_times_each_epoch 4.0 \
  --max_efficient_groups 64

cp $0 $SAVE_PATH
echo "done"