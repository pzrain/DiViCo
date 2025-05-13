#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

MODEL_VERSION=vicuna-v1-3-15b
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########
gpu_vis=1 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=29571

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT disen_llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version $PROMPT_VERSION \
    --data_path LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder LLaVA-Pretrain/images \
    --tune_mm_mlp_adapter True \
    --freeze_mm_mlp_adapter True \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --pretrain_mm_mlp_adapter /villa/panzr/experiment/Disen_LLava/liuhaotian/llava-v1.5-13b/mm_projector.bin \
    --bf16 False \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pretrain-1119-1 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4360 \
    --save_total_limit 10 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
