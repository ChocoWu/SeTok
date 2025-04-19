#!/bin/bash


IMAGE_FOLDER=(
    "./data/LLaVA-Instruct-150K/images"
    )

DATA_PATH=(
    "./data/LLaVA-Instruct-150K/llava_instruct_150k.json"
    )

DATA_MULTIPLE=(
    1
)

DATASET_NAME=(
    "LLaVA150K"
)


IMAGE_FOLDER="${IMAGE_FOLDER[@]}"
DATA_PATH="${DATA_PATH[@]}"
DATASET_NAME="${DATASET_NAME[*]}"
DATA_MULTIPLE="${DATA_MULTIPLE[@]}"


accelerate launch train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./pretrained_ckpt/vicuna-7b-v1.5 \
    --version plain \
    --data_path $DATA_PATH \
    --dataset_name $DATASET_NAME \
    --image_folder $IMAGE_FOLDER \
    --data_multiple $DATA_MULTIPLE \
    --vision_tokenizer setok \
    --vision_tower ./pretrained_ckpt/siglip-so400m-patch14-384 \
    --pretrain_vision_tokenizer ./checkpoints/ \
    --pretrain_vision_detokenizer ./checkpoints/ \
    --mm_in_projector_type mlp2x_gelu \
    --tune_mm_in_mlp_adapter True \
    --mm_out_projector_type mlp2x_gelu \
    --tune_mm_out_mlp_adapter True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --feature_mapper_path_or_name ./pretrained_ckpt/bert-base-uncased \
    --bf16 True \
    --output_dir ./checkpoints/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
