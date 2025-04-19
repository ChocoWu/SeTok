#!/bin/bash




IMAGE_FOLDER=(
    "./data/cc3m/images"  # 5240031
    )

DATA_PATH=(
    "./data/cc3m/cc3m.json"
    )

DATASET_NAME=(
    "cc3m"
)

DATA_MULTIPLE=(
    1
)

echo $IMAGE_FOLDER
echo $DATA_PATH
echo $DATASET_NAME

IMAGE_FOLDER="${IMAGE_FOLDER[*]}"
DATA_PATH="${DATA_PATH[*]}"
DATASET_NAME="${DATASET_NAME[*]}"
DATA_MULTIPLE="${DATA_MULTIPLE[*]}"


echo $IMAGE_FOLDER
echo $DATA_PATH
echo $DATASET_NAME



deepspeed train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable False \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --data_multiple $DATA_MULTIPLE \
    --dataset_name $DATASET_NAME \
    --image_size $IMAGE_SIZE \
    --vision_tower ./pretrained_ckpt/siglip-so400m-patch14-384 \
    --feature_mapper_path_or_name ./pretrained_ckpt/bert-base-uncased \
    --bf16 False \
    --output_dir ./checkpoints/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fp16 False \
    --model_max_length 77 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \



