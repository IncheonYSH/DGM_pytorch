#!/bin/bash
# Usage: bash train.sh TRAIN_LIST VAL_LIST [BATCH_SIZE] [D_STEPS] [GPU]
set -e

TRAIN_LIST="./data/filtered_labeled_train_sd.txt"
VAL_LIST="./data/filtered_labeled_validation_sd.txt"
ARCHITECTURE="new" # orginal or new (original -> CNN, new -> ViT)
BATCH_SIZE=64
D_STEPS=1
GPU=3

python train.py \
    --batch_size "$BATCH_SIZE" \
    --d_steps "$D_STEPS" \
    --gpu "$GPU" \
    --train_file_list "$TRAIN_LIST" \
    --architecture "$ARCHITECTURE" \
    --base_GAN "NSGAN" \
    --target_image_size 256 \
    --lr_scheduler "cosine_decay" \
    --val_file_list "$VAL_LIST"
