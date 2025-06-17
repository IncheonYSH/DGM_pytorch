#!/bin/bash
# Usage: bash train.sh TRAIN_LIST VAL_LIST [BATCH_SIZE] [D_STEPS] [GPU]
set -e

TRAIN_LIST="$1"
VAL_LIST="$2"
BATCH_SIZE="${3:-64}"
D_STEPS="${4:-1}"
GPU="${5:-0}"

python train.py \
    --batch_size "$BATCH_SIZE" \
    --d_steps "$D_STEPS" \
    --gpu "$GPU" \
    --train_file_list "$TRAIN_LIST" \
    --val_file_list "$VAL_LIST"
