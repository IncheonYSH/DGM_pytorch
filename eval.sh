#!/bin/bash
# Usage: bash eval.sh CHECKPOINT FILE_LIST LABEL_CSV [OUTPUT_DIR] [NUM_SAMPLES] [GPU]
set -e

CHECKPOINT="$1"
FILE_LIST="$2"
LABEL_CSV="$3"
OUTPUT_DIR="${4:-./outputs}"
NUM_SAMPLES="${5:-5}"
GPU="${6:-0}"

python test.py \
    --checkpoint "$CHECKPOINT" \
    --file_list_path "$FILE_LIST" \
    --label_csv_path "$LABEL_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --gpu "$GPU"
