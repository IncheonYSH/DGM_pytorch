#!/bin/bash
# Usage: bash preprocess.sh BASE_PATH METADATA_CSV SPLIT_CSV LABEL_CSV OUTPUT_PATH [MIMIC_CSV] [CHEXBERT_CKPT] [DEVICE]
set -e

BASE_PATH="$1"
METADATA_CSV="$2"
SPLIT_CSV="$3"
LABEL_CSV="$4"
OUTPUT_PATH="$5"
MIMIC_CSV="${6:-}"
CHEXBERT_CKPT="${7:-}"
DEVICE="${8:-cpu}"

python data_preprocess_all.py \
    --base_path "$BASE_PATH" \
    --metadata_csv_file "$METADATA_CSV" \
    --split_csv_file "$SPLIT_CSV" \
    --label_csv_file "$LABEL_CSV" \
    --output_path "$OUTPUT_PATH" \
    ${MIMIC_CSV:+--mimic_csv "$MIMIC_CSV"} \
    ${CHEXBERT_CKPT:+--chexbert_ckpt "$CHEXBERT_CKPT"} \
    --device "$DEVICE"
