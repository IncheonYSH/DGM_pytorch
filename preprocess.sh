#!/bin/bash
# Usage: bash preprocess.sh BASE_PATH METADATA_CSV SPLIT_CSV LABEL_CSV OUTPUT_PATH [MIMIC_CSV] [CHEXBERT_CKPT] [DEVICE]
set -e

BASE_PATH="/data/mimic-cxr-jpg/files"
METADATA_CSV="/data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv"
SPLIT_CSV="/data/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv"
LABEL_CSV="/data/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv"
OUTPUT_PATH="./data"
MIMIC_CSV="/home/shyoon/rrg_ttc/forced_generation/mimic_forced_generation.csv"
CHEXBERT_CKPT="/home/shyoon/CheXbert/chexbert.pth"
DEVICE="gpu"

python data_preprocess_all.py \
    --base_path "$BASE_PATH" \
    --metadata_csv_file "$METADATA_CSV" \
    --split_csv_file "$SPLIT_CSV" \
    --label_csv_file "$LABEL_CSV" \
    --output_path "$OUTPUT_PATH" \
    ${MIMIC_CSV:+--mimic_csv "$MIMIC_CSV"} \
    ${CHEXBERT_CKPT:+--chexbert_ckpt "$CHEXBERT_CKPT"} \
    --device "$DEVICE"
