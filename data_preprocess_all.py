import os
import pandas as pd
import logging
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

def setup_logger(output_path):
    # 로그 파일을 저장할 디렉토리가 없으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 로깅 설정
    log_file = os.path.join(output_path, 'split_generation.log')
    
    # 루트 로거 가져오기
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 핸들러가 이미 설정되어 있지 않은 경우에만 추가
    if not logger.handlers:
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)


class bert_labeler(nn.Module):
    """Minimal CheXbert model definition."""
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None):
        super(bert_labeler, self).__init__()
        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p)
        hidden_size = self.bert.pooler.dense.in_features

        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, input_ids, attention_mask):
        final_hidden = self.bert(input_ids, attention_mask=attention_mask)[0]
        cls_hidden = final_hidden[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))
        return out


def load_chexbert_model_and_tokenizer(checkpoint_path, device="cpu"):
    if isinstance(device, str) and device.lower() == "gpu":
        device = "cuda"
    model = bert_labeler()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


@torch.no_grad()
def get_chexbert_predictions(text, model, tokenizer, device="cpu"):
    text = text.strip()
    if not text:
        return None
    inputs = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    preds = []
    for logits in outputs:
        preds.append(torch.argmax(logits, dim=1).item())
    return preds


def chexbert_label_to_binary(preds):
    """Return 0 for normal and 1 for abnormal based on CheXbert predictions."""
    if preds is None:
        return None
    no_finding_present = (preds[13] == 1)
    other_present = any(p == 1 for p in preds[:13])
    if no_finding_present and not other_present:
        return 0
    return 1

def generate_split_files(base_path, csv_file, output_path):
    # 로거 설정
    setup_logger(output_path)
    
    # CSV 파일 읽기
    logging.info("CSV 파일 읽는 중: %s", csv_file)
    df = pd.read_csv(csv_file)
    
    # 출력 경로가 존재하지 않으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 파일을 열어 쓰기 모드로 준비
    train_file = open(os.path.join(output_path, 'train.txt'), 'w')
    val_file = open(os.path.join(output_path, 'validation.txt'), 'w')
    test_file = open(os.path.join(output_path, 'test.txt'), 'w')
    
    logging.info("Train, validation, test 파일 생성 완료.")

    # CSV 파일의 각 행에 대해 파일 경로를 만듦
    for index, row in df.iterrows():
        dicom_id = row['dicom_id']
        subject_id = row['subject_id']
        study_id = row['study_id']
        split = row['split']
        
        # 주어진 포맷에 맞게 경로 구성
        subject_folder = f"p{str(subject_id)[:2]}/p{subject_id}"
        study_folder = f"s{study_id}"
        file_path = os.path.join(base_path, subject_folder, study_folder, f"{dicom_id}.jpg")
        
        # 파일 경로가 실제로 존재하는지 확인하고 split에 맞게 txt 파일에 경로 추가
        if os.path.exists(file_path):
            logging.info(f"경로 확인됨: {file_path}")
            if split == 'train':
                train_file.write(file_path + '\n')
            elif split == 'validate':
                val_file.write(file_path + '\n')
            elif split == 'test':
                test_file.write(file_path + '\n')
        else:
            logging.warning(f"경로가 존재하지 않습니다: {file_path}")
    
    # 파일 닫기
    train_file.close()
    val_file.close()
    test_file.close()

    logging.info("파일 처리 완료. Train, validation, test 파일 저장 위치: %s", output_path)

def generate_pa_split_files(base_path, metadata_csv_file, split_csv_file, output_path):
    # Metadata CSV 파일 읽기
    logging.info("Metadata CSV 파일 읽는 중: %s", metadata_csv_file)
    df_metadata = pd.read_csv(metadata_csv_file)
    
    # Split CSV 파일 읽기
    logging.info("Split CSV 파일 읽는 중: %s", split_csv_file)
    df_split = pd.read_csv(split_csv_file)
    
    # 'ViewPosition'이 'PA'인 데이터만 선택
    df_pa = df_metadata[df_metadata['ViewPosition'] == 'PA']
    logging.info("PA 이미지 개수: %d", len(df_pa))
    
    # 병합 키 컬럼을 문자열로 변환
    merge_keys = ['dicom_id', 'subject_id', 'study_id']
    for key in merge_keys:
        df_pa[key] = df_pa[key].astype(str)
        df_split[key] = df_split[key].astype(str)
    
    # df_pa와 df_split를 병합
    df_merged = pd.merge(df_pa, df_split, on=merge_keys, how='inner')
    logging.info("병합 후 데이터 개수: %d", len(df_merged))
    
    if df_merged.empty:
        logging.warning("병합 결과가 비어 있습니다. 조건을 확인하세요.")
    
    # 출력 경로가 존재하지 않으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 파일을 열어 쓰기 모드로 준비
    train_file = open(os.path.join(output_path, 'pa_train.txt'), 'w')
    val_file = open(os.path.join(output_path, 'pa_validation.txt'), 'w')
    test_file = open(os.path.join(output_path, 'pa_test.txt'), 'w')
    
    logging.info("PA Train, validation, test 파일 생성 완료.")
    
    # 각 행에 대해 파일 경로를 만듦
    for index, row in df_merged.iterrows():
        dicom_id = row['dicom_id']
        subject_id = row['subject_id']
        study_id = row['study_id']
        split = row['split']
        
        # 파일 경로 구성 (generate_split_files 함수와 동일하게 수정)
        subject_folder = f"p{str(subject_id)[:2]}/p{subject_id}"
        study_folder = f"s{study_id}"
        file_path = os.path.join(base_path, subject_folder, study_folder, f"{dicom_id}.jpg")
        
        # 파일 경로가 실제로 존재하는지 확인하고 split에 맞게 txt 파일에 경로 추가
        if os.path.exists(file_path):
            logging.info(f"경로 확인됨: {file_path}")
            if split == 'train':
                train_file.write(file_path + '\n')
            elif split == 'validate':
                val_file.write(file_path + '\n')
            elif split == 'test':
                test_file.write(file_path + '\n')
            else:
                logging.warning(f"알 수 없는 split 값 '{split}' (파일: {file_path})")
        else:
            logging.warning(f"경로가 존재하지 않습니다: {file_path}")
    
    # 파일 닫기
    train_file.close()
    val_file.close()
    test_file.close()
    
    logging.info("파일 처리 완료. PA Train, validation, test 파일 저장 위치: %s", output_path)

def process_pa_labels(pa_file, mimic_file, output_file):
    """
    pa_file의 파일명과 mimic_file의 파일명을 비교하여 일치하는 파일 경로를 output_file에 저장합니다.
    """
    # pa_file 읽기
    with open(pa_file, 'r') as f:
        pa_paths = [line.strip() for line in f.readlines()]
    # mimic_file 읽기
    with open(mimic_file, 'r') as f:
        mimic_paths = [line.strip() for line in f.readlines()]
    
    # mimic_paths를 파일명(key)과 전체 경로(value)의 딕셔너리로 변환
    mimic_dict = {os.path.basename(path): path for path in mimic_paths}
    
    matched_paths = []
    
    for pa_path in pa_paths:
        pa_filename = os.path.basename(pa_path)
        if pa_filename in mimic_dict:
            matched_paths.append(mimic_dict[pa_filename])
        else:
            logging.warning(f"{pa_filename}이 mimic 파일에서 발견되지 않았습니다.")
            # 필요에 따라 continue 또는 break를 선택
            continue
    
    # output_file에 저장
    with open(output_file, 'w') as f:
        for path in matched_paths:
            f.write(path + '\n')

    logging.info(f"매칭된 경로를 {output_file}에 저장하였습니다.")

def extract_ids_from_path(path):
    parts = Path(path).parts
    # 'p'로 시작하고 뒤에 숫자가 오는 부분을 찾음
    subject_part = [part for part in parts if part.startswith('p') and part[1:].isdigit()]
    # 's'로 시작하고 뒤에 숫자가 오는 부분을 찾음
    study_part = [part for part in parts if part.startswith('s') and part[1:].isdigit()]
    if subject_part and study_part:
        subject_id = subject_part[-1][1:]  # 마지막 'p'로 시작하는 부분 사용
        study_id = study_part[-1][1:]      # 마지막 's'로 시작하는 부분 사용
        return subject_id, study_id
    else:
        return None, None

def get_label_from_row(row):
    # 라벨링 규칙에 따라 이미지의 라벨을 결정
    # Support Devices는 무시
    no_finding = row.get('No Finding', None)
    findings_to_drop = ['subject_id', 'study_id', 'No Finding'] # Support device -> abnormal image
    findings_columns = [col for col in row.index if col not in findings_to_drop]
    other_findings = row[findings_columns].values
    
    # other_findings를 numeric으로 변환
    other_findings = pd.to_numeric(other_findings, errors='coerce')

    if no_finding == 1.0:
        if (-1.0 in other_findings):
            return None  # 제외
        else:
            return 0  # 정상
    elif no_finding == -1.0:
        if (1.0 in other_findings):
            return 1  # 비정상
        else:
            return None  # 제외
    elif pd.isna(no_finding):
        if (1.0 in other_findings):
            return 1 # 비정상
        else:
            return None # 제외
    else:
        # no_finding == 0.0
        return 1  # 비정상

def process_labels(pa_label_file, label_csv_file, output_file):
    # 라벨 CSV 파일 로드
    labels_df = pd.read_csv(label_csv_file)
    labels_df['subject_id'] = labels_df['subject_id'].astype(str)
    labels_df['study_id'] = labels_df['study_id'].astype(str)
    
    # pa_label_file 읽기
    with open(pa_label_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    normal_count = 0
    abnormal_count = 0
    discarded_count = 0

    output_lines = []

    for image_path in image_paths:
        subject_id, study_id = extract_ids_from_path(image_path)
        if subject_id is None or study_id is None:
            discarded_count +=1
            logging.warning(f"경로에서 ID를 추출할 수 없습니다: {image_path}")
            continue
        # 라벨 데이터에서 해당 subject_id와 study_id를 가진 행을 찾음
        label_row = labels_df[(labels_df['subject_id'] == subject_id) & (labels_df['study_id'] == study_id)]
        if label_row.empty:
            discarded_count +=1
            logging.warning(f"해당하는 라벨이 없습니다: subject_id={subject_id}, study_id={study_id}")
            continue
        else:
            label = get_label_from_row(label_row.iloc[0])
            if label is None:
                discarded_count +=1
                logging.info(f"라벨링 규칙에 따라 제외된 이미지: {image_path}")
                continue
            elif label == 0:
                normal_count +=1
            elif label ==1:
                abnormal_count +=1
            else:
                # 예상치 못한 라벨 값
                discarded_count +=1
                logging.warning(f"예상치 못한 라벨 값: {label} 이미지: {image_path}")
                continue
            # output_lines에 추가
            output_lines.append(f"{image_path}, {label}")
    
    # 결과를 output_file에 저장
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    # 통계 정보 출력
    logging.info(f"총 처리된 이미지 수: {len(image_paths)}")
    logging.info(f"정상 이미지 수: {normal_count}")
    logging.info(f"비정상 이미지 수: {abnormal_count}")
    logging.info(f"제외된 이미지 수: {discarded_count}")
    logging.info(f"최종 결과를 {output_file}에 저장하였습니다.")


def filter_with_mimic_forced_generation(label_file, mimic_csv_file, output_file, chexbert_ckpt, device="cpu", log_interval=100):
    """Filter labeled data using findings and impression text.

    The function uses CheXbert to predict labels for the ``findings`` and
    ``impression`` sections in ``mimic_forced_generation``. Images are labeled
    as normal only when *both* sections are predicted as ``No Finding``.  Any
    other combination is treated as abnormal.  This differs from the previous
    implementation which required the two predictions to match exactly.

    Parameters
    ----------
    label_file : str
        Path to the label file produced in the previous steps.
    mimic_csv_file : str
        CSV file from mimic_forced_generation containing findings and impression columns.
    output_file : str
        Destination path for the filtered output.
    chexbert_ckpt : str
        Path to the CheXbert checkpoint.
    device : str, optional
        Device for running CheXbert. Defaults to ``"cpu"``.
    log_interval : int, optional
        Interval for logging progress information. Defaults to ``100``.
    """
    mimic_df = pd.read_csv(mimic_csv_file)
    mimic_df['subject_id'] = mimic_df['subject_id'].astype(str)
    mimic_df['study_id'] = mimic_df['study_id'].astype(str)

    mimic_df = mimic_df[(mimic_df['has_findings'] == 1) & (mimic_df['has_impression'] == 1)]

    chex_model, chex_tokenizer = load_chexbert_model_and_tokenizer(chexbert_ckpt, device=device)

    label_dict = {}
    total_rows = len(mimic_df)
    for idx, (_, row) in enumerate(mimic_df.iterrows(), 1):
        preds_find = get_chexbert_predictions(str(row['findings']), chex_model, chex_tokenizer, device=device)
        preds_imp = get_chexbert_predictions(str(row['impression']), chex_model, chex_tokenizer, device=device)
        if preds_find is None or preds_imp is None:
            continue

        label_find = chexbert_label_to_binary(preds_find)
        label_imp = chexbert_label_to_binary(preds_imp)
        if label_find is None or label_imp is None:
            continue

        # Normal label only when both sections are predicted as 'No Finding'
        if label_find == 0 and label_imp == 0:
            label = 0
        else:
            # If any section indicates abnormality, mark as abnormal
            label = 1

        label_dict[(row['subject_id'], row['study_id'])] = label
        if idx % log_interval == 0 or idx == total_rows:
            logging.info(
                "ChexBERT filtering %d/%d rows processed", idx, total_rows
            )

    with open(label_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    out_lines = []
    total_lines = len(lines)
    for idx, line in enumerate(lines, 1):
        try:
            img_path, _ = line.split(',')
            img_path = img_path.strip()
        except ValueError:
            continue
        subj_id, study_id = extract_ids_from_path(img_path)
        if subj_id is None or study_id is None:
            continue
        key = (subj_id, study_id)
        if key in label_dict:
            out_lines.append(f"{img_path}, {label_dict[key]}")
        if idx % log_interval == 0 or idx == total_lines:
            logging.info(
                "Applying ChexBERT filter to labels %d/%d", idx, total_lines
            )

    with open(output_file, 'w') as f:
        for line in out_lines:
            f.write(line + '\n')

    logging.info(f"Filtered {len(out_lines)} entries saved to {output_file}.")


"""
python data_preprocess_all.py \
    --base_path /data/mimic-cxr-jpg/files \
    --metadata_csv_file  /data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv \
    --split_csv_file /data/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv \
    --label_csv_file /data/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv \
    --output_path ./data
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전체 파이프라인을 실행하는 스크립트입니다.")
    parser.add_argument('--base_path', type=str, required=True, help='이미지가 저장된 기본 디렉토리 경로입니다.')
    parser.add_argument('--metadata_csv_file', type=str, required=True, help='PA 이미지를 위한 기본 디렉토리 경로입니다.')
    parser.add_argument('--split_csv_file', type=str, required=True, help='split 정보가 포함된 CSV 파일 경로입니다.')
    parser.add_argument('--label_csv_file', type=str, required=True, help='라벨링된 CSV 파일 경로입니다.')
    parser.add_argument('--output_path', type=str, required=True, help='결과 파일들이 저장될 디렉토리 경로입니다.')
    parser.add_argument('--mimic_csv', type=str, default=None, help='mimic_forced_generation CSV 파일 경로')
    parser.add_argument('--chexbert_ckpt', type=str, default=None, help='CheXbert checkpoint 경로')
    parser.add_argument('--device', type=str, default='cpu', help='CheXbert 실행 장치(cpu, cuda 등)')

    args = parser.parse_args()

    setup_logger(args.output_path)
    # Step 1: train.txt, validation.txt, test.txt 생성
    generate_split_files(args.base_path, args.split_csv_file, args.output_path)

    # Step 2: pa_train.txt, pa_validation.txt, pa_test.txt 생성
    generate_pa_split_files(args.base_path, args.metadata_csv_file, args.split_csv_file, args.output_path)

    # Step 3: pa_label_train.txt, pa_label_validation.txt, pa_label_test.txt 생성
    process_pa_labels(
        os.path.join(args.output_path, 'pa_train.txt'),
        os.path.join(args.output_path, 'train.txt'),
        os.path.join(args.output_path, 'pa_label_train.txt')
    )
    process_pa_labels(
        os.path.join(args.output_path, 'pa_validation.txt'),
        os.path.join(args.output_path, 'validation.txt'),
        os.path.join(args.output_path, 'pa_label_validation.txt')
    )
    process_pa_labels(
        os.path.join(args.output_path, 'pa_test.txt'),
        os.path.join(args.output_path, 'test.txt'),
        os.path.join(args.output_path, 'pa_label_test.txt')
    )

    # Step 4: 최종 라벨링된 파일 생성
    process_labels(
        os.path.join(args.output_path, 'pa_label_train.txt'),
        args.label_csv_file,
        os.path.join(args.output_path, 'labeled_train_sd.txt')
    )
    process_labels(
        os.path.join(args.output_path, 'pa_label_validation.txt'),
        args.label_csv_file,
        os.path.join(args.output_path, 'labeled_validation_sd.txt')
    )
    process_labels(
        os.path.join(args.output_path, 'pa_label_test.txt'),
        args.label_csv_file,
        os.path.join(args.output_path, 'labeled_test_sd.txt')
    )

    # Step 5: filter with mimic_forced_generation if provided
    if args.mimic_csv and args.chexbert_ckpt:
        filter_with_mimic_forced_generation(
            os.path.join(args.output_path, 'labeled_train_sd.txt'),
            args.mimic_csv,
            os.path.join(args.output_path, 'filtered_labeled_train_sd.txt'),
            args.chexbert_ckpt,
            device=args.device
        )
        filter_with_mimic_forced_generation(
            os.path.join(args.output_path, 'labeled_validation_sd.txt'),
            args.mimic_csv,
            os.path.join(args.output_path, 'filtered_labeled_validation_sd.txt'),
            args.chexbert_ckpt,
            device=args.device
        )
        filter_with_mimic_forced_generation(
            os.path.join(args.output_path, 'labeled_test_sd.txt'),
            args.mimic_csv,
            os.path.join(args.output_path, 'filtered_labeled_test_sd.txt'),
            args.chexbert_ckpt,
            device=args.device
        )
