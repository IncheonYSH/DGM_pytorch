import os
import pandas as pd
import logging

def setup_logger(output_path):
    # 로그 파일을 저장할 디렉토리가 없으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 로깅 설정
    log_file = os.path.join(output_path, 'split_generation.log')
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # 'w' 모드로 설정하여 파일을 덮어씀
        level=logging.INFO,  # 로그 레벨을 INFO로 설정
        format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 포맷 지정
    )
    
    # 콘솔에도 로그를 출력하도록 설정
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

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
        
        # 주어진 포맷에 맞게 경로 구성 (사용자가 지정한 base_path 아래의 구조를 따름)
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

# python test_train_split.py --base_path /data3/physionet.org/files/mimic-cxr-jpg/2.1.0/files --csv_file /data3/physionet.org/files/mimic-cxr-jpg/2.1.0/files/mimic-cxr-2.0.0-split.csv --output_path ./data
if __name__ == "__main__":
    # 사용자로부터 기본 디렉토리, csv 파일 경로 및 출력 경로를 args로 받음
    import argparse

    parser = argparse.ArgumentParser(description="Generate train, validation, test split text files from CSV.")
    parser.add_argument('--base_path', type=str, required=True, help='The base directory where images are stored.')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file with split information.')
    parser.add_argument('--output_path', type=str, required=True, help='The directory where output txt files will be stored.')
    
    args = parser.parse_args()

    # 함수 호출
    generate_split_files(args.base_path, args.csv_file, args.output_path)
