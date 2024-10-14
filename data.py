import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import argparse
import logging

import torchvision.transforms.functional as TF

# 로깅 설정
logging.basicConfig(level=logging.INFO)

class ImageDataset(Dataset):
    def __init__(self, data_file_paths, target_image_size=224):
        self.target_image_size = target_image_size

        # 이미지 경로와 라벨을 담을 리스트 초기화
        self.image_paths = []
        self.labels = []

        normal_count = 0
        abnormal_count = 0

        # 각 데이터 파일에서 이미지 경로와 라벨을 읽어옴
        for data_file in data_file_paths:
            with open(data_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) != 2:
                        logging.warning(f"잘못된 형식의 줄이 있습니다: {data_file}: {line}")
                        continue
                    image_path = parts[0].strip()
                    label_str = parts[1].strip()
                    try:
                        label = int(label_str)
                    except ValueError:
                        logging.warning(f"잘못된 라벨 값: {data_file}: {label_str}")
                        continue
                    if label == 0:
                        normal_count += 1
                    elif label == 1:
                        abnormal_count += 1
                    else:
                        logging.warning(f"예상치 못한 라벨 값: {label} (줄: {line})")
                        continue
                    self.image_paths.append(image_path)
                    self.labels.append(label)

        logging.info(f"총 이미지 수: {len(self.image_paths)}")
        logging.info(f"정상 이미지 수: {normal_count}")
        logging.info(f"비정상 이미지 수: {abnormal_count}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(path).convert('L')  # 흑백 이미지로 로드

        # 이미지 전처리
        s = min(img.size)

        if s < self.target_image_size:
            logging.warning(f"이미지 크기가 너무 작습니다: {img.size}, 경로: {path}")
            s = self.target_image_size

        r = self.target_image_size / s
        new_size = (round(r * img.size[0]), round(r * img.size[1]))
        img = TF.resize(img, new_size, interpolation=Image.LANCZOS)
        img = TF.center_crop(img, output_size=(self.target_image_size, self.target_image_size))
        img = TF.to_tensor(img)
        img = 2. * img - 1.  # 픽셀 값을 [-1, 1] 범위로 스케일링

        label = self.labels[index]
        return img, label

def get_dataloader(data_file_paths, batch_size=1, train=True):
    dataset = ImageDataset(data_file_paths, target_image_size=224)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=8)
    return dataloader

def get_validation_dataloader(data_file_paths, batch_size=1):
    dataset = ImageDataset(data_file_paths, target_image_size=224)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return val_loader

# 예시 실행 방법:
# python data.py --data_file_paths ./data/labeled_train.txt --batch_size 64
if __name__ == "__main__":
    # argparse를 통해 인자 받기
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file_paths', nargs='+', required=True, help='이미지 경로와 라벨이 담긴 txt 파일들의 경로')
    parser.add_argument('--batch_size', type=int, default=1, help='배치 크기')
    args = parser.parse_args()

    # 데이터로더 생성
    dataloader = get_dataloader(args.data_file_paths, batch_size=args.batch_size, train=True)

    # 예시로 데이터 로드 확인
    for imgs, labels in dataloader:
        print("이미지 크기:", imgs.shape)
        print("라벨:", labels)
        break
