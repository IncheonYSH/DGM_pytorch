import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloader
from model import EncoderG, EncoderF, Decoder, DecoderConcat
import torchvision.utils as vutils
import logging
from PIL import Image
import torchvision.transforms.functional as TF


# Logging 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(args, device):
    # 모델 초기화
    E_G = EncoderG().to(device)
    E_F = EncoderF().to(device)
    D_G = Decoder().to(device)
    D_F = Decoder().to(device)
    D_J = DecoderConcat().to(device)

    # 체크포인트 로드
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        E_G.load_state_dict(checkpoint['E_G'])
        E_F.load_state_dict(checkpoint['E_F'])
        D_G.load_state_dict(checkpoint['D_G'])
        D_F.load_state_dict(checkpoint['D_F'])
        D_J.load_state_dict(checkpoint['D_J'])
        logging.info(f"체크포인트 '{args.checkpoint}'에서 모델을 로드하였습니다.")
    else:
        logging.error(f"체크포인트 '{args.checkpoint}'가 존재하지 않습니다.")
        exit()

    return E_G, E_F, D_G, D_F, D_J

def infer_and_save_samples(args, E_G, E_F, D_G, D_F, D_J, device):
    # 데이터 로더 생성
    dataloader = get_dataloader(args.file_list_path, args.label_csv_path, batch_size=1, train=False)

    # num_samples만큼 데이터 로드
    num_samples = min(args.num_samples, len(dataloader))
    logging.info(f"총 {num_samples}개의 샘플을 처리합니다.")

    for i, (img, _) in enumerate(dataloader):  # 레이블은 무시하고 이미지만 처리
        if i >= num_samples:
            break
        
        img = img.to(device)

        # 모델 추론
        c_z = E_G(img)
        c_s = E_F(img)

        y_prime = D_G(c_z)                    # y': syn. normal image
        a = D_F(c_s)                          # a: residual map
        z_double_prime = y_prime + a          # z'': reconstructed image 2
        z_prime = D_J(torch.cat([c_z, c_s], dim=1))  # z': reconstructed image 1

        # 이미지 그리드로 변환
        def make_grid(images):
            return vutils.make_grid(images.cpu(), normalize=True, scale_each=True, nrow=1)

        # 이미지 저장
        grid_y_prime = make_grid(y_prime)
        grid_z_prime = make_grid(z_prime)
        grid_z_double_prime = make_grid(z_double_prime)
        grid_a = make_grid(a)

        # 저장 디렉토리 생성
        save_dir = os.path.join(args.output_dir, f"sample_{i}")
        os.makedirs(save_dir, exist_ok=True)

        # 각 이미지 저장
        vutils.save_image(grid_y_prime, os.path.join(save_dir, 'y_prime.png'))
        vutils.save_image(grid_z_prime, os.path.join(save_dir, 'z_prime.png'))
        vutils.save_image(grid_z_double_prime, os.path.join(save_dir, 'z_double_prime.png'))
        vutils.save_image(grid_a, os.path.join(save_dir, 'a.png'))

        logging.info(f"샘플 {i+1}/{num_samples} 처리 완료. 이미지 저장됨.")

if __name__ == "__main__":
    # argparse 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--file_list_path', nargs='+', required=True, help='이미지 경로가 담긴 txt 파일들의 경로')
    parser.add_argument('--label_csv_path', type=str, required=True, help='라벨 CSV 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='출력 디렉토리 경로')
    parser.add_argument('--num_samples', type=int, default=5, help='처리할 샘플 수')
    parser.add_argument('--gpu', type=int, default=0, help='사용할 GPU 번호')
    args = parser.parse_args()

    # 시드 설정
    set_seed(42)

    # 장치 설정
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    E_G, E_F, D_G, D_F, D_J = load_model(args, device)

    # 추론 및 이미지 저장
    infer_and_save_samples(args, E_G, E_F, D_G, D_F, D_J, device)
