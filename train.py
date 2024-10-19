import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloader, get_validation_dataloader
from model import EncoderG, EncoderF, Decoder, DecoderConcat, Discriminator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils
import random
import numpy as np
import logging
import sys
import math

# Logging 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stderr)

def log_hyperparameters(args):
    logging.info("===== Hyperparameters =====")
    logging.info(f"Continue Training: {args.continue_train}")
    logging.info(f"Checkpoint Directory: {args.checkpoint_dir}")
    logging.info(f"GPU: {args.gpu}")
    logging.info(f"Total Epochs: {args.epochs}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Generator Initial Learning Rate: {args.initial_lr_g}")
    logging.info(f"Discriminator Initial Learning Rate: {args.initial_lr_d}")
    logging.info(f"Weight Decay: {args.weight_decay}")
    logging.info(f"Min Learning Rate: {args.min_lr}")
    logging.info(f"Warm-up Iterations: {args.warmup_iters}")
    logging.info(f"Learning Rate Decay Iterations: {args.lr_decay_iters}")
    logging.info(f"Learning Rate Scheduler: {args.lr_scheduler}")
    
    if args.lr_scheduler == 'step_decay':
        logging.info(f"Learning Rate Decay Epoch: {args.decay_epoch}")
        logging.info(f"Learning Rate Decay Factor: {args.lr_decay}")
    elif args.lr_scheduler == 'cosine_decay':
        logging.info("Cosine decay with warm-up enabled")
        
    logging.info(f"Adversarial Loss Weight (lambda_A): {args.lambda_A}")
    logging.info(f"Reconstruction Loss Weight (lambda_R): {args.lambda_R}")
    logging.info(f"Total Variation Loss Weight (lambda_TV): {args.lambda_TV}")
    
    if args.alt_loss:
        logging.info(f"Alternative Loss enabled for {args.alt_loss_epochs} epochs")
    
    logging.info(f"Discriminator Update Steps per Iteration: {args.d_steps}")
    logging.info(f"Training File List: {args.train_file_list}")
    logging.info(f"Validation File List: {args.val_file_list}")
    logging.info("===========================")

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    log_hyperparameters(args)

    # 모델 초기화
    E_G = EncoderG().to(device)
    E_F = EncoderF().to(device)
    D_G = Decoder().to(device)
    D_F = Decoder().to(device)
    D_J = DecoderConcat().to(device)
    D_Disc = Discriminator().to(device)

    # 옵티마이저 설정
    params = list(E_G.parameters()) + list(E_F.parameters()) + \
             list(D_G.parameters()) + list(D_F.parameters()) + \
             list(D_J.parameters())
    optimizer_G = optim.Adam(params, lr=args.initial_lr_g, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    optimizer_D = optim.Adam(D_Disc.parameters(), lr=args.initial_lr_d, betas=(0.5, 0.999))

    # 체크포인트 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pth')

    # 로드 옵션
    start_epoch = 0
    best_val_loss = float('inf')
    if args.continue_train:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            E_G.load_state_dict(checkpoint['E_G'])
            E_F.load_state_dict(checkpoint['E_F'])
            D_G.load_state_dict(checkpoint['D_G'])
            D_F.load_state_dict(checkpoint['D_F'])
            D_J.load_state_dict(checkpoint['D_J'])
            D_Disc.load_state_dict(checkpoint['D_Disc'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"체크포인트 '{checkpoint_path}'에서 로드하였습니다. 재개 에포크: {start_epoch}")
        else:
            print(f"체크포인트 '{checkpoint_path}'가 존재하지 않습니다. 새로 학습을 시작합니다.")
    else:
        print("새로 학습을 시작합니다.")

    # 손실 함수
    criterion_L1 = nn.L1Loss()
    criterion_MSE = nn.MSELoss()
    criterion_BCE = nn.BCELoss()

    # 데이터 로더
    train_file_list = args.train_file_list
    val_file_list = args.val_file_list

    dataloader = get_dataloader(train_file_list, batch_size=args.batch_size, train=True)
    val_loader = get_validation_dataloader(val_file_list, batch_size=args.batch_size)

    # 텐서보드 설정
    writer = SummaryWriter()

    # 하이퍼파라미터 설정
    lambda_A = args.lambda_A
    lambda_R = args.lambda_R
    lambda_TV = args.lambda_TV
    warmup_iters = args.warmup_iters
    lr_decay_iters = args.lr_decay_iters
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        E_G.train()
        E_F.train()
        D_G.train()
        D_F.train()
        D_J.train()
        D_Disc.train()

        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        for idx, (img, label) in enumerate(loop):
            img = img.to(device)
            label = label.to(device)
            if args.lr_scheduler == 'cosine_decay':
                # Cosine decay with warmup 적용
                new_lr_g = get_lr(global_step, warmup_iters, lr_decay_iters, args.initial_lr_g, args.min_lr)
                for param_group in optimizer_G.param_groups:
                    param_group['lr'] = new_lr_g                
                new_lr_d = get_lr(global_step, warmup_iters, lr_decay_iters, args.initial_lr_d, args.min_lr)
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] = new_lr_d

            elif args.lr_scheduler == 'step_decay' and (epoch + 1) % args.decay_epoch == 0:
                # 기존의 step decay 적용
                for param_group in optimizer_G.param_groups:
                    param_group['lr'] *= args.lr_decay
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] *= args.lr_decay

            ############################
            # (1) Update D network: k steps
            ###########################
            for _ in range(args.d_steps):
                # 인코딩
                c_z = E_G(img)
                c_s = E_F(img)

                # 디코딩
                with torch.no_grad():
                    y_prime = D_G(c_z)

                # 판별자 손실 계산
                D_real = D_Disc(img)
                D_fake = D_Disc(y_prime.detach())

                # BCE Loss - original GAN
                # loss_D_real = criterion_BCE(torch.sigmoid(D_real), torch.ones_like(D_real))
                # loss_D_fake = criterion_BCE(torch.sigmoid(D_fake), torch.zeros_like(D_fake))
                # loss_D = 0.5 * (loss_D_real + loss_D_fake)

                # LSGAN
                loss_D_real = criterion_MSE(D_real, torch.ones_like(D_real))
                loss_D_fake = criterion_MSE(D_fake, torch.zeros_like(D_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            ############################
            # (2) Update G network
            ###########################
            # 인코딩
            c_z = E_G(img)
            c_s = E_F(img)

            # 디코딩
            y_prime = D_G(c_z)
            a = D_F(c_s)
            z_double_prime = y_prime + a  # z'' 재구성된 이미지 2
            z_prime = D_J(torch.cat([c_z, c_s], dim=1))  # z' 재구성된 이미지 1

            # 생성자 손실 계산
            # reduction='mean' is applied in default
            D_fake = D_Disc(y_prime)

            # 학습 초기에는 log D(G(z)) 사용 - This option is not recommanded in this model
            if args.alt_loss and epoch < args.alt_loss_epochs:
                # loss_G_adv = criterion_BCE(torch.sigmoid(D_fake), torch.ones_like(D_fake))
                # MSE loss instead of BCE loss(LSGAN)
                loss_G_adv = - 0.5 * criterion_MSE(D_fake, torch.zeros_like(D_fake))
            else:
                loss_G_adv = 0.5 * criterion_MSE(D_fake, torch.ones_like(D_fake))

            loss_R1 = criterion_L1(z_prime, img)
            loss_R2 = criterion_L1(z_double_prime, img)

            # label이 0인 경우(정상 데이터)에 대해서만 loss_R3 계산
            mask = (label == 0).float().view(-1, 1, 1, 1)
            loss_R3 = criterion_L1(a * mask, torch.zeros_like(a))

            tv_loss = total_variation_loss(y_prime)

            # 각 손실에 가중치 적용
            loss_G = lambda_A * loss_G_adv + lambda_R * (loss_R1 + loss_R2 + loss_R3) + lambda_TV * tv_loss

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # 로깅
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

            # console log
            # logging.info(f"Epoch [{epoch+1}/{args.epochs}] Batch [{idx+1}/{len(dataloader)}]: "
            #              f"loss_G = {loss_G.item():.6f}, loss_D = {loss_D.item():.6f}")

            # 텐서보드 로깅
            global_step = epoch * len(dataloader) + idx
            writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
            writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)

            # 학습률 로깅 (Generator와 Discriminator 각각)
            writer.add_scalar('LR/Generator', optimizer_G.param_groups[0]['lr'], global_step)
            writer.add_scalar('LR/Discriminator', optimizer_D.param_groups[0]['lr'], global_step)

            # Generator 손실의 각 항목 로깅
            writer.add_scalar('Loss_G/Adversarial', loss_G_adv.item(), global_step)
            writer.add_scalar('Loss_G/Reconstruction', (loss_R1 + loss_R2 + loss_R3).item(), global_step)
            writer.add_scalar('Loss_G/TotalVariation', tv_loss.item(), global_step)

        # 검증 손실 계산
        val_loss = 0.0
        E_G.eval()
        E_F.eval()
        D_G.eval()
        D_F.eval()
        D_J.eval()
        with torch.no_grad():
            for img, label in val_loader:
                img = img.to(device)
                label = label.to(device)

                c_z = E_G(img)
                c_s = E_F(img)

                y_prime = D_G(c_z)
                a = D_F(c_s)
                z_double_prime = y_prime + a
                z_prime = D_J(torch.cat([c_z, c_s], dim=1))

                loss_R1 = criterion_L1(z_prime, img)
                loss_R2 = criterion_L1(z_double_prime, img)

                mask = (label == 0).float().view(-1, 1, 1, 1)
                loss_R3 = criterion_L1(a * mask, torch.zeros_like(a))

                val_loss += (loss_R1 + loss_R2 + loss_R3).item()

            val_loss /= len(val_loader)
            writer.add_scalar('Loss/Validation', val_loss, epoch)

        # 최고 검증 손실 갱신 및 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'E_G': E_G.state_dict(),
                'E_F': E_F.state_dict(),
                'D_G': D_G.state_dict(),
                'D_F': D_F.state_dict(),
                'D_J': D_J.state_dict(),
                'D_Disc': D_Disc.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"Epoch [{epoch+1}/{args.epochs}] - Validation loss updated: {val_loss:.4f}. model saved.")
        else:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Validation loss: {val_loss:.4f}.")

        # 생성된 이미지 텐서보드에 기록
        with torch.no_grad():
            num_samples = min(16, img.size(0))
            sample_img = img[:num_samples]  # 첫 num_samples개 이미지
            label_sample = label[:num_samples]
            c_z_sample = E_G(sample_img)
            c_s_sample = E_F(sample_img)

            y_prime_sample = D_G(c_z_sample)          # y': syn. normal image
            a_sample = D_F(c_s_sample)                # a: residual map
            z_double_prime_sample = y_prime_sample + a_sample  # z'': reconstructed image 2
            z_prime_sample = D_J(torch.cat([c_z_sample, c_s_sample], dim=1))  # z': reconstructed image 1

            # 각 이미지를 그리드로 생성
            def make_grid(images):
                return vutils.make_grid(images.cpu(), normalize=True, scale_each=True, nrow=4)

            grid_original = make_grid(sample_img)
            grid_y_prime = make_grid(y_prime_sample)
            grid_a = make_grid(a_sample)
            grid_z_prime = make_grid(z_prime_sample)
            grid_z_double_prime = make_grid(z_double_prime_sample)

            # 텐서보드에 이미지 기록
            writer.add_image('Original Images', grid_original, epoch)
            writer.add_image('Syn Normal (y\')', grid_y_prime, epoch)
            writer.add_image('Residual (a)', grid_a, epoch)
            writer.add_image('Recon1 (z\')', grid_z_prime, epoch)
            writer.add_image('Recon2 (z\'\')', grid_z_double_prime, epoch)

        # 학습률 감소
        if (epoch + 1) % args.decay_epoch == 0:
            for param_group in optimizer_G.param_groups:
                param_group['lr'] *= args.lr_decay
            for param_group in optimizer_D.param_groups:
                param_group['lr'] *= args.lr_decay

    writer.close()

def total_variation_loss(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# Add --continue_train for continuous training. LR and hyperparameters will would be adjusted automatically.
# python train.py --batch_size 64 --d_steps 1 --gpu 0 --train_file_list ./data/labeled_train.txt --val_file_list ./data/labeled_validation.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_train', action='store_true', help='계속 학습 여부')
    parser.add_argument('--checkpoint_dir', type=str, default='./chkpts', help='체크포인트 저장 디렉토리')
    parser.add_argument('--gpu', type=int, default=0, help='사용할 GPU 번호')
    parser.add_argument('--epochs', type=int, default=30, help='총 학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=1, help='배치 크기')
    parser.add_argument('--initial_lr_g', type=float, default=1e-5, help='generator 초기 학습률')
    parser.add_argument('--initial_lr_d', type=float, default=1e-5, help='discriminator 초기 학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='가중치 감소율')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='최소 학습률')
    parser.add_argument('--warmup_iters', type=int, default=1000, help='warmup 단계의 반복 수')
    parser.add_argument('--lr_decay_iters', type=int, default=45000, help='학습률이 최소가 되는 반복 수')
    parser.add_argument('--lr_scheduler', type=str, choices=['step_decay', 'cosine_decay'], default='step_decay', help='학습률 스케줄링 방식 선택')
    parser.add_argument('--decay_epoch', type=int, default=10, help='step decay에서 학습률 감소 주기')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='step decay에서 학습률 감소 비율')
    parser.add_argument('--lambda_A', type=float, default=10.0, help='적대적 손실 가중치')
    parser.add_argument('--lambda_R', type=float, default=10.0, help='복원 손실 가중치')
    parser.add_argument('--lambda_TV', type=float, default=1.0, help='총 변동 손실 가중치')
    parser.add_argument('--alt_loss', action='store_true', help='대안적인 학습 알고리즘 사용 여부')
    parser.add_argument('--alt_loss_epochs', type=int, default=5, help='대안 손실 함수를 사용할 에포크 수')
    parser.add_argument('--d_steps', type=int, default=1, help='D를 최적화할 스텝 수')

    # 추가된 인자들
    parser.add_argument('--train_file_list', nargs='+', required=True, help='학습용 이미지 경로 리스트 파일')
    parser.add_argument('--val_file_list', nargs='+', required=True, help='검증용 이미지 경로 리스트 파일')

    args = parser.parse_args()
    set_seed(42)
    train(args)
