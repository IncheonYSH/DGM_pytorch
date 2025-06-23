import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloader, get_validation_dataloader
from model import EncoderG, EncoderF, Decoder, DecoderConcat, Discriminator
from ladagan import Discriminator as LadaDiscriminator
from ladagan import Generator as LadaGenerator
from clip_encoder import VisionTransformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils
import random
import numpy as np
import logging
import sys
import math
import torch.nn.functional as F
from torch.autograd import grad
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
# os.environ["NUMEXPR_MAX_THREADS"] = "24"
# torch.set_num_threads(24)


# Logging 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stderr)

def log_hyperparameters(args):
    logging.info("===== Hyperparameters =====")
    logging.info(f"Always save checkpoint: {args.always_save_checkpoint}")
    logging.info(f"Continue Training: {args.continue_train}")
    logging.info(f"Checkpoint Directory: {args.checkpoint_dir}")
    logging.info(f"GPU: {args.gpu}")
    logging.info(f"Total Epochs: {args.epochs}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Use Gradient Checkpointing: {args.use_checkpoint}")
    logging.info(f"Backbone architecture: {args.architecture}")
    logging.info(f"Latent space dimension: {args.z_dimension}")
    logging.info(f"GAN type: {args.base_GAN}")
    logging.info(f"Target image size: {args.target_image_size}")
    logging.info(f"Generator Initial Learning Rate: {args.initial_lr_g}")
    logging.info(f"Discriminator Initial Learning Rate: {args.initial_lr_d}")
    logging.info(f"Optimizer momentum, beta 1: {args.beta_1}")
    logging.info(f"Optimizer momentum, beta 2: {args.beta_2}")
    logging.info(f"Weight Decay: {args.weight_decay}")
    logging.info(f"Learning Rate Scheduler: {args.lr_scheduler}")
    
    if args.lr_scheduler == 'step_decay':
        logging.info(f"Learning Rate Decay Epoch: {args.decay_epoch}")
        logging.info(f"Learning Rate Decay Factor: {args.lr_decay}")
    elif args.lr_scheduler == 'cosine_decay':
        logging.info("Cosine decay with warm-up enabled")
        logging.info(f"Warm-up Iterations: {args.warmup_iters}")
        logging.info(f"Learning Rate Decay Iterations: {args.lr_decay_iters}")
        logging.info(f"Min Learning Rate: {args.min_lr}")
        
    logging.info(f"Adversarial Loss Weight (lambda_A): {args.lambda_A}")
    logging.info(f"Reconstruction Loss Weight (lambda_R): {args.lambda_R}")
    logging.info(f"Total Variation Loss Weight (lambda_TV): {args.lambda_TV}")

    if args.base_GAN == 'WGANGP':
        logging.info(f"Gradient Penalty Weight (lambda_GP): {args.lambda_GP}")
    
    if args.alt_loss:
        logging.info(f"Alternative Loss enabled for {args.alt_loss_epochs} epochs")
    
    logging.info(f"Discriminator Update Steps per Iteration: {args.d_steps}")
    logging.info(f"Discriminator batch normalization: {args.is_batch_normalization}")
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

def wgan_gp(D, real_samples, fake_samples, center = 1):
    """Calculates the gradient penalty"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    # Get gradient w.r.t. interpolates
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean()
    return gradient_penalty

# zero-centered gradient penalty term
# Penalize discriminator gradients on true data distribution
def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

def decode_with_attention(decoder, inputs):
    """Run decoder and gracefully handle outputs with or without attention maps."""
    output = decoder(inputs)
    if isinstance(output, (list, tuple)) and len(output) == 2:
        return output[0], output[1]
    else:
        return output, None

def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(device)
    # 현재 PyTorch와 CUDA 버전 확인
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    log_hyperparameters(args)

    # 체크포인트 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = ""
    if args.checkpoint_filename:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_filename)

    # 로드 옵션
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    backbone_architecture = args.architecture

    # 모델 초기화
    if backbone_architecture == 'original':
        E_G = EncoderG().to(device)
        E_F = EncoderF().to(device)
        D_G = Decoder().to(device)
        D_F = Decoder().to(device)
        D_J = DecoderConcat().to(device)
        D_Disc = Discriminator(is_batch_normalization=args.is_batch_normalization).to(device)
    else:
        z_dim = args.z_dimension
        E_G = VisionTransformer(
            image_size=args.target_image_size,
            patch_size=16,
            width=512,
            layers=6,
            heads=8,
            mlp_ratio=4.0,
            output_dim=z_dim,
            pool_type='tok',
        ).to(device)
        E_F = VisionTransformer(
            image_size=args.target_image_size,
            patch_size=16,
            width=512,
            layers=6,
            heads=8,
            mlp_ratio=4.0,
            output_dim=z_dim,
            pool_type='tok',
        ).to(device)
        D_G = LadaGenerator(
            z_dim=z_dim,
            img_size=args.target_image_size,
            model_dim=[1024, 512, 256],
            dec_dim=[32, 16, 8],
        ).to(device)
        D_F = LadaGenerator(
            z_dim=z_dim,
            img_size=args.target_image_size,
            model_dim=[1024, 512, 256],
            dec_dim=[32, 16, 8],
            relu_activation=True,
        ).to(device)
        D_J = LadaGenerator(
            z_dim=z_dim*2,
            img_size=args.target_image_size,
            model_dim=[1024, 512, 256],
            dec_dim=[32, 16, 8],
        ).to(device)
        D_Disc = LadaDiscriminator(
            img_size=args.target_image_size,
            enc_dim=[64, 128, 256, 512, 1024],
            heads=4
            ).to(device)

    # 옵티마이저 설정
    params = list(E_G.parameters()) + list(E_F.parameters()) + \
             list(D_G.parameters()) + list(D_F.parameters()) + \
             list(D_J.parameters())
    optimizer_G = optim.Adam(params, lr=args.initial_lr_g, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)
    optimizer_D = optim.Adam(D_Disc.parameters(), lr=args.initial_lr_d, betas=(args.beta_1, args.beta_2))


    # Criterion setup based on selecte GAN
    # format: criterion_adversarial(arg1, arg2)
    criterion_L1 = nn.L1Loss()
    clip_value = torch.inf
    is_clamping = False
    is_wgan_penalty = False
    is_zero_centered_penalty = False
    if args.base_GAN == 'LSGAN':
        criterion_adversarial = lambda arg1, arg2: F.mse_loss(arg1, arg2)
    elif args.base_GAN == 'GAN':
        criterion_adversarial = lambda arg1, arg2: F.binary_cross_entropy(torch.sigmoid(arg1), arg2)
    elif args.base_GAN == 'WGAN':
        criterion_adversarial = lambda arg1, arg2: torch.where(arg2.view(-1)[0] == 1, -torch.mean(arg1), torch.mean(arg1))
        clip_value = args.clip_value
        is_clamping = True
    elif args.base_GAN == 'WGANGP':
        criterion_adversarial = lambda arg1, arg2: torch.where(arg2.view(-1)[0] == 1, -torch.mean(arg1), torch.mean(arg1))
        is_wgan_penalty = True
    elif args.base_GAN == 'NSGAN': # not saturated + gradient penalty
        criterion_adversarial = lambda arg1, arg2: F.binary_cross_entropy_with_logits(arg1, arg2) # default reduction: mean
        is_zero_centered_penalty = True
    else:
        criterion_adversarial = None

    # 데이터 로더
    train_file_list = args.train_file_list
    val_file_list = args.val_file_list

    dataloader = get_dataloader(train_file_list, batch_size=args.batch_size, train=True, target_image_size=args.target_image_size)
    val_loader = get_validation_dataloader(val_file_list, batch_size=args.batch_size, target_image_size=args.target_image_size)

    # 텐서보드 설정
    writer = SummaryWriter()

    # 하이퍼파라미터 설정
    lambda_A = args.lambda_A
    lambda_R = args.lambda_R
    lambda_TV = args.lambda_TV
    lambda_GP = args.lambda_GP
    warmup_iters = args.warmup_iters
    lr_decay_iters = args.lr_decay_iters

                
    if args.continue_train:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            E_G.load_state_dict(checkpoint['E_G'])
            E_F.load_state_dict(checkpoint['E_F'])
            D_G.load_state_dict(checkpoint['D_G'])
            D_F.load_state_dict(checkpoint['D_F'])
            D_J.load_state_dict(checkpoint['D_J'])
            # Discriminator setup
            D_Disc.load_state_dict(checkpoint['D_Disc'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            completed_epoch = checkpoint['epoch']
            start_epoch = completed_epoch + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            global_step = checkpoint.get('global_step', completed_epoch * len(dataloader))
            backbone_architecture = checkpoint.get('backbone_architecture')
            print(f"체크포인트 '{checkpoint_path}'에서 로드하였습니다. 재개 에포크: {start_epoch}")
        else:
            print(f"체크포인트 '{checkpoint_path}'가 존재하지 않습니다. 새로 학습을 시작합니다.")
            current_time = datetime.now().strftime('%y%m%d%H%M%S')
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint{current_time}.pth')
            print(f"Checkpoint path: {checkpoint_path}")
    else:
        print("새로 학습을 시작합니다.")
        current_time = datetime.now().strftime('%y%m%d%H%M%S')
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint{current_time}.pth')
        print(f"Checkpoint path: {checkpoint_path}")

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
                optimizer_D.zero_grad()

                # 인코딩
                c_z = E_G(img)
                c_s = E_F(img)

                with torch.no_grad():
                    y_prime, y_prime_attn = decode_with_attention(D_G, c_z)

                # 판별자 손실 계산
                img.requires_grad_(True)
                D_real = D_Disc(img)
                D_fake = D_Disc(y_prime.detach())

                # Loss calculation for discriminator
                loss_D_real = criterion_adversarial(D_real, torch.ones_like(D_real))
                loss_D_fake = criterion_adversarial(D_fake, torch.zeros_like(D_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)
                writer.add_scalar('Loss_D/D-real', loss_D_real.item(), global_step)
                writer.add_scalar('Loss_D/D-fake', loss_D_fake.item(), global_step)

                # Gradient penalty for WGAN-GP
                if is_wgan_penalty:
                    gradient_penalty = wgan_gp(D_Disc, img, y_prime.detach())
                    writer.add_scalar('Loss_D/Gradient penalty', gradient_penalty.item(), global_step)
                    loss_D += lambda_GP * gradient_penalty
                # Zero centered gradient penalty
                if is_zero_centered_penalty:
                    gradient_penalty = compute_grad2(D_real, img).mean()
                    writer.add_scalar('Loss_D/Gradient penalty', gradient_penalty.item(), global_step)
                    loss_D += lambda_GP * gradient_penalty
                
                loss_D.backward()
                optimizer_D.step()

                # Gradient clipping for WGAN and WGAN variant
                if is_clamping:
                    for p in D_Disc.parameters():
                        p.data.clamp_(-clip_value, clip_value)

            ############################
            # (2) Update G network
            ###########################
            optimizer_G.zero_grad()

            # 인코딩
            c_z = E_G(img)
            c_s = E_F(img)

            # 디코딩
            y_prime, y_prime_attn = decode_with_attention(D_G, c_z)
            a, a_attn = decode_with_attention(D_F, c_s)
            z_double_prime = y_prime + a  # z'' 재구성된 이미지 2
            z_prime, z_prime_attn = decode_with_attention(D_J, torch.cat([c_z, c_s], dim=1))  # z' 재구성된 이미지 1

            # 생성자 손실 계산
            # reduction='mean' is applied in default
            D_fake = D_Disc(y_prime)

            # Alternative loss function for stability in early steps of training
            # It's from vanila GAN paper. But, it is simply expanded with changes of adverserial loss function. Need experiment for this changes.
            if args.alt_loss and epoch < args.alt_loss_epochs:
                loss_G_adv = criterion_adversarial(D_fake, torch.zeros_like(D_fake))
            else:
                loss_G_adv = criterion_adversarial(D_fake, torch.ones_like(D_fake))

            loss_R1 = criterion_L1(z_prime, img)
            loss_R2 = criterion_L1(z_double_prime, img)

            # label이 0인 경우(정상 데이터)에 대해서만 loss_R3 계산
            mask = (label == 0).float().view(-1, 1, 1, 1)
            loss_R3 = criterion_L1(a * mask, torch.zeros_like(a))

            tv_loss = total_variation_loss(y_prime)

            # 각 손실에 가중치 적용
            loss_G = lambda_A * loss_G_adv + lambda_R * (loss_R1 + loss_R2 + loss_R3) + lambda_TV * tv_loss

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

                y_prime, y_prime_attn = decode_with_attention(D_G, c_z)
                a, a_attn = decode_with_attention(D_F, c_s)
                z_double_prime = y_prime + a
                z_prime, z_prime_attn = decode_with_attention(D_J, torch.cat([c_z, c_s], dim=1))

                loss_R1 = criterion_L1(z_prime, img)
                loss_R2 = criterion_L1(z_double_prime, img)

                mask = (label == 0).float().view(-1, 1, 1, 1)
                loss_R3 = criterion_L1(a * mask, torch.zeros_like(a))

                val_loss += (loss_R1 + loss_R2 + loss_R3).item()

            val_loss /= len(val_loader)
            writer.add_scalar('Loss/Validation', val_loss, epoch)

        # 최고 검증 손실 갱신 및 모델 저장
        if val_loss < best_val_loss or args.always_save_checkpoint:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'E_G': E_G.state_dict(),
                'E_F': E_F.state_dict(),
                'D_G': D_G.state_dict(),
                'D_F': D_F.state_dict(),
                'D_J': D_J.state_dict(),
                'D_Disc': D_Disc.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'best_val_loss': best_val_loss,
                'is_batch_normalization': args.is_batch_normalization,
                'backbone_architecture': args.architecture
            }, checkpoint_path)
            print(f"Epoch [{epoch+1}/{args.epochs}] - Validation loss: {val_loss:.4f}. model saved.")
        else:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Validation loss: {val_loss:.4f}.")

        # 생성된 이미지 텐서보드에 기록
        with torch.no_grad():
            num_samples = min(16, img.size(0))
            sample_img = img[:num_samples]  # 첫 num_samples개 이미지
            label_sample = label[:num_samples]
            c_z_sample = E_G(sample_img)
            c_s_sample = E_F(sample_img)

            # 디코더를 통해 이미지 생성
            y_prime_sample, y_prime_attn_sample = decode_with_attention(D_G, c_z_sample)          # y': syn. normal image
            a_sample, a_attn_sample = decode_with_attention(D_F, c_s_sample)                # a: residual map
            z_double_prime_sample = y_prime_sample + a_sample  # z'': reconstructed image 2
            z_prime_sample, z_prime_attn_sample = decode_with_attention(D_J, torch.cat([c_z_sample, c_s_sample], dim=1))  # z': reconstructed image 1

            def to_01(image):
                return (image + 1) / 2
            
            # 4x4 그리드로 이미지 생성 함수 정의
            def make_grid(images, normalize=False):
                return vutils.make_grid(images.cpu(), normalize=normalize, scale_each=True, nrow=4)

            # 각 이미지를 4x4 그리드로 생성
            grid_original = make_grid(to_01(sample_img))
            grid_y_prime = make_grid(to_01(y_prime_sample))
            grid_a = make_grid(to_01(a_sample))
            grid_z_prime = make_grid(to_01(z_prime_sample))
            grid_z_double_prime = make_grid(to_01(z_double_prime_sample))

            # 텐서보드에 이미지 기록
            writer.add_image('Original Images', grid_original, epoch)
            writer.add_image('Syn Normal (y\')', grid_y_prime, epoch)
            writer.add_image('Residual (a)', grid_a, epoch)
            writer.add_image('Recon1 (z\')', grid_z_prime, epoch)
            writer.add_image('Recon2 (z\'\')', grid_z_double_prime, epoch)

            # 어텐션 맵 처리 및 텐서보드에 기록
            # 어텐션 맵은 여러 헤드를 가지므로 평균을 내서 시각화
            def process_attention_map(attn_map):
                # attn_map: (batch_size, num_heads, N)
                N = attn_map.size(-1)
                sqrt_N = int(N ** 0.5)
                if sqrt_N * sqrt_N != N:
                    raise ValueError("Attention map size is not a perfect square")
                attn_map = attn_map.view(attn_map.size(0), attn_map.size(1), sqrt_N, sqrt_N)
                mean_attn = torch.mean(attn_map, dim=1, keepdim=True)  # (batch_size, 1, H, W)
                mean_attn = F.interpolate(mean_attn, size=(args.target_image_size, args.target_image_size), mode='nearest')
                return mean_attn
            
            def process_and_log_attention_maps(attn_maps, title):
                for idx, attn_map in enumerate(attn_maps):
                    mean_attn = process_attention_map(attn_map)
                    grid_attn = make_grid(mean_attn, normalize=True)
                    writer.add_image(f'Attention Maps/{title} Layer {idx+1}', grid_attn, epoch)

            if y_prime_attn_sample is not None:
                process_and_log_attention_maps(y_prime_attn_sample, 'Syn Normal (y\')')
            if a_attn_sample is not None:
                process_and_log_attention_maps(a_attn_sample, 'Residual (a)')
            if z_prime_attn_sample is not None:
                process_and_log_attention_maps(z_prime_attn_sample, 'Recon1 (z\')')

            # 텍스트 형식의 라벨 정보 기록
            for i in range(num_samples):
                writer.add_text(f'Label Info/Sample {i + 1}', f"Label: {label_sample[i].item()}", epoch)


    writer.close()

def total_variation_loss(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# Add --continue_train for continuous training. LR and hyperparameters will would be adjusted automatically.
# python train.py --batch_size 64 --d_steps 1 --gpu 0 --train_file_list ./data/labeled_train.txt --val_file_list ./data/labeled_validation.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Train/val set(necessary) ###
    parser.add_argument('--train_file_list', nargs='+', required=True, help='학습용 이미지 경로 리스트 파일')
    parser.add_argument('--val_file_list', nargs='+', required=True, help='검증용 이미지 경로 리스트 파일')
    ### Image size ###
    parser.add_argument('--target_image_size', type=int, default=224, help="입력 이미지 사이즈. 자동으로 이 크기에 맞추어 resizing 됨")
    ### Architecture ##
    # unet + attention without encoder-decoder skip connection.  
    parser.add_argument('--architecture', type=str, default='original', choices=['original', 'new'], help='Select backbone architecture to train.')
    parser.add_argument('--z_dimension', type=int, default=128, help="latent space dimension")
    ### Gradient Checkpointing 옵션 ###
    parser.add_argument('--use_checkpoint', action='store_true', help='Gradient Checkpointing 사용 여부')
    ### Save options ###
    parser.add_argument('--always_save_checkpoint', action='store_true', help='Always save a checkpoint after each eval0')
    parser.add_argument('--checkpoint_dir', type=str, default='./chkpts', help='체크포인트 저장 디렉토리')
    parser.add_argument('--checkpoint_filename', type=str, help='체크포인트 파일명')
    ### Continue train ###
    parser.add_argument('--continue_train', action='store_true', help='계속 학습 여부')
    ### GPU ###
    parser.add_argument('--gpu', type=int, default=0, help='사용할 GPU 번호')
    ### Training setup ###
    parser.add_argument('--epochs', type=int, default=30, help='총 학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=1, help='배치 크기')
    ### Select main training algorithm(default is the LSGAN as mentioned in original paper) ###
    ### Batch normalization is not recommaned in WGANs(from paper) ###
    parser.add_argument('--base_GAN',type=str, default='LSGAN', choices=['GAN', 'LSGAN', 'WGAN', 'WGANGP', 'NSGAN'], help='Select basement GAN')
    parser.add_argument('--is_batch_normalization', type=lambda x: x.lower() == 'true', default=True, help='Discriminator 의 batch normalization 사용 여부')
    ### WGAN clip boundary ###
    parser.add_argument('--clip_value', type=float, default=0.01, help='Clip boundary for WGAN training') # default value from paper
    ### Common optimizer setup ###
    parser.add_argument('--initial_lr_g', type=float, default=5e-5, help='generator 초기 학습률')
    parser.add_argument('--initial_lr_d', type=float, default=5e-5, help='discriminator 초기 학습률')
    parser.add_argument('--lr_scheduler', type=str, choices=['step_decay', 'cosine_decay'], default='step_decay', help='학습률 스케줄링 방식 선택')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='가중치 감소율')
    # (0, 0.9) for WGANs. (0.5, 0.999) for others
    parser.add_argument('--beta_1', type=float, default=0.5, help="Adam optimizer momentum parameter")
    parser.add_argument('--beta_2', type=float, default=0.999, help="Adam optimizer momentum parameter")
    ### For cosine decay ###
    parser.add_argument('--min_lr', type=float, default=1e-6, help='최소 학습률')
    parser.add_argument('--warmup_iters', type=int, default=1000, help='warmup 단계의 반복 수')
    parser.add_argument('--lr_decay_iters', type=int, default=40000, help='학습률이 최소가 되는 반복 수')
    ### For step decay ###
    parser.add_argument('--decay_epoch', type=int, default=10, help='step decay에서 학습률 감소 주기')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='step decay에서 학습률 감소 비율')
    ### Loss function hyperparameters ###
    parser.add_argument('--lambda_A', type=float, default=10.0, help='적대적 손실 가중치')
    parser.add_argument('--lambda_R', type=float, default=10.0, help='복원 손실 가중치')
    parser.add_argument('--lambda_TV', type=float, default=0.5, help='총 변동 손실 가중치')
    parser.add_argument('--lambda_GP', type=float, default=10, help='Gradienta penalty 가중치')
    ### Modified training algorithm options ###
    parser.add_argument('--alt_loss', action='store_true', help='수정된 adverserial loss function 사용 여부')
    parser.add_argument('--alt_loss_epochs', type=int, default=1, help='수정된 adverserial loss function 사용할 에포크 수')
    # In original vanilla gan paper, 1~5 is recommanded. For WGAN, 5 is recommanded in original paper.
    parser.add_argument('--d_steps', type=int, default=1, help='D를 최적화할 스텝 수')


    args = parser.parse_args()
    set_seed(42)
    train(args)
