import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


# 인코더 E_G
class EncoderG(nn.Module):
    def __init__(self):
        super(EncoderG, self).__init__()
        self.layers = nn.Sequential(
            # Conv, IN, ReLU
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            # 이하 동일하게 구성
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            # Residual Blocks ×4
            *[nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.InstanceNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.InstanceNorm2d(512)
            ) for _ in range(4)]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            if isinstance(layer, nn.Sequential):
                residual = out
                out = layer(out)
                out += residual
            else:
                out = layer(out)
        return out

# 인코더 E_F
class EncoderF(nn.Module):
    def __init__(self):
        super(EncoderF, self).__init__()
        self.layers = nn.Sequential(
            # Conv, ReLU
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # 이하 동일하게 구성
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Residual Blocks ×4
            *[nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1)
            ) for _ in range(4)]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            if isinstance(layer, nn.Sequential):
                residual = out
                out = layer(out)
                out += residual
            else:
                out = layer(out)
        return out

# 디코더 D_G와 D_F
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.initial_layers = nn.Sequential(
            # Residual Blocks ×4
            *[nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.LayerNorm([512, 7, 7]),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.LayerNorm([512, 7, 7])
            ) for _ in range(4)]
        )
        self.upsample_layers = nn.Sequential(
            # 이하 동일하게 구성
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.LayerNorm([512, 14, 14]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
            nn.LayerNorm([256, 28, 28]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.LayerNorm([128, 56, 56]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.LayerNorm([64, 112, 112]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        out = x
        for layer in self.initial_layers:
            residual = out
            out = layer(out)
            out += residual
        out = self.upsample_layers(out)
        return out

# 디코더 D_J
class DecoderConcat(nn.Module):
    def __init__(self):
        super(DecoderConcat, self).__init__()
        self.initial_layers = nn.Sequential(
            # Residual Blocks ×4
            *[nn.Sequential(
                nn.Conv2d(1024 if i == 0 else 512, 512, kernel_size=3, padding=1),
                nn.LayerNorm([512, 7, 7]),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.LayerNorm([512, 7, 7])
            ) for i in range(4)]
        )
        self.upsample_layers = nn.Sequential(
            # 이하 동일하게 구성
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.LayerNorm([512, 14, 14]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
            nn.LayerNorm([256, 28, 28]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.LayerNorm([128, 56, 56]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.LayerNorm([64, 112, 112]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.initial_layers):
            residual = out
            out = layer(out)
            # 첫 번째 Residual Block에서 잔차 연결 조정
            if i == 0 and residual.shape[1] != out.shape[1]:
                # 입력 채널 수를 맞추기 위해 1x1 Conv 사용
                residual = nn.Conv2d(residual.shape[1], out.shape[1], kernel_size=1).to(residual.device)(residual)
            out += residual
        out = self.upsample_layers(out)
        return out

# 판별자 D
class Discriminator(nn.Module):
    def __init__(self, is_batch_normalization=False):
        super(Discriminator, self).__init__()
        
        layers = [
            # 첫 번째 Conv 레이어 (BatchNorm 없음)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 두 번째 Conv 레이어
        layers.append(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        if is_batch_normalization:
            layers.append(nn.BatchNorm2d(128))  # BatchNorm 적용 여부 제어
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 세 번째 Conv 레이어
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        if is_batch_normalization:
            layers.append(nn.BatchNorm2d(256))  # BatchNorm 적용 여부 제어
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 네 번째 Conv 레이어
        layers.append(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        if is_batch_normalization:
            layers.append(nn.BatchNorm2d(512))  # BatchNorm 적용 여부 제어
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 출력 레이어
        layers.append(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0))
        
        # 레이어 리스트를 시퀀셜 모델로 변환
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
