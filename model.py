import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def conv_nd(dims, in_channels, out_channels, kernel_size, stride=1, padding=0):
    if dims == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    elif dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    elif dims == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")

def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels, num_groups=32):
    return nn.GroupNorm(num_groups, channels)

# Attention Block implementation
class AttentionBlock(nn.Module):
    """
    Attention block for spatial information
    """

    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()
        self.channels = channels

        if num_head_channels != -1:
            assert channels % num_head_channels == 0, "channels must be divisible by num_head_channels"
            self.num_heads = channels // num_head_channels
        else:
            self.num_heads = num_heads

        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, kernel_size=1))
        self.use_checkpoint = use_checkpoint


    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x_in = x
        x = x.reshape(b, c, -1)
        x = self.norm(x)
        qkv = self.qkv(x)
        h = self._attention(qkv)
        h = self.proj_out(h)
        h = h.reshape(b, c, *spatial)
        return x_in + h

    def _attention(self, qkv):
        b, c, n = qkv.shape
        qkv = qkv.reshape(b * self.num_heads, -1, n)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)
        scale = 1 / torch.sqrt(torch.sqrt(q.new_tensor(c // (3 * self.num_heads))))

        # 어텐션 계산
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight, dim=-1)
        h = torch.einsum("bts,bcs->bct", weight, v)

        return h

class BigGANResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        up=False,
        down=False,
        num_groups=32,
        use_checkpoint=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.use_checkpoint = use_checkpoint


        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.activation = nn.SiLU()
        self.conv1 = conv_nd(2, in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = conv_nd(2, out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels or up or down:
            self.skip_conv = conv_nd(2, in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

        if up:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        elif down:
            self.downsample = nn.AvgPool2d(2)
        else:
            self.upsample = self.downsample = nn.Identity()

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        h = x

        # 첫 번째 블록
        h = self.norm1(h)
        h = self.activation(h)
        if self.up:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)

        # 두 번째 블록
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)

        if self.down:
            h = self.downsample(h)
            x = self.downsample(x)

        # 스킵 연결
        x = self.skip_conv(x)
        return h + x

# 새로운 인코더 클래스
class NewEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        model_channels=64,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        num_heads=1,
        num_head_channels=64,
        num_groups=32,
        use_checkpoint=False,
    ):
        super(NewEncoder, self).__init__()
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint

        self.input_conv = conv_nd(2, in_channels, model_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch = model_channels
        ds = 1  # 현재 다운샘플링 비율
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    BigGANResBlock(
                        in_channels=ch,
                        out_channels=out_ch,
                        down=False,
                        num_groups=num_groups,
                        use_checkpoint=use_checkpoint,
                    )
                )
                ch = out_ch
                # 어텐션 블록 추가
                resolution = 256 // ds
                if resolution in attention_resolutions:
                    self.down_blocks.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                        )
                    )
            if level != len(channel_mult) - 1:
                self.down_blocks.append(
                    BigGANResBlock(
                        in_channels=ch,
                        out_channels=ch,
                        down=True,
                        num_groups=num_groups,
                        use_checkpoint=use_checkpoint,
                    )
                )
                ds *= 2  # 다운샘플링 비율 업데이트

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        h = self.input_conv(x)
        for block in self.down_blocks:
            if self.use_checkpoint and self.training:
                h = checkpoint.checkpoint(block, h)
            else:
                h = block(h)
        return h

# 새로운 디코더 클래스
class NewDecoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        model_channels=64,
        channel_mult=(8, 4, 2, 1),
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        num_heads=1,
        num_head_channels=64,
        num_groups=32,
        use_checkpoint=False,
    ):
        super(NewDecoder, self).__init__()
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint  # 저장

        self.up_blocks = nn.ModuleList()
        ch = model_channels * channel_mult[0]
        ds = 2 ** (len(channel_mult) - 1)  # 현재 업샘플링 비율

        for level in range(len(channel_mult)):
            mult = channel_mult[level]
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(
                    BigGANResBlock(
                        in_channels=ch,
                        out_channels=out_ch,
                        up=False,
                        num_groups=num_groups,
                        use_checkpoint=use_checkpoint,
                    )
                )
                ch = out_ch
                # 어텐션 블록 추가
                resolution = 256 // ds
                if resolution in attention_resolutions:
                    self.up_blocks.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                        )
                    )
            if level != len(channel_mult) - 1:
                self.up_blocks.append(
                    BigGANResBlock(
                        in_channels=ch,
                        out_channels=ch,
                        up=True,
                        num_groups=num_groups,
                        use_checkpoint=use_checkpoint,
                    )
                )
                ds //= 2  # 업샘플링 비율 업데이트

        self.output_norm = nn.GroupNorm(num_groups, ch)
        self.output_activation = nn.SiLU()
        self.output_conv = zero_module(conv_nd(2, ch, out_channels, 3, padding=1))

    def forward(self, h):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward, h)
        else:
            return self._forward(h)

    def _forward(self, x):
        h = x
        for block in self.up_blocks:
            if self.use_checkpoint and self.training:
                h = checkpoint.checkpoint(block, h)
            else:
                h = block(h)
        h = self.output_norm(h)
        h = self.output_activation(h)
        h = self.output_conv(h)
        return h

# Concat을 위한 디코더
class NewDecoderConcat(nn.Module):
    def __init__(
        self,
        out_channels=1,
        model_channels=64,
        channel_mult=(8, 4, 2, 1),
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        num_heads=1,
        num_head_channels=64,
        num_groups=32,
        use_checkpoint=False,
    ):
        super(NewDecoderConcat, self).__init__()
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint  # 저장

        self.up_blocks = nn.ModuleList()
        ch = model_channels * channel_mult[0] * 2  # 두 인코더의 출력을 concat 하므로 *2
        ds = 2 ** (len(channel_mult) - 1)

        for level in range(len(channel_mult)):
            mult = channel_mult[level]
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(
                    BigGANResBlock(
                        in_channels=ch,
                        out_channels=out_ch,
                        up=False,
                        num_groups=num_groups,
                        use_checkpoint=use_checkpoint,
                    )
                )
                ch = out_ch
                # 어텐션 블록 추가
                resolution = 256 // ds
                if resolution in attention_resolutions:
                    self.up_blocks.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                        )
                    )
            if level != len(channel_mult) - 1:
                self.up_blocks.append(
                    BigGANResBlock(
                        in_channels=ch,
                        out_channels=ch,
                        up=True,
                        num_groups=num_groups,
                        use_checkpoint=use_checkpoint,
                    )
                )
                ds //= 2

        self.output_norm = nn.GroupNorm(num_groups, ch)
        self.output_activation = nn.SiLU()
        self.output_conv = zero_module(conv_nd(2, ch, out_channels, 3, padding=1))

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        h = x
        for block in self.up_blocks:
            if self.use_checkpoint and self.training:
                h = checkpoint.checkpoint(block, h)
            else:
                h = block(h)
        h = self.output_norm(h)
        h = self.output_activation(h)
        h = self.output_conv(h)
        return h


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
