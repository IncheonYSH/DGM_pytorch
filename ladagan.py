"""LadaGAN model for PyTorch.

Reference:
  - [Efficient generative adversarial networks using linear 
    additive-attention Transformers](https://arxiv.org/abs/2401.09596)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# def pixel_upsample(x, H, W):
#     print(x.shape)
#     B, N, C = x.size() # (B, N, C)
#     assert N == H * W
#     x = x.permute(0, 2, 1).view(B, C, H, W) # (B, N, C) -> (B, C, N) -> (B, C, H, W)
#     x = F.pixel_shuffle(x, upscale_factor=2)
#     B, C, H, W = x.size()
#     return x, H, W, C

def pixel_upsample(x, H, W):
    print("upsample_x", x.shape)
    B, C, N = x.size() # (B, C, N)
    assert N == H * W
    x = x.reshape(B, C, H, W) # (B, C, N) -> (B, C, H, W)
    x = F.pixel_shuffle(x, upscale_factor=2) # (*, C * r^2, H, W) -> (*, C, H * r, W * r)
    B, C, H, W = x.size()
    return x, C, H, W


class SMLayerNormalization(nn.Module):
    def __init__(self, input_dim, epsilon=1e-6):
        super(SMLayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.h = nn.Linear(input_dim, input_dim, bias=True)
        self.gamma = nn.Linear(input_dim, input_dim, bias=True)
        self.beta = nn.Linear(input_dim, input_dim, bias=True)
        self.ln = nn.LayerNorm(input_dim, eps=epsilon, elementwise_affine=False) # Affine=False for customized scaling and shifting

    def forward(self, x, z): # z -> generator input(embedding from encoder) (B, N)
        print("before norm x:", x.shape)
        print("z:", z.shape)
        x = self.ln(x)
        h = self.h(z)
        h = F.relu(h)
        scale = self.gamma(h).unsqueeze(2) # tensorflow: expand_dim(scale/shift, 1) -> (B, N, C) in tensorflow
        shift = self.beta(h).unsqueeze(2)
        x = x * scale + shift
        return x


class AdditiveAttention(nn.Module):
    def __init__(self, model_dim, n_heads):
        super(AdditiveAttention, self).__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim

        assert model_dim % self.n_heads == 0

        self.depth = model_dim // self.n_heads

        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        
        self.q_attn = nn.Linear(model_dim, n_heads)

        self.to_out = nn.Linear(model_dim, model_dim)

    def split_into_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        B = q.size(0)
        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  
        attn = self.q_attn(q).permute(0, 2, 1) / (self.depth ** 0.5)
        attn = F.softmax(attn, dim=-1)  

        q = self.split_into_heads(q, B)  
        k = self.split_into_heads(k, B)  
        v = self.split_into_heads(v, B)

        global_q = torch.einsum('b h n, b h n d -> b h d', attn, q).unsqueeze(2)
        p = global_q * k 
        r = p * v

        r = r.permute(0, 2, 1, 3)
        original_size_attention = r.contiguous().view(B, -1, self.model_dim)

        output = self.to_out(original_size_attention)
        return output, attn


class SMLadaformer(nn.Module):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, 
                 rate=0.0, eps=1e-6):
        super(SMLadaformer, self).__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, model_dim)
        )
        self.norm1 = SMLayerNormalization(model_dim, epsilon=eps)
        self.norm2 = SMLayerNormalization(model_dim, epsilon=eps)
        self.drop1 = nn.Dropout(rate)
        self.drop2 = nn.Dropout(rate)

    def forward(self, inputs, z):
        x_norm1 = self.norm1(inputs, z)
        attn_output, attn_maps = self.attn(x_norm1, x_norm1, x_norm1)
        attn_output = inputs + self.drop1(attn_output) 
        x_norm2 = self.norm2(attn_output, z)
        mlp_output = self.mlp(x_norm2)
        output = attn_output + self.drop2(mlp_output)
        return output, attn_maps


class PositionalEmbedding(nn.Module):
    def __init__(self, n_patches, model_dim):
        super(PositionalEmbedding, self).__init__()
        self.n_patches = n_patches
        self.position_embedding = nn.Embedding(n_patches, model_dim)

    def forward(self, patches):
        positions = torch.arange(0, self.n_patches, device=patches.device)
        position_embeddings = self.position_embedding(positions)
        return patches + position_embeddings.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, z_dim=100, img_size=32, model_dim=[1024, 256, 64], heads=[2, 2, 2], 
                 mlp_dim=[2048, 1024, 512], dec_dim=False):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.init = nn.Sequential(
            nn.Linear(z_dim, 8 * 8 * model_dim[0], bias=False),
        )
        self.pos_emb_8 = PositionalEmbedding(8*8, model_dim[0])
        self.block_8 = SMLadaformer(model_dim[0], heads[0], mlp_dim[0])

        self.conv_8 = nn.Conv2d(model_dim[0], model_dim[1], kernel_size=3, padding=1)
        self.pos_emb_16 = PositionalEmbedding(16*16, model_dim[1])
        self.block_16 = SMLadaformer(model_dim[1], heads[1], mlp_dim[1])

        self.conv_16 = nn.Conv2d(model_dim[1], model_dim[2], kernel_size=3, padding=1)
        self.pos_emb_32 = PositionalEmbedding(32*32, model_dim[2])
        self.block_32 = SMLadaformer(model_dim[2], heads[2], mlp_dim[2])

        self.dec_dim = dec_dim
        if self.dec_dim:
            layers = []
            for dim in self.dec_dim:
                layers.extend([
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU(0.2)
                ])
            self.dec = nn.Sequential(*layers)
        else:
            self.patch_size = img_size // 32

        self.ch_conv = nn.Conv2d(model_dim[2], 3, kernel_size=3, padding=1)

    def forward(self, z):
        B = z.size(0)
        x = self.init(z).view(B, 8*8, -1)  # (B, N, C)
        x = self.pos_emb_8(x)
        x, attn_8 = self.block_8(x, z)
        x, H, W, _ = pixel_upsample(x, 8, 8)
        x = self.conv_8(x)
        x_flat = x.view(B, x.size(1), -1).permute(0, 2, 1)
        x_flat = self.pos_emb_16(x_flat)
        x_flat, attn_16 = self.block_16(x_flat, z)
        x, H, W, _ = pixel_upsample(x_flat, H, W)
        x = self.conv_16(x)
        x_flat = x.view(B, x.size(1), -1).permute(0, 2, 1)
        x_flat = self.pos_emb_32(x_flat)
        x_flat, attn_32 = self.block_32(x_flat, z)
        x = x_flat.permute(0, 2, 1).view(B, -1, H, W)
        if self.dec_dim:
            x = self.dec(x)
        elif self.patch_size != 1:
            x = F.pixel_shuffle(x, upscale_factor=self.patch_size)
        x = self.ch_conv(x)
        return [x, [attn_8, attn_16, attn_32]]


class downBlock(nn.Module):
    def __init__(self, filters, kernel_size=3, strides=2):
        super(downBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(filters // 2, filters, kernel_size=kernel_size, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2)
        )
        self.direct = nn.Sequential(
            nn.AvgPool2d(kernel_size=strides, stride=strides),
            nn.Conv2d(filters // 2, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x1 = self.main(x)
        x2 = self.direct(x)
        return (x1 + x2) / 2


class Ladaformer(nn.Module):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, 
                 rate=0.0, eps=1e-6):
        super(Ladaformer, self).__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim, eps=eps)
        self.norm2 = nn.LayerNorm(model_dim, eps=eps)
        self.drop1 = nn.Dropout(rate)
        self.drop2 = nn.Dropout(rate)

    def forward(self, inputs):
        x_norm1 = self.norm1(inputs)
        attn_output, attn_maps = self.attn(x_norm1, x_norm1, x_norm1)
        attn_output = inputs + self.drop1(attn_output) 
        x_norm2 = self.norm2(attn_output)
        mlp_output = self.mlp(x_norm2)
        output = attn_output + self.drop2(mlp_output)
        return output, attn_maps


class Discriminator(nn.Module):
    def __init__(self, img_size=32, enc_dim=[64, 128], out_dim=[512, 1024], mlp_dim=512, 
                 heads=2):
        super(Discriminator, self).__init__()
        if img_size == 32:
            assert len(enc_dim) == 2, "Incorrect length of enc_dim for img_size 32"
        elif img_size == 64:
            assert len(enc_dim) == 3, "Incorrect length of enc_dim for img_size 64"
        elif img_size == 128:
            assert len(enc_dim) == 4, "Incorrect length of enc_dim for img_size 128"
        elif img_size == 256:
            assert len(enc_dim) == 5, "Incorrect length of enc_dim for img_size 256"
        else:
            raise ValueError(f"img_size = {img_size} not supported")
        self.enc_dim = enc_dim
        self.inp_conv = nn.Sequential(
            nn.Conv2d(3, enc_dim[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.encoder = nn.ModuleList([
            downBlock(enc_dim[i], kernel_size=3, strides=2) for i in range(1, len(enc_dim))
        ])
        self.pos_emb_8 = PositionalEmbedding(16*16, enc_dim[-1])
        self.block_8 = Ladaformer(enc_dim[-1], heads, mlp_dim)
        self.conv_4 = nn.Conv2d(enc_dim[-1]*4, out_dim[0], kernel_size=3, padding=1)
        self.down_4 = nn.Sequential(
            nn.Conv2d(out_dim[0], out_dim[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_dim[1], 1, kernel_size=4, stride=1, padding=0, bias=False)
        )
        self.logits = nn.Sequential(
            nn.Flatten(),
            nn.Identity()
        )

    def forward(self, img):
        x = self.inp_conv(img)
        for layer in self.encoder:
            x = layer(x)
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        x_flat = self.pos_emb_8(x_flat)
        x_flat, maps_16 = self.block_8(x_flat)
        x = x_flat.permute(0, 2, 1).view(B, C, H, W)
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_4(x)
        x = self.down_4(x)
        return [self.logits(x)]



if __name__=="__main__":

    # 테스트용 Generator 및 Discriminator 초기화
    generator = Generator(z_dim=128, img_size=256)
    discriminator = Discriminator(img_size=32)

    # 테스트용 입력 생성
    z = torch.randn(1, 128)            # Generator 입력 (배치 크기 1, z_dim 128)
    img = torch.randn(1, 3, 32, 32)    # Discriminator 입력 (배치 크기 1, 3채널 32x32 이미지)

    # Generator 테스트 (출력 이미지와 어텐션 맵을 생성)
    gen_img, gen_attn_maps = generator(z)
    print("Generator output shape:", gen_img.shape)  # 예상 출력: (1, 3, 32, 32)
    print("Generator attention maps:", [attn.shape for attn in gen_attn_maps])

    # Discriminator 테스트 (이미지 판별 출력)
    disc_output = discriminator(img)
    print("Discriminator output shape:", disc_output.shape)  # 예상 출력: (1, 1, H_out, W_out)

