"""LadaGAN model for PyTorch.

Reference:
  - [Efficient generative adversarial networks using linear 
    additive-attention Transformers](https://arxiv.org/abs/2401.09596)

It is more easy to understand regarding tensor shape as (B, L, E) rather than N: (H,W) and C.
Sequence length = H, W
Embedding dim = C
(B, C, N) is basically used in pytorch.
But, we are using (B, N, C) to avoid reshape operator before attention layer.
Be careful about tensor shape.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# pixel shuffle
# input x: (B, N, C)
# output x: (B, C, H, W)
def pixel_upsample(x, H, W):
    B, N, C = x.size() # (B, N, C)
    assert N == H * W
    x = x.permute(0, 2, 1).view(B, C, H, W) # (B, N, C) -> (B, C, N) -> (B, C, H, W)
    x = F.pixel_shuffle(x, upscale_factor=2)
    B, C, H, W = x.size()
    return x, H, W, C # x: (B, C, H, W)

# Tensorflow style normalization layer for fixed axis normalization
class CustomNorm(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super(CustomNorm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        var = x.var(dim=self.dim, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)

# Modified self modulation. Self modulation -> https://openreview.net/pdf?id=Hkl5aoR5tm
# Normalization dim: C
class SMLayerNormalization(nn.Module):
    def __init__(self, input_channel, latent_feature, epsilon=1e-6):
        super(SMLayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.h = nn.Linear(latent_feature, input_channel, bias=True) # z: (B, N) -> (B, C)
        self.gamma = nn.Linear(input_channel, input_channel, bias=True)
        self.beta = nn.Linear(input_channel, input_channel, bias=True)
        self.ln = CustomNorm(dim=-1) # Channel normalization, instead of pytorch layernorm

    def forward(self, x, z):
        # x: (B, N, C)
        # z: generator input(embedding from encoder) (B, N)
        x = self.ln(x) # (B, N, C) -> (B, N, C): Normalize C for each B, N without learnable parameters
        h = self.h(z) # (B, N) -> (B, C)
        h = F.relu(h) # (B, C) -> (B, C)
        scale = self.gamma(h).unsqueeze(1) # (B, C) -> (B, 1, C)
        shift = self.beta(h).unsqueeze(1) # (B, C) -> (B, 1, C)
        x = x * scale + shift 
        return x

# model_dim: C
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
        # input x: (B, N, C)
        x = x.view(batch_size, -1, self.n_heads, self.depth) # (B, L, h, d_k) or (B, N, h, d_k=C/h)
        return x.permute(0, 2, 1, 3) # (B, h, N, d_k=C/h) 

    def forward(self, q, k, v):
        B = q.size(0)
        q = self.wq(q) # (B, N, C) 
        k = self.wk(k) # (B, N, C)
        v = self.wv(v) # (B, N, C)
        # Attention weight of the i-th query vector in additive attention
        # From fastformer https://arxiv.org/pdf/2108.09084
        attn = self.q_attn(q).permute(0, 2, 1) / (self.depth ** 0.5) # (B, h, N)
        attn = F.softmax(attn, dim=-1) # softmax for N dimension (B, h, N)

        q = self.split_into_heads(q, B) # (B, h, N, d_k) 
        k = self.split_into_heads(k, B) # (B, h, N, d_k) 
        v = self.split_into_heads(v, B) # (B, h, N, d_k) 

        global_q = torch.einsum('b h n, b h n d -> b h d', attn, q).unsqueeze(2) # (B, h, 1, d_k)
        p = global_q * k # (B, h, N, d_k)
        # (value * p) instead of global key calculation(different from original fastformer)
        # to propagate attn directly instead of compressing
        # in original fastformer, key is calculated from sum((w_k * p_i softmax score) * p_i)
        r = p * v # (B, h, N, d_k)

        r = r.permute(0, 2, 1, 3) # (B, N, h, d_k)
        original_size_attention = r.reshape(B, -1, self.model_dim) # concat: (B, N, h*d_k=C)

        output = self.to_out(original_size_attention) # (B, N, C)
        return output, attn


class SMLadaformer(nn.Module):
    def __init__(self, model_dim, latent_feature, n_heads=2, mlp_dim=512, 
                 rate=0.0, eps=1e-6):
        super(SMLadaformer, self).__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, model_dim)
        )
        self.norm1 = SMLayerNormalization(model_dim, latent_feature, epsilon=eps)
        self.norm2 = SMLayerNormalization(model_dim, latent_feature, epsilon=eps)
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
    def __init__(self, z_dim=100, img_size=256, model_dim=[1024, 512, 256], heads=[4, 4, 4], 
                 mlp_dim=[512, 512, 512], dec_dim=False, relu_activation=False):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.init = nn.Sequential(
            nn.Linear(z_dim, 8 * 8 * model_dim[0], bias=False),
        )
        self.pos_emb_8 = PositionalEmbedding(8*8, model_dim[0])
        self.block_8 = SMLadaformer(model_dim[0], z_dim, heads[0], mlp_dim[0])

        self.conv_8 = nn.Conv2d(model_dim[0] // 4, model_dim[1], kernel_size=3, padding=1)
        self.pos_emb_16 = PositionalEmbedding(16*16, model_dim[1])
        self.block_16 = SMLadaformer(model_dim[1], z_dim, heads[1], mlp_dim[1])

        self.conv_16 = nn.Conv2d(model_dim[1] // 4, model_dim[2], kernel_size=3, padding=1)
        self.pos_emb_32 = PositionalEmbedding(32*32, model_dim[2])
        self.block_32 = SMLadaformer(model_dim[2], z_dim, heads[2], mlp_dim[2])

        self.dec_dim = dec_dim
        self.relu_activation = relu_activation
        if self.dec_dim:
            layers = []
            dim_before = model_dim[2]
            for dim in self.dec_dim:
                layers.extend([
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(dim_before, dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU(0.2)
                ])
                dim_before = dim
            self.dec = nn.Sequential(*layers)
            conv_in_dim = self.dec_dim[-1]
        else:
            self.patch_size = img_size // 32
            conv_in_dim = model_dim[2] // (self.patch_size ** 2)

        # We use grayscale image. If you want to use RGB image, edit conv2d output channel to 3.
        self.ch_conv = nn.Conv2d(conv_in_dim, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, z):
        B = z.size(0)
        x = self.init(z).view(B, 8 * 8, -1) # (B, L, d_model): (B, N, C)
        x = self.pos_emb_8(x)
        x, attn_8 = self.block_8(x, z)
        x, H, W, C = pixel_upsample(x, 8, 8)
        x = self.conv_8(x)
        x = x.permute(0, 2, 3, 1).view(B, H * W, -1)
        x = self.pos_emb_16(x)
        x, attn_16 = self.block_16(x, z)
        x, H, W, C = pixel_upsample(x, H, W)
        x = self.conv_16(x)
        x = x.permute(0, 2, 3, 1).view(B, H * W, -1)
        x = self.pos_emb_32(x)
        x, attn_32 = self.block_32(x, z)
        x = x.permute(0, 2, 1).view(B, -1, H, W)
        if self.dec_dim:
            x = self.dec(x)
        elif self.patch_size != 1:
            x = F.pixel_shuffle(x, upscale_factor=self.patch_size)
        x = self.ch_conv(x)
        if self.relu_activation:
            x = self.relu(x)
        return [x, [attn_8, attn_16, attn_32]]


class downBlock(nn.Module):
    def __init__(self, filters, kernel_size=3, strides=2):
        super(downBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(filters // 2, filters, kernel_size=kernel_size, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False),
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
                 heads=4):
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
            # We use grayscale image. If you want to use RGB image, edit conv2d input channel to 3.
            nn.Conv2d(1, enc_dim[0], kernel_size=3, stride=1, padding=1, bias=False),
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
        B, C, H, W = x.size() # (B, C, N)
        x = x.view(B, C, -1).permute(0, 2, 1) # (B, N, C)
        x = self.pos_emb_8(x)
        x, maps_16 = self.block_8(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_4(x)
        x = self.down_4(x)
        return self.logits(x)



if __name__=="__main__":
    # Random input tensors for testing
    z = torch.randn((4, 128))  # Batch size of 4, latent vector of 128 dimensions
    img = torch.randn((4, 1, 256, 256))  # Batch size of 4, image size 32x32, 3 channels

    # Initialize Generator and Discriminator
    generator = Generator(z_dim=z.shape[-1], img_size=256, model_dim=[1024, 512, 256], dec_dim=[32, 16, 8])
    discriminator = Discriminator(img_size=256, enc_dim=[64, 128, 256, 512, 1024], heads=4)

    # Test Generator output
    gen_output, gen_attn_maps = generator(z)
    print("Generator output shape:", gen_output.shape)
    print("Generator attention maps shapes:", [attn.shape for attn in gen_attn_maps])

    # Test Discriminator output
    disc_output = discriminator(gen_output)
    print("Discriminator output shape:", disc_output.shape)

