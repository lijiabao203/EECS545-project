import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, N, C = x.size()
        qkv = self.qkv_proj(x)  
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]  
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1, is_causal=False)  
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, C)  
        x = x + self.out_proj(attn_output)
        x = self.norm1(x)

        x = x + self.mlp(x)
        x = self.norm2(x)

        return x


class VisionTransformer3D(nn.Module):
    def __init__(
        self,
        input_size=(182, 218, 182),
        num_channels=1,
        num_classes=1,
        patch_size=(32, 32, 32),
        embedding_dim=512,
        num_heads=8,
        num_layers=8
    ):
        super(VisionTransformer3D, self).__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        D, H, W = input_size
        pD, pH, pW = patch_size
        self.num_patches = (D // pD) * (H // pH) * (W // pW)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.patch_embed = nn.Conv3d(
            in_channels=num_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.transformer_layers = nn.ModuleList([
            FlashTransformerEncoderLayer(embedding_dim, num_heads) for _ in range(num_layers)
        ])

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)  
        x = x.flatten(2).transpose(1, 2)  
        x = x + self.pos_embed     

        for layer in self.transformer_layers:
            x = layer(x)

        x = x.mean(dim=1) 
        x = self.fc(x)
        return x