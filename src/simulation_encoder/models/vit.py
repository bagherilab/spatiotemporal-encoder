from typing import Any

import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, image_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, E, H/P, W/P)
        x = self.proj(x)
        B, E, H_p, W_p = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H_p * W_p, E)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = (x + attn_out)
        x = (x + self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=128,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.0,
        out_dim=128,
        pool="mean",
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, image_size, patch_size)
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.pool = pool
        self.head = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        # 1. Patchify
        x = self.patch_embed(x) # (B, N, E)
        
        # 2. Positional encoding
        x = (x + self.pos_embed).contiguous()
        
        # 3. Transformer Layers
        for block in self.blocks:
            x = block(x)
            
        # 4. Final Norm
        x = self.norm(x)
        
        # 5. Pooling
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else: # CLS or Max
            x = x[:, 0]
            
        # 6. Latent Projection
        # Use .contiguous() here so the Decoder's Unflatten layer gets a clean tensor
        return self.head(x).contiguous()


class VisionTransformerDecoder(nn.Module):
    """
    Decoder that mirrors the ViT encoder: expand latent to patch sequence,
    transformer blocks, then unpatchify to image (B, C, H, W).
    """

    def __init__(
        self,
        latent_dim: int,
        image_size: int,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.image_size = image_size
        H_p = image_size // patch_size
        W_p = image_size // patch_size
        self.num_patches = H_p * W_p

        self.proj_in = nn.Linear(latent_dim, self.num_patches * embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.unpatchify = nn.ConvTranspose2d(
            embed_dim, in_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, latent_dim)
        B = x.shape[0]
        x = self.proj_in(x)  # (B, num_patches * embed_dim)
        x = x.reshape(B, self.num_patches, self.embed_dim)
        x = (x + self.pos_embed).contiguous()

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # (B, N, E) -> (B, E, H_p, W_p)
        H_p = self.image_size // self.patch_size
        W_p = self.image_size // self.patch_size
        x = x.permute(0, 2, 1).reshape(B, self.embed_dim, H_p, W_p)
        x = self.unpatchify(x)  # (B, in_channels, image_size, image_size)
        return x


def build_vision_transformer(config: dict[str, Any], context: dict[str, int]) -> nn.Module:
    """
    Build a VisionTransformerEncoder from a layer config and context.

    Context must contain: in_channels, image_size, out_dim.
    Config may contain: patch_size, embed_dim, depth, num_heads, mlp_ratio, dropout, pool.
    """
    in_channels = context["in_channels"]
    image_size = context["image_size"]
    out_dim = context["out_dim"]
    return VisionTransformer(
        in_channels=in_channels,
        image_size=image_size,
        out_dim=out_dim,
        patch_size=config.get("patch_size", 16),
        embed_dim=config.get("embed_dim", 192),
        depth=config.get("depth", 6),
        num_heads=config.get("num_heads", 3),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        dropout=config.get("dropout", 0.0),
        pool=config.get("pool", "mean"),
    )


def build_vision_transformer_decoder(
    config: dict[str, Any], context: dict[str, int]
) -> nn.Module:
    """
    Build a VisionTransformerDecoder from a layer config and context.

    Context must contain: latent_dim, image_size, in_channels.
    Config may contain: patch_size, embed_dim, depth, num_heads, mlp_ratio, dropout.
    """
    latent_dim = context["latent_dim"]
    image_size = context["image_size"]
    in_channels = context["in_channels"]
    return VisionTransformerDecoder(
        latent_dim=latent_dim,
        image_size=image_size,
        in_channels=in_channels,
        patch_size=config.get("patch_size", 16),
        embed_dim=config.get("embed_dim", 192),
        depth=config.get("depth", 6),
        num_heads=config.get("num_heads", 3),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        dropout=config.get("dropout", 0.0),
    )
