# %%
import math
from typing import Optional

import torch
import torch.nn as nn


# %%
class MLP(nn.Module):
    def __init__(self, width: int, in_channels: Optional[int] = None, out_channels: Optional[int] = None):
        super().__init__()
        if in_channels is None:
            in_channels = width
        if out_channels is None:
            out_channels = width
        self.width = width
        self.c_fc = nn.Linear(in_channels, width * 4)
        self.c_proj = nn.Linear(width * 4, out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        dropout=0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=width, num_heads=heads, batch_first=True, dropout=dropout)
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP(width=width)
        self.ln_3 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)  # Dropout for MLP output

    def forward(self, x: torch.Tensor, kv: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None):
        q = self.ln_1(x)
        kv = self.ln_2(kv)
        x = x + self.attn(q, kv, kv, key_padding_mask)[0]
        x = x + self.dropout(self.mlp(self.ln_3(x)))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        dropout=0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=width, num_heads=heads, batch_first=True, dropout=dropout)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width)
        self.ln_2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)  # Dropout for MLP output

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        qkv = self.ln_1(x)
        x = x + self.attn(qkv, qkv, qkv, key_padding_mask)[0]
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


class SelfAttentionBlocks(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        layers: int,
        dropout=0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width=width,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        for block in self.resblocks:
            x = block(x, key_padding_mask)
        return x


class CrossAttentionBlocks(nn.Module):
    """
    Only does cross attention
    """

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    width=width,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        for block in self.resblocks:
            q = block(q, kv, key_padding_mask=key_padding_mask)
        return q


def sinusoidal_positional_encoding(length, d_model):
    """Positional encoding for transformer models using PyTorch.
    Args:
      length: Length of the sequence.
      d_model: Depth of the model. Must be an even number.
    Returns:
      pos_encoding: Positional encoding of shape (length, depth).
    """
    depth = d_model // 2
    pos_encoding = torch.zeros(length, d_model)
    positions = torch.arange(length).unsqueeze(1).float()  # (seq, 1)
    depths = torch.arange(depth).unsqueeze(0) / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding[:, 0::2] = torch.sin(angle_rads)  # even indices
    pos_encoding[:, 1::2] = torch.cos(angle_rads)  # odd indices

    return pos_encoding
