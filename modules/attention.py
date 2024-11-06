import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.position_embedding import RotaryPositionEmbeddingND


class SelfAttentionWithRoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_module: nn.Module):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.pe_dim = dim // num_heads

        self.rope_module = rope_module
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q = self.rope_module(q)
        k = self.rope_module(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, "B H L D -> B L (H D)")
        x = self.proj(x)
        return x


class AttentionWithRoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_module: nn.Module):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.pe_dim = dim // num_heads

        self.rope_module = rope_module
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = einops.rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
        k = einops.rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
        v = einops.rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)
        q = self.rope_module(q)
        k = self.rope_module(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, "B H L D -> B L (H D)")
        x = self.proj(x)
        return x