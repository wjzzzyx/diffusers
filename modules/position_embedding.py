import math

import einops
import torch
import torch.nn as nn


class RotaryPositionEmbeddingND(nn.Module):
    "https://arxiv.org/abs/2104.09864"
    def __init__(self, dims_per_axis: list[int]):
        super().__init__()
        self.dims_per_axis = dims_per_axis
        self.dim = sum(dims_per_axis)
        self.c = 10000
    
    def compute_rotary_params(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids: shape (batch, seq, num_axes)
        num_axes = position_ids.size(-1)
        assert(num_axes == len(self.dims_per_axis))
        all_axes_params = list()
        for axis in num_axes:
            axis_params = self.rope(position_ids[..., axis], self.dims_per_axis[axis])
            all_axes_params.append(axis_params)
        self.coefficients = torch.cat(all_axes_params, dim=-3)
        self.coefficients = self.coefficients.unsqueeze(1)    # shape (batch, head, seq, dim // 2, 2, 2)
    
    def rope(self, pos: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim % 2 == 0
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (self.c ** scale)
        out = torch.einsum("...n,d->...nd", pos, omega)
        out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        # shape (batch, seq, dim // 2, 2, 2)
        out = einops.rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
        return out.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (batch, seq, dim)
        dim = x.size(-1)
        x = x.float().reshape(*x.shape[:-1], dim // 2, 1, 2)
        x_emb = self.coefficients[..., 0] * x[..., 0] + self.coefficients[..., 1] * x[..., 1]
        x_emb = x_emb.reshape(*x.shape).type_as(x)
        return x_emb


def sinusoidal_embedding(time: torch.Tensor, dim):
    # time: shape (batch,), value in [0, 1]
    factor = 1000
    T = 10000
    assert dim % 2 == 0

    time = time * 1000
    freqs = torch.exp(-torch.log(T) * torch.arange(0, dim // 2, dtype=torch.float, device=time.device) * 2 / dim)
    args = time.unsqueeze(1).float() * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding.to(time.dtype)