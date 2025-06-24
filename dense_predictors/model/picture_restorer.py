import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention import SelfAttentionWithRoPE
from modules.position_embedding import RotaryPositionEmbeddingND, sinusoidal_embedding


class TransformerBlock(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, mlp_ratio: float, rope_module: nn.Module):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(in_channels * mlp_ratio)

        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.attn = SelfAttentionWithRoPE(in_channels, num_heads, rope_module)
        self.norm2 = nn.LayerNorm(in_channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, self.mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.mlp_hidden_dim, in_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x


class TransformerParallelBlock(nn.Module):
    """A DiT block with parallel linear layers as described in https://arxiv.org/abs/2302.05442"""
    def __init__(self, in_channels: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(in_channels * mlp_ratio)

        self.pre_norm = nn.LayerNorm(in_channels, eps=1e-6)
        # qkv and mlp_in
        self.linear1 = nn.Linear(in_channels, in_channels * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(in_channels + self.mlp_hidden_dim, in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.pre_norm(x)
        x = self.linear1(x)
        qkv, mlp = torch.split(x, [self.in_channels * 3, self.mlp_hidden_dim], dim=-1)
        q, k, v = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q = self.rope_module(q)
        k = self.rope_module(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, "B H L D -> B L (H D)")
        mlp = F.gelu(x, approximate="tanh")
        x = torch.cat((x, mlp), dim=2)
        x = self.linear2(x)
        x = shortcut + x
        return x


class ImageEncoder(nn.Module):
    def __init__(
        self, in_channels: int, num_blocks: int, hidden_size: int, num_heads: int, mlp_ratio: float
    ):
        assert(hidden_size % num_heads == 0)
        self.conv_in = nn.Conv2d(in_channels, hidden_size, 2, stride=2, padding=0)
        pe_dim = hidden_size // num_heads
        self.rope_module = RotaryPositionEmbeddingND(dims_per_axis=[pe_dim // 2, pe_dim // 2])
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, self.rope_module)
            for _ in range(num_blocks)
        ])
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, c, h, w = image.size()
        image_position_id = torch.stack(torch.meshgrid(
            torch.arange(h, device=image.device).unsqueeze(1),
            torch.arange(w, device=image.device).unsqueeze(0)
        ), dim=0)
        image_position_id = einops.repeat(image_position_id, "H W C -> B (H W) C", B=b)
        self.rope_module.compute_rotary_params(image_position_id)

        x = self.conv_in(image)
        b, c, h, w = image.size()
        x = einops.rearrange(x, "B C H W -> B (H W) C")
        for block in self.blocks:
            x = block(x)
        x = einops.rearrange(x, "B (H W) C -> B C H W", H=h, W=w)
        return x


class FlowBlockWithModulation(nn.Module):
    """Transformer block with time-modulated layer norm."""
    def __init__(self, channels: int, num_heads: int, mlp_ratio: float, rope_module: nn.Module):
        self.channels = channels
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_size = int(channels * mlp_ratio)

        self.mod = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(channels, channels * 6)
        )
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttentionWithRoPE(channels, num_heads, rope_module)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Lienar(channels, self.mlp_hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.mlp_hidden_size, channels)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        scale1, shift1, gate1, scale2, shift2, gate2 = self.mod(y).chunk(6, dim=-1)

        shortcut = x
        x = (1 + scale1) * self.norm1(x) + shift1
        x = self.attn(x)
        x = shortcut + gate1 * x
        shortcut = x
        x = (1 + scale2) * self.norm2(x) + shift2
        x = self.mlp(x)
        x = shortcut + gate2 * x

        return x


class LinearWithModulation(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels, in_channels * 2)
        )
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mod(y).chunk(2, dim=-1)
        x = self.norm(x)
        x = (1 + scale) * x + shift
        x = self.linear(x)
        return x


class FlowDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_blocks_per_stage: list[int],
        upsample_per_stage: list[int],
        num_heads: int,
        mlp_ratio: float
    ):
        self.out_channels = out_channels
        self.num_stages = len(num_blocks_per_stage)
        self.num_blocks_per_stage = num_blocks_per_stage
        self.upsample_per_stage = upsample_per_stage
        self.stages = nn.ModuleList()
        self.stage_rope_modules = nn.ModuleList()
        self.stage_upsample_modules = nn.ModuleList()

        for stage in range(self.num_stages):
            blocks = nn.ModuleList()

            if upsample_per_stage[stage] > 1:
                self.stage_upsample_modules.append(
                    nn.ConvTranspose2d(channels, channels // 2, 2, stride=2)
                )
                channels = channels // 2
            else:
                self.stage_upsample_modules.append(nn.Identity())
            
            pe_dim = channels // num_heads
            rope_module = RotaryPositionEmbeddingND(dims_per_axis=[pe_dim // 2, pe_dim // 2])
            self.stage_rope_modules.append(rope_module)
            for _ in range(num_blocks_per_stage[stage]):
                blocks.append(
                    FlowBlockWithModulation(channels, num_heads, mlp_ratio, rope_module)
                )

            self.stages.append(blocks)
        
        self.last_layer = LinearWithModulation(channels, out_channels)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for istage in enumerate(self.stages):
            upsample_module = self.stage_upsample_modules[istage]
            x = upsample_module(x)
            
            b, c, h, w = x.size()
            image_position_id = torch.stack(torch.meshgrid(
                torch.arange(h, device=x.device).unsqueeze(1),
                torch.arange(w, device=x.device).unsqueeze(0)
            ), dim=0)
            image_position_id = einops.repeat(image_position_id, "H W C -> B (H W) C", B=b)
            rope_module = self.stage_rope_modules[istage]
            rope_module.compute_rotary_params(image_position_id)

            x = einops.rearrange(x, "B C H W -> B (H W) C")
            for block in self.stages[istage]:
                x = block(x, t_emb)
            x = einops.rearrange(x, "B (H W) C -> B C H W", H=h, W=w)
        
        x = einops.rearrange(x, "B C H W -> B (H W) C")
        x = self.last_layer(x, t_emb)
        return x


class PictureRestorer(nn.Module):
    def __init__(self, config):
        self.config = config

        self.time_embedder = nn.Sequential(
            nn.Linear(config.time_sinu_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.encoder = ImageEncoder(
            config.in_channels,
            config.encoder_num_blocks,
            config.hidden_size,
            config.encoder_num_heads,
            config.mlp_ratio
        )
        self.flow_decoder = FlowDecoder(
            config.hidden_size,
            config.out_channels,
            config.decoder_num_blocks_per_stage,
            config.decoder_upsample_per_stage,
            config.decoder_num_heads,
            config.mlp_ratio
        )
    
    def forward(self, image: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = self.encoder(image)
        t_emb = sinusoidal_embedding(time, self.config.time_sinu_dim)
        t_emb = self.time_embedder(t_emb)
        x = self.flow_decoder(x, t_emb)
        return x