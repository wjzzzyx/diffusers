from dataclasses import dataclass
import einops
import lightning
import math
import numpy as np
import os
from PIL import Image
import safetensors
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel, T5Tokenizer, T5EncoderModel
from typing import Union

from .vae import Encoder, Decoder
from diffusers.model import ema
import utils


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = einops.rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


def timestep_embedding(t: torch.Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = einops.rearrange(x, "B H L D -> B L (H D)")

    return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, Union[ModulationOut, None]]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = einops.rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = einops.rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Union[float, None] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )
        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(f"Got {config.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if config.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                )
                for _ in range(config.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio)
                for _ in range(config.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.config.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            in_channels=config.in_channels,
            ch=config.base_channels,
            ch_mult=config.ch_mult,
            z_channels=config.z_channels,
            num_res_blocks=config.num_res_blocks,
            resolution=config.resolution,
            attn_type='flux',
            attn_resolutions=[]
        )
        self.decoder = Decoder(
            ch=config.base_channels,
            out_ch=config.out_channels,
            z_channels=config.z_channels,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            resolution=config.resolution,
            attn_type='flux',
            attn_resolutions=[]
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class FluxModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flow_model = Flux(config.flow).to(torch.bfloat16)
        if 'pretrained' in config.flow:
            checkpoint = safetensors.torch.load_file(config.flow.pretrained, device='cpu')
            missing, unexpected = self.flow_model.load_state_dict(checkpoint, strict=False, assign=True)

        self.ae = AutoEncoder(config.ae)
        if 'pretrained' in config.ae:
            checkpoint = safetensors.torch.load_file(config.ae.pretrained, device='cpu')
            missing, unexpected = self.ae.load_state_dict(checkpoint, strict=False, assign=True)

        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", max_length=128)
        self.t5_text_model = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl", torch_dtype=torch.bfloat16)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", max_length=77)
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)

        self.flow_model.eval()
        self.ae.eval()
        self.ae.requires_grad_(False)
        self.t5_text_model.eval()
        self.t5_text_model.requires_grad_(False)
        self.clip_text_model.eval()
        self.clip_text_model.requires_grad_(False)
    
    @torch.inference_mode()
    def sample(
        self, img: torch.Tensor, img_ids: torch.Tensor, txt: torch.Tensor, txt_ids: torch.Tensor,
        vec: torch.Tensor, timesteps: list[float], guidance: float = 4.0
    ):
        batch_size = img.shape[0]
        guidance_vec = torch.full((batch_size,), guidance, device=img.device, dtype=img.dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((batch_size,), t_curr, device=img.device, dtype=img.dtype)
            pred = self.flow_model(img, img_ids, txt, txt_ids, y=vec, timesteps=t_vec, guidance=guidance_vec)
            img = img + (t_prev - t_curr) * pred
        return img
    
    def forward_text_model(self, text: list[str]):
        tokens_out = self.t5_tokenizer(
            text,
            truncation=True,
            max_length=self.t5_tokenizer.model_max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = tokens_out.input_ids
        input_ids = input_ids.to(next(self.t5_text_model.parameters()).device)
        outputs = self.t5_text_model(input_ids, attention_mask=None, output_hidden_states=False)
        t5_emb = outputs['last_hidden_state']
        txt_ids = torch.zeros(t5_emb.size(0), t5_emb.size(1), 3, device=t5_emb.device)

        tokens_out = self.clip_tokenizer(
            text,
            truncation=True,
            max_length=self.clip_tokenizer.model_max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = tokens_out['input_ids']
        input_ids = input_ids.to(next(self.clip_text_model.parameters()).device)
        outputs = self.clip_text_model(input_ids, attention_mask=None, output_hidden_states=False)
        clip_emb = outputs['pooler_output']

        return t5_emb, txt_ids, clip_emb
    
    def get_schedule(
        self,
        num_steps: int,
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
    ) -> list[float]:
        # extra step for zero
        timesteps = torch.linspace(1, 0, num_steps + 1)

        # shifting the schedule to favor high timesteps for higher signal images
        if shift:
            x1 = 256
            x2 = 4096
            m = (max_shift - base_shift) / (x2 - x1)
            b = base_shift - m * x1
            mu = m * image_seq_len + b

            timesteps = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1) ** 1.0)
        
        return timesteps.tolist()


class PLBase(lightning.LightningModule):
    def __init__(
        self,
        model_config,
        loss_config = None,
        optimizer_config = None,
        sampler_config = None,
        ema_config = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.sampler_config = sampler_config

        self.model = FluxModel(model_config)
        if loss_config:
            self.loss_fn = utils.instantiate_from_config(loss_config)
        if ema_config:
            self.model_ema = FluxModel(model_config)
            self.model_ema.require_grad_(False)
            self.ema_helper = ema.EMAHelper(
                max_decay=ema_config.max_decay,
                use_ema_warmup=ema_config.warmup,
                inv_gamma=ema_config.inv_gamma,
                power=ema_config.power,
            )
    
    def predict_step(self, batch, batch_idx):
        batch_size = len(batch['prompt'])
        width, height = batch['width'][0], batch['height'][0]
        image = torch.randn(
            batch_size, 16, 2 * height // 16, 2 * width // 16,
            dtype=torch.bfloat16,
            generator=torch.Generator()
        ).cuda(self.local_rank)
        c, h, w = image.shape[1:]
        image = einops.rearrange(image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = einops.repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        img_ids = img_ids.cuda(self.local_rank)

        self.model.t5_text_model.to(self.local_rank)
        self.model.clip_text_model.to(self.local_rank)
        txt, txt_ids, vec = self.model.forward_text_model(batch['prompt'])
        self.model.t5_text_model.cpu()
        self.model.clip_text_model.cpu()
        torch.cuda.empty_cache()

        timesteps = self.model.get_schedule(
            self.sampler_config.num_steps, image.size(1),
            shift=self.sampler_config.shift_schedule
        )

        self.model.flow_model.cuda(self.local_rank)
        image = self.model.sample(image, img_ids, txt, txt_ids, vec, timesteps, self.sampler_config.guidance)
        self.model.flow_model.cpu()
        torch.cuda.empty_cache()

        image = image.float()
        image = einops.rearrange(image, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=height // 16, w=width // 16, ph=2, pw=2)
        self.model.ae.cuda(self.local_rank)
        image = self.model.ae.decode(image)
        self.model.ae.cpu()

        image = image.clamp(-1, 1)
        image = (image + 1) / 2
        log_image_dict = {'image': image}
        log_keys=['image']
        self.log_image(log_image_dict, log_keys, batch_idx, mode='predict')
    
    @torch.no_grad()
    def log_image(self, batch, batch_keys, batch_idx, mode):
        """
        Args:
            batch: dictionary, key: str -> value: tensor
        """
        dirname = os.path.join(self.trainer.default_root_dir, 'log_images', mode)
        os.makedirs(dirname, exist_ok=True)
        for key in batch_keys:
            image_t = batch[key].permute(0, 2, 3, 1).squeeze(-1)
            image_np = image_t.detach().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            for i in range(image_np.shape[0]):
                filename = f'gs-{self.global_step:04}_e-{self.current_epoch:04}_b-{batch_idx:04}-{i:02}_{key}.png'
                Image.fromarray(image_np[i]).save(os.path.join(dirname, filename))