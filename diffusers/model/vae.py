import einops
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, List
import xformers

from diffusers.model.discriminator import NLayerDiscriminator
from diffusers.loss.lpips import LPIPS


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        """
        Normalization-first Resnet Block.
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = F.silu(h)    # is this the same as original?
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:,:,None,None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h * w) # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class AttnBlockV2(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = einops.rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = einops.rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = einops.rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return einops.rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(x))


class MemoryEfficientAttnBlock(nn.Module):
    """
        Uses xformers efficient implementation,
        see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
        Note: this is a single-head self-attention operation
    """
    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: einops.rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = einops.rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x+out


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if mask is not None:
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None):
        b, c, h, w = x.shape
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        out = super().forward(x, context=context, mask=mask)
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=h, w=w, c=c)
        return x + out


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = einops.rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = einops.rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in ["vanilla", "flux", "vanilla-xformers", "memory-efficient-cross-attn", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "flux":
        return AttnBlockV2(in_channels)
    elif attn_type == "vanilla-xformers":
        return MemoryEfficientAttnBlock(in_channels)
    elif type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)
        # raise NotImplementedError()


class Encoder(nn.Module):
    def __init__(
        self, in_channels: int, ch: int, z_channels: int, ch_mult: List[int], num_res_blocks: int,
        attn_resolutions: List[int], resolution: int, dropout=0.0, resamp_with_conv=True, attn_type="vanilla"
    ):
        """
        Args:
            in_channels: input channels
            ch: base channels
            out_ch: not used
            ch_mult: channel multiplicator for each stage
            num_res_blocks: num of blocks in each stage
            attn_resolutions: use attention layer at this resolution
            dropout: dropout
            resamp_with_conv: bool, if true, downsample with strided conv, else with avgpool
            resolution: input resolution
            z_channels: latent space channels
            attn_type: attention layer type
        """
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self, ch: int, out_ch: int, z_channels: int, ch_mult: List[int], num_res_blocks: int,
        attn_resolutions: List[int], resolution, dropout=0.0, resamp_with_conv=True,
        give_pre_end=False, tanh_out=False, attn_type="vanilla"
    ):
        """
        Args:
            ch: base channels
            out_ch: out channels
            ch_mult: channel multiplicator for each stage
            num_res_blocks: num of blocks in each stage - 1 ???
            attn_resolutions: use attention layer at this resolution
            dropout: dropout
            resamp_with_conv: bool, if use a conv layer after upsampling
            in_channels: not used
            resolution: input resolution
            z_channels: latent space channels
            give_pre_end: bool, if true, return before the norm/act/conv out layer
            tanh_out: bool, if true, use a last tanh out layer
            attn_type: type of attention layer
        """
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class AutoEncoderKL(nn.Module):
    "Variational AutoEncoder."
    def __init__(
        self,
        in_channels,
        base_channels,
        out_channels,
        z_channels,
        ch_mult,
        num_res_blocks,
        attn_resolutions,
        dropout = 0.0,
        attn_type = 'vanilla'
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels, base_channels, z_channels, ch_mult, num_res_blocks,
            attn_resolutions, resolution=256, dropout=dropout, attn_type=attn_type
        )
        self.decoder = Decoder(
            base_channels, out_channels, z_channels, ch_mult, num_res_blocks,
            attn_resolutions, resolution=256, dropout=dropout, attn_type=attn_type
        )
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
    
    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec


class AutoEncoderWithDisc(nn.Module):
    "Variational Autoencoder with Discriminator"
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        self.lw_perceptual = model_config.loss.weight_perceptual
        self.lw_kl = model_config.loss.weight_kl
        self.lw_gan = model_config.loss.weight_gan
        self.gan_start_step = model_config.loss.gan_start_step
        self.ae = AutoEncoderKL(model_config.autoencoder)
        self.discriminator = NLayerDiscriminator(**model_config.discriminator).apply(weights_init)
        self.perceptual_loss = LPIPS().eval()
        self.logvar = nn.Parameter(torch.ones(size=()) * model_config.logvar_init)

    def forward(self, x, optimizer_idx, global_step, sample_posterior=True):
        # TODO make it more efficient by specifying run what parts
        recon, posterior = self.ae(x, sample_posterior)

        if optimizer_idx == 0:    # train generator
            # reconstruction loss
            loss_rec = torch.abs(x - recon)
            # perceptual loss
            if self.lw_perceptual > 0:
                loss_per = self.perceptual_loss(x, recon)
                loss_rec = loss_rec + self.lw_perceptual * loss_per
            # nll loss of Laplacian distribution P(X|Z)?
            loss_nll = loss_rec / torch.exp(self.logvar) + self.logvar
            # No weights are used
            loss_nll_weighted = loss_nll
            loss_nll = torch.sum(loss_nll) / loss_nll.size(0)
            loss_nll_weighted = torch.sum(loss_nll_weighted) / loss_nll_weighted.size(0)
            # prior loss
            loss_kl = posterior.kl()
            loss_kl = torch.sum(loss_kl) / loss_kl.size(0)

            loss = loss_nll_weighted + self.lw_kl * loss_kl
            log_dict = {
                'loss_nll': loss_nll.detach().mean(),
                'loss_rec': loss_rec.detach().mean(),
                'loss_kl': loss_kl.detach().mean(),
                'logvar': self.logvar.detach(),
            }

            # GAN loss
            if global_step >= self.gan_start_step:
                logits_fake = self.discriminator(recon)
                loss_g = -torch.mean(logits_fake)
                if self.training:
                    lw_gan_adap = self.calculate_adaptive_weight(loss_nll, loss_g)
                else:
                    lw_gan_adap = torch.tensor(0.0)

                loss += self.lw_gan * lw_gan_adap * loss_g
                log_dict.update({
                    'loss_g': loss_g.detach().mean(),
                    'd_weight': lw_gan_adap.detach(),
                })
            
            log_dict['loss_total'] = loss.clone().detach().mean()
            return loss, log_dict
        
        if optimizer_idx == 1:    # train discriminator
            logits_real = self.discriminator(x.detach())
            logits_fake = self.discriminator(recon.detach())
            loss_d = self.hinge_d_loss(logits_real, logits_fake)

            loss = loss_d

            log_dict = {
                'loss_d': loss_d.clone().detach().mean(),
                'logits_real': logits_real.detach().mean(),
                'logits_fake': logits_fake.detach().mean(),
            }
            return loss, log_dict
    
    def calculate_adaptive_weight(self, loss_nll, loss_g):
        grad_nll = torch.autograd.grad(loss_nll, self.ae.decoder.conv_out.weight, retain_graph=True)[0]
        grad_g = torch.autograd.grad(loss_g, self.ae.decoder.conv_out.weight, retain_graph=True)[0]
        d_weight = torch.norm(grad_nll) / (torch.norm(grad_g) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
    
    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def vanilla_d_loss(self, logits_real, logits_fake):
        d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
        return d_loss


class PLAutoEncoderWithDisc(pl.LightningModule):

    def __init__(
        self,
        model_config: OmegaConf,
        ckpt_path=None,
    ):
        super().__init__()
        self.config = model_config
        self.model = AutoEncoderWithDisc(model_config)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        self.automatic_optimization = False
    
    def init_from_ckpt(self, path):
        ckpt = torch.load(path, map_location='cpu')
        missing, unexpected = self.model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'Restored from checkpoint {path}.')
        print(f'Missing keys: {missing}.')
        print(f'Unexpected keys: {unexpected}.')
    
    def forward(self, x, sample_posterior=True):
        return self.model(x, sample_posterior)

    def training_step(self, batch, batch_idx):
        # Should we train the generator and discriminator at different steps?
        if self.global_step >= self.config.loss.gan_start_step \
            and self.global_step % 2 == 1:    # train discriminator
            optimizer_idx = 1
        else:    # train generator
            optimizer_idx = 0
        optimizer_g, optimizer_d = self.optimizers(use_pl_optimizer=True)
        loss, log_dict = self.model(batch['image'], optimizer_idx, self.global_step, sample_posterior=True)
        log_dict = {f'train/{k}': v for k, v in log_dict.items()}
        
        if optimizer_idx == 0:
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            self.manual_backward(loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

            self.log('aeloss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch['image'].size(0))
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=batch['image'].size(0))
        
        else:
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad()
            self.manual_backward(loss)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

            self.log('discloss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch['image'].size(0))
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False,batch_size=batch['image'].size(0))
    
    def validation_step(self, batch, batch_idx):
        loss_ae, log_dict_ae = self.model(batch['image'], 0, self.global_step, sample_posterior=True)
        loss_disc, log_dict_disc = self.model(batch['image'], 1, self.global_step, sample_posterior=True)
        log_dict_ae = {f'val/{k}': v for k, v in log_dict_ae.items()}
        log_dict_disc = {f'val/{k}': v for k, v in log_dict_disc.items()}
        self.log('val/loss_rec', log_dict_ae['val/loss_rec'], batch_size=batch['image'].size(0))
        self.log_dict(log_dict_ae, batch_size=batch['image'].size(0))
        self.log_dict(log_dict_disc, batch_size=batch['image'].size(0))
    
    def configure_optimizers(self,):
        # TODO logvar is not optimized?
        lr = self.config.learning_rate
        optimizer_g = torch.optim.Adam(
            self.model.ae.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        optimizer_d = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [optimizer_g, optimizer_d], []

    @torch.no_grad()
    def log_images(self, batch, **unused_kwargs):
        image = batch['image']
        rec, posterior = self.model.ae(image, sample_posterior=True)
        sample = self.model.ae.decode(torch.randn_like(posterior.sample()))
        return {
            'inputs': image,
            'reconstructions': rec,
            'samples': sample,
        }
