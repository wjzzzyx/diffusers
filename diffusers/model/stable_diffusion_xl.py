import einops
import lightning
import logging
import math
import numpy as np
from omegaconf import ListConfig
from packaging import version
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Any, Optional, Union, List, Tuple, Dict

logger = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    logger.warn(
        f"No SDP backend available, likely because you are running in pytorch "
        f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
        f"You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    logger.warn("no module 'xformers'. Processing without...")

from diffusers.model.stable_diffusion_stabilityai import (
    TimestepBlock, conv_nd, Upsample, Downsample, zero_module, AttentionBlock, timestep_embedding,
    TimestepEmbedSequential, FeedForward
)
import torch_utils
import utils


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = einops.repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = einops.repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = einops.rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        logger.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a "
            f"dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = einops.repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = einops.repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

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
        if version.parse(xformers.__version__) >= version.parse("0.0.21"):
            # NOTE: workaround for
            # https://github.com/facebookresearch/xformers/issues/845
            max_bs = 32768
            N = q.shape[0]
            n_batches = math.ceil(N / max_bs)
            out = list()
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out.append(
                    xformers.ops.memory_efficient_attention(
                        q[batch],
                        k[batch],
                        v[batch],
                        attn_bias=None,
                        op=self.attention_op,
                    )
                )
            out = torch.cat(out, 0)
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

        # TODO: Use this directly in the attention operation, as a bias
        if mask is not None:
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            logger.warn(
                f"Attention mode '{attn_mode}' is not available. Falling "
                f"back to native attention. This is not a problem in "
                f"Pytorch >= 2.0. FYI, you are running with PyTorch "
                f"version {torch.__version__}."
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            logger.warn(
                "We do not support vanilla attention anymore, as it is too "
                "expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                logger.info("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            logger.debug(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        if self.checkpoint:
            # inputs = {"x": x, "context": context}
            return checkpoint(self._forward, x, context)
            # return checkpoint(self._forward, inputs, self.parameters(), self.checkpoint)
        else:
            return self._forward(**kwargs)

    def _forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
    ):
        super().__init__()
        logger.debug(
            f"constructing {self.__class__.__name__} of depth {depth} w/ "
            f"{in_channels} channels and {n_heads} heads."
        )

        if context_dim is not None and not isinstance(context_dim, list):
            context_dim = [context_dim]
        if context_dim is not None and isinstance(context_dim, list):
            if depth != len(context_dim):
                logger.warn(
                    f"{self.__class__.__name__}: Found context dims "
                    f"{context_dim} of depth {len(context_dim)}, which does not "
                    f"match the specified 'depth' of {depth}. Setting context_dim "
                    f"to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv_in_skip: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        skip_t_emb: bool = False,
        exchange_temb_dims: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv_in_skip = use_conv_in_skip
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.skip_t_emb = skip_t_emb
        self.exchange_temb_dims = exchange_temb_dims

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if skip_t_emb:
            self.emb_layers = None
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv_in_skip:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        if self.skip_t_emb:
            emb_out = torch.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = einops.rearrange(emb_out, 'b t c ... -> b c t ...')
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(AttentionBlock):
    def forward(self, x):
        return checkpoint(self._forward, x)


class Timestep(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t, self.dim)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: Union[List, Tuple] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[Union[int, str]] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        disable_self_attentions: Optional[List[bool]] = None,
        num_attention_blocks: Optional[List[int]] = None,
        disable_middle_self_attn: bool = False,
        disable_middle_transformer: bool = False,
        use_linear_in_transformer: bool = False,
        spatial_transformer_attn_type: str = "softmax",
        adm_in_channels: Optional[int] = None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = transformer_depth[-1]

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        nn.Linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        nn.Linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        nn.Linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        nn.Linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if context_dim is not None and disable_self_attentions is not None:
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        num_attention_blocks is None
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                attn_type=spatial_transformer_attn_type,
                use_checkpoint=use_checkpoint,
            )
            if not disable_middle_transformer
            else nn.Identity(),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if disable_self_attentions is not None:
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        num_attention_blocks is None
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)

        return self.out(h)


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: Union[List, ListConfig]):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = utils.instantiate_from_config(embconfig)
            # assert isinstance(
            #     embedder, AbstractEmbModel
            # ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                # embedder.train = disabled_train
                embedder.requires_grad_(False)
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {torch_utils.count_params(embedder)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(
                    f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                )

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    def possibly_get_ucg_val(self, embedder, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None
    ) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate)
                                * torch.ones(emb.shape[0], device=emb.device)
                            ),
                            emb,
                        )
                        * emb
                    )
                if (
                    hasattr(embedder, "input_key")
                    and embedder.input_key in force_zero_embeddings
                ):
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat(
                        (output[out_key], emb), self.KEY2CATDIM[out_key]
                    )
                else:
                    output[out_key] = emb
        return output

    def get_unconditional_conditioning(
        self,
        batch_c: Dict,
        batch_uc: Optional[Dict] = None,
        force_uc_zero_embeddings: Optional[List[str]] = None,
        force_cond_zero_embeddings: Optional[List[str]] = None,
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c, force_cond_zero_embeddings)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c):
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = (
            self.n_classes - 1
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc


class ConcatTimestepEmbedderND(nn.Module):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = einops.rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = einops.rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb


class StableDiffusionXL(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.prediction_type = model_config.prediction_type
        self.scale_factor = model_config.scale_factor

        self.diffusion_model = utils.instantiate_from_config(model_config.network_config)
        self.conditioner = utils.instantiate_from_config(model_config.conditioner_config)
        self.first_stage_model = utils.instantiate_from_config(model_config.first_stage_config)
    
        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)
    
    def forward_diffusion_model(self, xt, t, cond_prompt):
        output = self.diffusion_model(xt, t, context=cond_prompt)
        return output
    
    def encode_first_stage(self, x):
        encoder_posterior = self.first_stage_model.encode(x)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)
    

class PLBase(lightning.LightningModule):
    def __init__(self, model_config, sampler_config, loss_config, optimizer_config, ema_config):
        self.optimizer_config = optimizer_config
        self.ema_config = ema_config

        self.model = StableDiffusionXL(model_config)
        self.sampler = utils.instantiate_from_config(sampler_config)
        self.loss_fn = utils.instantiate_from_config(loss_config)
        if self.ema_config:
            self.model_ema = StableDiffusionXL(model_config)
            self.model_ema.requires_grad_(False)
            self.ema_helper = ema.EMAHelper(
                max_decay=ema_config.max_decay,
                use_ema_warmup=ema_config.warmup,
                inv_gamma=ema_config.inv_gamma,
                power=ema_config.power,
            )

    def training_step(self, batch, batch_idx):
        x = self.encode_first_stage(batch['image'])
        loss, log_dict = self.loss_fn(self.model, self.denoiser, x, batch)
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def on_train_batch_end(self, output, batch, batch_idx):
        if self.ema_config:
            self.ema_helper.step(self.model, self.model_ema)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.optimizer_config.learning_rate,
            betas=(self.optimizer_config.adam_beta1, self.optimizer_config.adam_beta2),
            weight_decay=self.optimizer_config.weight_decay,
        )
        lr_scheduler = WarmupLinearScheduler()
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}
        }