import copy
import itertools
import logging
import os
from PIL import Image
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torchmetrics
from timm.layers import DropPath, to_2tuple, trunc_normal_

from modules.multiscale_deform_attn import MSDeformAttn
from modules.position_embedding import SinusoidalEmbedding2D
from classifiers.model.swin_transformer import PatchEmbed, PatchMerging, SwinTransformerBlock
from detectors.model.deformable_detr import DeformableTransformerEncoderLayer, DeformableTransformerEncoder
from segmentors.loss.matching_based import SetClassSegmentLoss
import torch_utils
import utils


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x


class D2SwinTransformer(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        out_features=["res2", "res3", "res4", "res5"],
        frozen_stages=-1,
        use_checkpoint=False,
    ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.patch_size = (patch_size, patch_size)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()

        self._out_features = out_features

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        
        x = self.maybe_pad(x)
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic")
            x = x + absolute_pos_embed

        x = self.pos_drop(x)

        x = x.permute(0, 2, 3, 1)

        outs = {}
        for i in range(self.num_layers):
            x_out, x = self.layers[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                outs["res{}".format(i + 2)] = x_out.permute(0, 3, 1, 2).contiguous()

        outs = {k: v for k, v in outs.items() if k in self._out_features}
        return outs
    
    def maybe_pad(self, x):    # pad image to be dividable by patch_size
        H, W = x.size(2), x.size(3)
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()
    
    def output_shape(self):
        return {
            name: dict(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        # if not torch.jit.is_scripting():
        #     # Dynamo doesn't support context managers yet
        #     is_dynamo_compiling = check_if_dynamo_compiling()
        #     if not is_dynamo_compiling:
        #         with warnings.catch_warnings(record=True):
        #             if x.numel() == 0 and self.training:
        #                 # https://github.com/pytorch/pytorch/issues/12013
        #                 assert not isinstance(
        #                     self.norm, torch.nn.SyncBatchNorm
        #                 ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation,
            num_feature_levels, nhead, enc_n_points, img2col_step=128
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = [src.flatten(2).transpose(1, 2) for src in srcs]
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = [mask.flatten(1) for mask in masks]
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = []
        for lvl, pos_embed in enumerate(pos_embeds):
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = [src.shape[2:] for src in srcs]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        bs = memory.shape[0]
        split_size_or_sections = []
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_size_or_sections.append(level_start_index[i + 1] - level_start_index[i])
            else:
                split_size_or_sections.append(memory.shape[1] - level_start_index[i])
        outs = torch.split(memory, split_size_or_sections, dim=1)
        outs = [out.transpose(1, 2).view(bs, -1, h, w) for out, (h, w) in zip(outs, spatial_shapes)]

        return outs


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(
        self,
        config,
        input_shape: Dict[str, Dict],
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        conv_dim = config.sem_seg_head.convs_dim
        mask_dim = config.sem_seg_head.mask_dim
        norm = config.sem_seg_head.norm
        transformer_dropout = config.mask_former.dropout
        transformer_nheads = config.mask_former.nheads
        transformer_dim_feedforward = 1024    # use 1024 for deformable transformer encoder
        transformer_enc_layers = config.sem_seg_head.transformer_enc_layers
        self.transformer_in_features = config.sem_seg_head.deformable_transformer_encoder_in_features
        common_stride = config.sem_seg_head.common_stride
        self.out_ms_feature_levels = 3  # always use 3 scales

        # this is the input shape of pixel decoder
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v["stride"] for k, v in input_shape]
        self.feature_channels = [v["channels"] for k, v in input_shape]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = [(k, v) for k, v in input_shape if k in self.transformer_in_features]
        transformer_in_channels = [v["channels"] for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v["stride"] for k, v in transformer_input_shape]  # to decide extra FPN layers

        if len(self.transformer_in_features) > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=len(self.transformer_in_features),
        )
        self.pe_layer = SinusoidalEmbedding2D(conv_dim // 2, normalize=True)

        # use 1x1 conv instead
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_uniform_(self.mask_features.weight, a=1)
        nn.init.constant_(self.mask_features.bias, 0)

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(common_stride))

        lateral_convs = []
        output_convs = []
        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = nn.GroupNorm(32, conv_dim)
            output_norm = nn.GroupNorm(32, conv_dim)
            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias,
                norm=output_norm, activation=F.relu,
            )

            nn.init.kaiming_uniform_(lateral_conv.weight, a=1)
            if lateral_conv.bias:
                nn.init.constant_(lateral_conv.bias, 0)
            nn.init.kaiming_uniform_(output_conv.weight, a=1)
            if output_conv.bias:
                nn.init.constant_(output_conv.bias, 0)
            
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
    
    @torch.amp.autocast("cuda", enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        outs = self.transformer(srcs, pos)
        
        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            cur_fpn = self.lateral_convs[idx](x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(outs[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = self.output_convs[idx](y)
            outs.append(y)

        multi_scale_features = outs[:self.out_ms_feature_levels]

        return self.mask_features(outs[-1]), outs[0], multi_scale_features


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[torch.Tensor] = None,
                     tgt_key_padding_mask: Optional[torch.Tensor] = None,
                     query_pos: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[torch.Tensor] = None,
                    tgt_key_padding_mask: Optional[torch.Tensor] = None,
                    query_pos: Optional[torch.Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory,
                memory_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        shortcut = tgt
        if self.normalize_before:
            tgt = self.norm(tgt)
        tgt = self.multihead_attn(
            query=tgt + query_pos, key=memory + pos, value=memory,
            attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = shortcut + self.dropout(tgt)
        if not self.normalize_before:
            tgt = self.norm(tgt)
        return tgt


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(self, config, in_channels, mask_classification):
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        hidden_dim = config.mask_former.hidden_dim

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = SinusoidalEmbedding2D(N_steps, normalize=True)

        num_classes = config.sem_seg_head.num_classes
        self.num_queries = config.mask_former.num_object_queries
        # transformer parameters
        self.num_heads = config.mask_former.nheads
        dim_feedforward = config.mask_former.dim_feedforward
        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert config.mask_former.dec_layers >= 1
        self.num_layers = config.mask_former.dec_layers - 1
        pre_norm = config.mask_former.pre_norm
        enforce_input_project = config.mask_former.enforce_input_proj
        mask_dim = config.sem_seg_head.mask_dim

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())
        
        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
    
    def forward(self, x, mask_features):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = [x[i].shape[-2:] for i in range(len(x))]

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(x[i]).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            # flatten NxCxHW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        pred_logits = []
        pred_masks = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        pred_logits.append(outputs_class)
        pred_masks.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            pred_logits.append(outputs_class)
            pred_masks.append(outputs_mask)

        assert len(pred_logits) == self.num_layers + 1

        return {
            'pred_logits': pred_logits[-1],
            'pred_masks': pred_masks[-1],
            'aux_outputs': [{"pred_logits": a, "pred_masks": b} for a, b in zip(pred_logits[:-1], pred_masks[:-1])]
        }
    
    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)    # shape (batch, query, class)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask


class MaskFormerHead(nn.Module):
    def __init__(
        self,
        config,
        input_shape: Dict[str, Dict],
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = {k: v for k, v in input_shape.items() if k in config.sem_seg_head.in_features}
        input_shape = sorted(input_shape.items(), key=lambda x: x[1]["stride"])
        self.in_features = [k for k, v in input_shape]

        # self.ignore_value = config.sem_seg_head.ignore_value
        # self.common_stride = 4
        # self.loss_weight = config.seg_seg_head.loss_weight
        self.transformer_in_feature = config.mask_former.transformer_in_feature
        self.num_classes = config.sem_seg_head.num_classes
        self.pixel_decoder = MSDeformAttnPixelDecoder(config, input_shape)

        if config.mask_former.transformer_in_feature == "transformer_encoder":
            predictor_in_channels = config.sem_seg_head.convs_dim
        elif config.mask_former.transformer_in_feature == "pixel_embedding":
            predictor_in_channels = config.sem_seg_head.mask_dim
        elif config.mask_former.transformer_in_feature == "multi_scale_pixel_decoder":  # for maskformer2
            predictor_in_channels = config.sem_seg_head.convs_dim
        else:
            predictor_in_channels = input_shape[config.mask_former.transformer_in_feature]["channels"]
        self.predictor = MultiScaleMaskedTransformerDecoder(config, predictor_in_channels, mask_classification=True)
    
    def forward(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result


class MaskFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = D2SwinTransformer(
            pretrain_img_size = config.swin.pretrain_img_size,
            patch_size = config.swin.patch_size,
            in_chans = 3,
            embed_dim = config.swin.embed_dim,
            depths = config.swin.depths,
            num_heads = config.swin.num_heads,
            window_size = config.swin.window_size,
            mlp_ratio = config.swin.mlp_ratio,
            qkv_bias = config.swin.qkv_bias,
            qk_scale = config.swin.qk_scale,
            drop_rate = config.swin.drop_rate,
            attn_drop_rate = config.swin.attn_drop_rate,
            drop_path_rate = config.swin.drop_path_rate,
            norm_layer = nn.LayerNorm,
            ape = config.swin.ape,
            patch_norm = config.swin.patch_norm,
            out_features = config.swin.out_features,
            use_checkpoint = config.swin.use_checkpoint
        )
        self.sem_seg_head = MaskFormerHead(config, self.backbone.output_shape())
        
        self.num_queries = config.mask_former.num_object_queries
        self.overlap_threshold = config.mask_former.test.overlap_threshold
        self.object_mask_threshold = config.mask_former.test.object_mask_threshold
        
        if config.mask_former.size_divisibility < 0:
            self.size_divisibility = self.backbone.size_divisibility
        else:
            self.size_divisibility = config.mask_former.size_divisibility
        self.sem_seg_postprocess_before_inference = (
            config.mask_former.test.sem_seg_postprocessing_before_inference
            or config.mask_former.test.panoptic_on
            or config.mask_former.test.instance_on
        )
        self.register_buffer("pixel_mean", torch.Tensor(config.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(config.pixel_std).view(-1, 1, 1), False)
        self.semantic_on = config.mask_former.test.semantic_on
        self.instance_on = config.mask_former.test.instance_on
        self.panoptic_on = config.mask_former.test.panoptic_on
        self.test_topk_per_image = 10
    
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, images):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images)
        outputs = self.sem_seg_head(features)
        return outputs
    
    def inference(self, batch, contiguous_id_to_dataset_id, dataset_thing_ids):
        images = batch['image']
        outputs = self(batch)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.shape[-2], images.shape[-1]),
            mode="bilinear",
            align_corners=False
        )
        del outputs

        processed_results = list()
        image_sizes = [(im.shape[-2], im.shape[-1]) for im in images]
        for mask_cls_result, mask_pred_result, height, width, image_size in zip(
            mask_cls_results, mask_pred_results, batch["height"], batch["width"], image_sizes
        ):
            # mask_cls_result: shape (query, class)
            # mask_pred_result: shape (query, h, w)
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = sem_seg_postprocess(mask_pred_result, image_size, height, width)
                mask_cls_result = mask_cls_result.to(mask_pred_result)
            
            if self.semantic_on:
                r = self.semantic_inference(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = sem_seg_postprocess(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r    # shape (class, h, w)
            
            if self.panoptic_on:
                panoptic_r = self.panoptic_inference(mask_cls_result, mask_pred_result, contiguous_id_to_dataset_id, dataset_thing_ids)
                processed_results[-1]["panoptic_seg"] = panoptic_r
            
            if self.instance_on:
                instance_r = self.instance_inference(mask_cls_result, mask_pred_result, dataset_thing_ids)
                processed_results[-1]["instances"] = instance_r
        
        return processed_results

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, contiguous_id_to_dataset_id, dataset_thing_ids):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks    # shape (query, h, w)

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # we didn't detect any mask
            return panoptic_seg, segments_info
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):    # for each query
                pred_class = cur_classes[k].item()
                pred_dataset_class = contiguous_id_to_dataset_id[int(pred_class)]
                isthing = pred_dataset_class in dataset_thing_ids
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    if not isthing:    # merge stuff regions
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append({
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": pred_dataset_class
                    })
            
            return panoptic_seg, segments_info
    
    def instance_inference(self, mask_cls, mask_pred, dataset_thing_ids):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]    # shape (topk)

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]    # shape (topk, h, w)

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in dataset_thing_ids
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        
        result = dict()
        result['pred_masks'] = (mask_pred > 0).float()
        result['pred_boxes'] = torch.zeros(mask_pred.size(0), 4)
        # average mask prob
        mask_scores_per_image = (mask_pred.sigmoid() * result['pred_masks']).sum([1, 2]) / (result['pred_masks'].sum([1, 2]) + 1e-6)
        result['scores'] = scores_per_image * mask_scores_per_image
        result['pred_classes'] = labels_per_image
        return result


class Trainer():
    def __init__(self, model_config, loss_config, optimizer_config, device):
        _model = MaskFormer(model_config)
        _model.cuda()
        self.model = DistributedDataParallel(_model, device_ids=[device])

        self.loss_fn = SetClassSegmentLoss(
            model_config.sem_seg_head.num_classes, loss_config.weight_empty_class,
            loss_config.weight_class, loss_config.weight_bce, loss_config.weight_dice,
            loss_config.num_points, loss_config.oversample_ratio, loss_config.importance_sample_ratio
        ).cuda()

        # prepare optimizers
        defaults = {"lr": optimizer_config.base_lr, "weight_decay": optimizer_config.weight_decay}
        groups = list()
        memo = set()
        for module_name, module in self.model.named_modules():
            for module_param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if param in memo:
                    raise ValueError("duplicate parameters")
                memo.add(param)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * optimizer_config.backbone_multiplier
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                    hyperparams["weight_decay"] = optimizer_config.weight_decay_norm
                if isinstance(module, nn.Embedding):
                    hyperparams["weight_decay"] = optimizer_config.weight_decay_embed
                groups.append({"params": [param], **hyperparams})

        self.optimizer = utils.get_obj_from_str(optimizer_config.optimizer)(
            groups, optimizer_config.base_lr
        )
        optimizer_config.lr_scheduler_params.T_max = optimizer_config.num_training_steps
        self.lr_scheduler = utils.get_obj_from_str(optimizer_config.lr_scheduler)(
            self.optimizer, **optimizer_config.lr_scheduler_params
        )
        if optimizer_config.warmup:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=optimizer_config.warmup
            )
            self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, [warmup_scheduler, self.lr_scheduler], milestones=[optimizer_config.warmup]
            )
        
        # prepare metrics
        self.loss_meters = {
            "loss_class": torch_utils.RunningStatistic(device),
            "loss_bce": torch_utils.RunningStatistic(device),
            "loss_dice": torch_utils.RunningStatistic(device)
        }
        if loss_config.deep_supervision:
            for i in range(model_config.mask_former.dec_layers - 1):
                self.loss_meters[f"loss_class_{i}"] = torch_utils.RunningStatistic(device)
                self.loss_meters[f"loss_bce_{i}"] = torch_utils.RunningStatistic(device)
                self.loss_meters[f"loss_dice_{i}"] = torch_utils.RunningStatistic(device)
        
        self.metric_pq = torchmetrics.detection.PanopticQuality()
    
    def on_train_epoch_start(self):
        self.model.train()
        for key in self.loss_meters:
            self.loss_meters[key].reset()
    
    def train_step(self, batch, batch_idx, global_step):
        batch_size = len(batch)
        images = torch.stack([x["image"] for x in batch]).cuda()
        outputs = self.model(images)

        gt_classes = [x["classes"].cuda() for x in batch]
        gt_masks = [x["masks"].cuda() for x in batch]
        loss_dict = self.loss_fn(
            outputs["pred_logits"], outputs["pred_masks"],
            gt_classes, gt_masks, outputs["aux_outputs"]
        )
        loss = sum(loss_dict.values())

        loss.backward()
        all_params = itertools.chain(*[x["params"] for x in self.optimizer.param_groups])
        nn.utils.clip_grad_norm_(all_params, max_norm=0.01)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()

        for key, value in loss_dict.items():
            self.loss_meters[key].update(value.detach(), batch_size)
        
        return outputs
        
    def on_train_epoch_end(self, epoch):
        return dict()
    
    def on_val_epoch_start(self):
        self.model.eval()
        self.metric_pq.reset()
    
    def val_step(self, batch, batch_idx):
        processed_results = self.model.inference(batch)
        for res in processed_results:
            panoptic_id_mask, panoptic_sem_mask, segments_info = res["panoptic_seg"]
            panoptic_mask = torch.stack((panoptic_sem_mask, panoptic_id_mask), dim=-1)
            self.metric_pq.update(panoptic_mask.unsqueeze(0), batch["panoptic_mask"])
    
    def on_val_epoch_end(self):
        logdict = dict()
        quality = self.metric_pq.compute()
        logdict[f"val/pq"] = quality[0].item()
        logdict[f"val/sq"] = quality[1].item()
        logdict[f"val/rq"] = quality[2].item()
        return logdict
    
    def log_step(self, batch, output, logdir, global_step, epoch, batch_idx):
        logdict = dict()
        for key in self.loss_meters.keys():
            val = self.loss_meters[key].compute()
            logdict[key] = val.item()
            self.loss_meters[key].reset()
        if dist.get_rank() == 0:
            self.log_image(batch, output, logdir, global_step, epoch, batch_idx)
        return logdict

    def log_image(self, batch, output, logdir, global_step, epoch, batch_idx):
        from segmentors.data.coco import COCO_CATEGORIES
        dirname = os.path.join(logdir, "log_images")
        os.makedirs(dirname, exist_ok=True)

        images = [x["image"] for x in batch]
        gt_classes = [x["classes"] for x in batch]
        gt_masks = [x["masks"] for x in batch]
        pred_logits = output["pred_logits"][:, :, :-1]
        pred_scores, pred_classes = pred_logits.max(dim=-1)
        pred_masks = output["pred_masks"]
        pred_scores, topk_index = pred_scores.topk(10, dim=1)    # shape (batch, k)
        pred_classes = torch.gather(pred_classes, 1, topk_index)
        batch_index = torch.arange(topk_index.size(0)).unsqueeze(1)
        pred_masks = pred_masks[batch_index, topk_index]
        pred_masks = F.interpolate(pred_masks, scale_factor=4, model="bilinear")
        pred_classes = pred_classes.cpu().numpy()
        pred_masks = pred_masks.detach().cpu().numpy()

        for i, (image, gt_class, gt_mask, pred_class, pred_mask) in enumerate(
            zip(images, gt_classes, gt_masks, pred_classes, pred_masks)):
            image = image.permute(1, 2, 0).cpu().numpy()
            mask = np.array(image.shape, dtype="uint8")
            for c, m in zip(gt_class, gt_mask.cpu().numpy()):
                color = COCO_CATEGORIES[c]["color"]
                mask[m > 0] = color
            Image.fromarray(image).save(
                os.path.join(dirname, f"gs{global_step}-e{epoch}-{i}_image.png")
            )
            Image.fromarray(mask).save(
                os.path.join(dirname, f"gs{global_step}-e{epoch}-{i}_mask.png")
            )
            mask = np.zeros(image.shape, dtype="uint8")
            for c, m in zip(pred_class, pred_mask):
                color = COCO_CATEGORIES[c]["color"]
                mask[m > 0] = color
            Image.fromarray(mask).save(
                os.path.join(dirname, f"gs{global_step}-e{epoch}-{i}_pred.png")
            )