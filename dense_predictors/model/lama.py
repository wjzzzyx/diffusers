from abc import abstractmethod, ABC
from collections import namedtuple, OrderedDict
import colorsys
import cv2
from kornia.geometry.transform import rotate
import lightning
import logging
import math
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import pandas as pd
from PIL import Image
import scipy.ndimage
import skimage.color
import skimage.segmentation
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torchvision
from typing import List, Tuple, Dict, Optional
import warnings

# from classifiers.model import mobilenet

LOGGER = logging.getLogger(__name__)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class LearnableSpatialTransformWrapper(nn.Module):
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans)
            return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

    def transform(self, x):
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)

        y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
        return y


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class FFCResNetGenerator(nn.Module):
    def __init__(
        self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
        padding_type='reflect', activation_layer=nn.ReLU,
        up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
        init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
        spatial_transform_layers=None, spatial_transform_kwargs={},
        add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}
    ):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(
                feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                norm_layer=norm_layer, **resnet_conv_kwargs
            )
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]


class BaseAdversarialLoss:
    def pre_generator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           generator: nn.Module, discriminator: nn.Module):
        """
        Prepare for generator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module):
        """
        Prepare for discriminator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def generator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                       discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate generator loss
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch
        :return: total generator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def discriminator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate discriminator loss and call .backward() on it
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch
        :return: total discriminator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def interpolate_mask(self, mask, shape):
        assert mask is not None
        assert self.allow_scale_mask or shape == mask.shape[-2:]
        if shape != mask.shape[-2:] and self.allow_scale_mask:
            if self.mask_scale_mode == 'maxpool':
                mask = F.adaptive_max_pool2d(mask, shape)
            else:
                mask = F.interpolate(mask, size=shape, mode=self.mask_scale_mode)
        return mask

def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty


class NonSaturatingWithR1(BaseAdversarialLoss):
    def __init__(self, gp_coef=5, weight=1, mask_as_fake_target=False, allow_scale_mask=False,
                 mask_scale_mode='nearest', extra_mask_weight_for_gen=0,
                 use_unmasked_for_gen=True, use_unmasked_for_discr=True):
        self.gp_coef = gp_coef
        self.weight = weight
        # use for discr => use for gen;
        # otherwise we teach only the discr to pay attention to very small difference
        assert use_unmasked_for_gen or (not use_unmasked_for_discr)
        # mask as target => use unmasked for discr:
        # if we don't care about unmasked regions at all
        # then it doesn't matter if the value of mask_as_fake_target is true or false
        assert use_unmasked_for_discr or (not mask_as_fake_target)
        self.use_unmasked_for_gen = use_unmasked_for_gen
        self.use_unmasked_for_discr = use_unmasked_for_discr
        self.mask_as_fake_target = mask_as_fake_target
        self.allow_scale_mask = allow_scale_mask
        self.mask_scale_mode = mask_scale_mode
        self.extra_mask_weight_for_gen = extra_mask_weight_for_gen

    def generator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                       discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                       mask=None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fake_loss = F.softplus(-discr_fake_pred)
        if (self.mask_as_fake_target and self.extra_mask_weight_for_gen > 0) or \
                not self.use_unmasked_for_gen:  # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            if not self.use_unmasked_for_gen:
                fake_loss = fake_loss * mask
            else:
                pixel_weights = 1 + mask * self.extra_mask_weight_for_gen
                fake_loss = fake_loss * pixel_weights

        return fake_loss.mean() * self.weight

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module):
        real_batch.requires_grad = True

    def discriminator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                           mask=None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        real_loss = F.softplus(-discr_real_pred)
        grad_penalty = make_r1_gp(discr_real_pred, real_batch) * self.gp_coef
        fake_loss = F.softplus(discr_fake_pred)

        if not self.use_unmasked_for_discr or self.mask_as_fake_target:
            # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            # use_unmasked_for_discr=False only makes sense for fakes;
            # for reals there is no difference beetween two regions
            fake_loss = fake_loss * mask
            if self.mask_as_fake_target:
                fake_loss = fake_loss + (1 - mask) * F.softplus(-discr_fake_pred)

        sum_discr_loss = real_loss + grad_penalty + fake_loss
        metrics = {
            'discr_real_out': discr_real_pred.mean().item(),
            'discr_fake_out': discr_fake_pred.mean().item(),
            'discr_real_gp': grad_penalty
        }
        return sum_discr_loss.mean(), metrics


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        if return_feature_maps:
            return conv_out
        return [x]


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']))
    return model


# Resnet Dilated
class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super().__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]

# Model Builder
class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.MobileNetV2(pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            # orig_resnet = resnet18(pretrained=pretrained)
            # net_encoder = Resnet(orig_resnet)
            net_encoder = resnet18(pretrained=pretrained)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet18(pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet50(pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            # orig_resnet = resnet50(pretrained=pretrained)
            # net_encoder = Resnet(orig_resnet)
            net_encoder = resnet50(pretrained=pretrained)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False, drop_last_conv=False):
        arch = arch.lower()
        if arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                drop_last_conv=drop_last_conv)
        elif arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                drop_last_conv=drop_last_conv)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

    @staticmethod
    def get_decoder(weights_path, arch_encoder, arch_decoder, fc_dim, drop_last_conv, *arts, **kwargs):
        path = os.path.join(weights_path, 'ade20k', f'ade20k-{arch_encoder}-{arch_decoder}/decoder_epoch_20.pth')
        return ModelBuilder.build_decoder(arch=arch_decoder, fc_dim=fc_dim, weights=path, use_softmax=True, drop_last_conv=drop_last_conv)

    @staticmethod
    def get_encoder(weights_path, arch_encoder, arch_decoder, fc_dim, segmentation,
                    *arts, **kwargs):
        if segmentation:
            path = os.path.join(weights_path, 'ade20k', f'ade20k-{arch_encoder}-{arch_decoder}/encoder_epoch_20.pth')
        else:
            path = ''
        return ModelBuilder.build_encoder(arch=arch_encoder, fc_dim=fc_dim, weights=path)


class ResNetPL(nn.Module):
    IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
    IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

    def __init__(self, weight=1,
                 weights_path=None, arch_encoder='resnet50dilated', segmentation=True):
        super().__init__()
        # self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
        #                                      arch_encoder=arch_encoder,
        #                                      arch_decoder='ppm_deepsup',
        #                                      fc_dim=2048,
        #                                      segmentation=segmentation)
        self.impl = ModelBuilder.build_encoder(
            arch=arch_encoder, fc_dim=2048, weights=weights_path
        )
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - self.IMAGENET_MEAN.to(pred)) / self.IMAGENET_STD.to(pred)
        target = (target - self.IMAGENET_MEAN.to(target)) / self.IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        result = torch.stack([F.mse_loss(cur_pred, cur_target)
                              for cur_pred, cur_target
                              in zip(pred_feats, target_feats)]).sum()
        return result


def ssim_fid100_f1(metrics, fid_scale=100):
    ssim = metrics[('ssim', 'total')]['mean']
    fid = metrics[('fid', 'total')]['mean']
    fid_rel = max(0, fid_scale - fid) / fid_scale
    f1 = 2 * ssim * fid_rel / (ssim + fid_rel + 1e-3)
    return f1


def lpips_fid100_f1(metrics, fid_scale=100):
    neg_lpips = 1 - metrics[('lpips', 'total')]['mean']  # invert, so bigger is better
    fid = metrics[('fid', 'total')]['mean']
    fid_rel = max(0, fid_scale - fid) / fid_scale
    f1 = 2 * neg_lpips * fid_rel / (neg_lpips + fid_rel + 1e-3)
    return f1


class EvaluatorScore(nn.Module):
    @abstractmethod
    def forward(self, pred_batch, target_batch, mask):
        pass

    @abstractmethod
    def get_value(self, groups=None, states=None):
        pass

    @abstractmethod
    def reset(self):
        pass


def get_groupings(groups):
    """
    :param groups: group numbers for respective elements
    :return: dict of kind {group_idx: indices of the corresponding group elements}
    """
    label_groups, count_groups = np.unique(groups, return_counts=True)

    indices = np.argsort(groups)

    grouping = dict()
    cur_start = 0
    for label, count in zip(label_groups, count_groups):
        cur_end = cur_start + count
        cur_indices = indices[cur_start:cur_end]
        grouping[label] = cur_indices
        cur_start = cur_end
    return grouping


class PairwiseScore(EvaluatorScore, ABC):
    def __init__(self):
        super().__init__()
        self.individual_values = None

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        individual_values = torch.cat(states, dim=-1).reshape(-1).cpu().numpy() if states is not None \
            else self.individual_values

        total_results = {
            'mean': individual_values.mean(),
            'std': individual_values.std()
        }

        if groups is None:
            return total_results, None

        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_scores = individual_values[index]
            group_results[label] = {
                'mean': group_scores.mean(),
                'std': group_scores.std()
            }
        return total_results, group_results

    def reset(self):
        self.individual_values = []


class SSIM(torch.nn.Module):
    """SSIM. Modified from:
    https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
    """

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer('window', self._create_window(window_size, self.channel))

    def forward(self, img1, img2):
        assert len(img1.shape) == 4

        channel = img1.size()[1]

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            # window = window.to(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - (window_size // 2)) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=(window_size // 2), groups=channel)
        mu2 = F.conv2d(img2, window, padding=(window_size // 2), groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=(window_size // 2), groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=(window_size // 2), groups=channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=(window_size // 2), groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()

        return ssim_map.mean(1).mean(1).mean(1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        return


class SSIMScore(PairwiseScore):
    def __init__(self, window_size=11):
        super().__init__()
        self.score = SSIM(window_size=window_size, size_average=False).eval()
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        batch_values = self.score(pred_batch, target_batch)
        self.individual_values = np.hstack([
            self.individual_values, batch_values.detach().cpu().numpy()
        ])
        return batch_values


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def name(self):
        return 'BaseModel'

    def initialize(self, use_gpu=True):
        self.use_gpu = use_gpu

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s' % save_path)
        network.load_state_dict(torch.load(save_path, map_location='cpu'))

    def update_learning_rate():
        pass

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'), flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'), [flag, ], fmt='%i')


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def upsample(in_tens, out_H=64):  # assumes scale factor is same for H and W
    in_H = in_tens.shape[2]
    scale_factor = 1. * out_H / in_H

    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False,
                 version='0.1', lpips=True):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()

        if (self.pnet_type in ['vgg', 'vgg16']):
            net_type = vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif (self.pnet_type == 'alex'):
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif (self.pnet_type == 'squeeze'):
            net_type = squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if (lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if (self.pnet_type == 'squeeze'):  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (
        in0, in1)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if (self.lpips):
            if (self.spatial):
                res = [upsample(self.lins[kk].model(diffs[kk]), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if (self.spatial):
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        if (retPerLayer):
            return (val, res)
        else:
            return val


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


def l2(p0, p1, range=255.):
    return .5 * np.mean((p0 / range - p1 / range) ** 2)


def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1, 2, 0))


def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255. / 2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)


def np2tensor(np_obj):
    # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def tensor2tensorlab(image_tensor, to_norm=True, mc_only=False):
    # image tensor to lab tensor
    from skimage import color

    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if (mc_only):
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    if (to_norm and not mc_only):
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
        img_lab = img_lab / 100.

    return np2tensor(img_lab)


class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if (self.colorspace == 'RGB'):
            (N, C, X, Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y),
                               dim=3).view(N)
            return value
        elif (self.colorspace == 'Lab'):
            value = l2(tensor2np(tensor2tensorlab(in0.data, to_norm=False)),
                       tensor2np(tensor2tensorlab(in1.data, to_norm=False)), range=100.).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            # if (self.use_gpu):
            #     ret_var = ret_var.cuda()
            return ret_var


def dssim(p0, p1, range=255.):
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if (self.colorspace == 'RGB'):
            value = dssim(1. * tensor2im(in0.data), 1. * tensor2im(in1.data), range=255.).astype('float')
        elif (self.colorspace == 'Lab'):
            value = dssim(tensor2np(tensor2tensorlab(in0.data, to_norm=False)),
                          tensor2np(tensor2tensorlab(in1.data, to_norm=False)), range=100.).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        # if (self.use_gpu):
        #     ret_var = ret_var.cuda()
        return ret_var



class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False,
                   model_path=None,
                   use_gpu=True, printNet=False, spatial=False,
                   is_train=False, lr=.0001, beta1=0.5, version='0.1'):
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
        '''
        BaseModel.initialize(self, use_gpu=use_gpu)

        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = '%s [%s]' % (model, net)

        if (self.model == 'net-lin'):  # pretrained net + linear layer
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net,
                               use_dropout=True, spatial=spatial, version=version, lpips=True)
            kw = dict(map_location='cpu')
            if (model_path is None):
                import inspect
                model_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'lpips_models', f'{net}.pth'))

            if (not is_train):
                self.net.load_state_dict(torch.load(model_path, **kw, weights_only=True), strict=False)

        elif (self.model == 'net'):  # pretrained network
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif (self.model in ['L2', 'l2']):
            self.net = L2(use_gpu=use_gpu, colorspace=colorspace)  # not really a network, only for testing
            self.model_name = 'L2'
        elif (self.model in ['DSSIM', 'dssim', 'SSIM', 'ssim']):
            self.net = DSSIM(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.trainable_parameters = list(self.net.parameters())

        if self.is_train:  # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = BCERankingLoss()
            self.trainable_parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.trainable_parameters, lr=lr, betas=(beta1, 0.999))
        else:  # test mode
            self.net.eval()

        # if (use_gpu):
            # self.net.to(gpu_ids[0])
            # self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            # if (self.is_train):
            #     self.rankLoss = self.rankLoss.to(device=gpu_ids[0])  # just put this on GPU0

        if (printNet):
            print('---------- Networks initialized -------------')
            print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''

        return self.net(in0, in1, retPerLayer=retPerLayer)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if (hasattr(module, 'weight') and module.kernel_size == (1, 1)):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

        # if (self.use_gpu):
        #     self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
        #     self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
        #     self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
        #     self.input_judge = self.input_judge.to(device=self.gpu_ids[0])

        # self.var_ref = Variable(self.input_ref, requires_grad=True)
        # self.var_p0 = Variable(self.input_p0, requires_grad=True)
        # self.var_p1 = Variable(self.input_p1, requires_grad=True)

    def forward_train(self):  # run forward pass
        # print(self.net.module.scaling_layer.shift)
        # print(torch.norm(self.net.module.net.slice1[0].weight).item(), torch.norm(self.net.module.lin0.model[1].weight).item())

        assert False, "We shoud've not get here when using LPIPS as a metric"

        self.d0 = self(self.var_ref, self.var_p0)
        self.d1 = self(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0, self.d1, self.input_judge)

        self.var_judge = Variable(1. * self.input_judge).view(self.d0.size())

        self.loss_total = self.rankLoss(self.d0, self.d1, self.var_judge * 2. - 1.)

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self, d0, d1, judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1 < d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                               ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256 / self.var_ref.data.size()[2]

        ref_img = tensor2im(self.var_ref.data)
        p0_img = tensor2im(self.var_p0.data)
        p1_img = tensor2im(self.var_p1.data)

        ref_img_vis = scipy.ndimage.zoom(ref_img, [zoom_factor, zoom_factor, 1], order=0)
        p0_img_vis = scipy.ndimage.zoom(p0_img, [zoom_factor, zoom_factor, 1], order=0)
        p1_img_vis = scipy.ndimage.zoom(p1_img, [zoom_factor, zoom_factor, 1], order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    def save(self, path, label):
        if (self.use_gpu):
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    def update_learning_rate(self, nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type, self.old_lr, lr))
        self.old_lr = lr


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='alex', colorspace='rgb', model_path=None, spatial=False, use_gpu=True):
        # VGG using our perceptually-learned weights (LPIPS metric)
        # def __init__(self, model='net', net='vgg', use_gpu=True): # "default" way of using VGG as a perceptual loss
        super(PerceptualLoss, self).__init__()
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.model = DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, colorspace=colorspace,
                              model_path=model_path, spatial=self.spatial)

    def forward(self, pred, target, normalize=True):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]
        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """

        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        return self.model(target, pred)


class LPIPSScore(PairwiseScore):
    def __init__(self, model='net-lin', net='vgg', model_path=None, use_gpu=True):
        super().__init__()
        self.score = PerceptualLoss(model=model, net=net, model_path=model_path,
                                    use_gpu=use_gpu, spatial=False).eval()
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        batch_values = self.score(pred_batch, target_batch).flatten()
        self.individual_values = np.hstack([
            self.individual_values, batch_values.detach().cpu().numpy()
        ])
        return batch_values


class InpaintingEvaluatorOnline(nn.Module):
    def __init__(self, scores, bins=10, image_key='image', inpainted_key='inpainted',
                 integral_func=None, integral_title=None, clamp_image_range=None):
        """
        :param scores: dict {score_name: EvaluatorScore object}
        :param bins: number of groups, partition is generated by np.linspace(0., 1., bins + 1)
        :param device: device to use
        """
        super().__init__()
        LOGGER.info(f'{type(self)} init called')
        self.scores = nn.ModuleDict(scores)
        self.image_key = image_key
        self.inpainted_key = inpainted_key
        self.bins_num = bins
        self.bin_edges = np.linspace(0, 1, self.bins_num + 1)

        num_digits = max(0, math.ceil(math.log10(self.bins_num)) - 1)
        self.interval_names = []
        for idx_bin in range(self.bins_num):
            start_percent, end_percent = round(100 * self.bin_edges[idx_bin], num_digits), \
                                         round(100 * self.bin_edges[idx_bin + 1], num_digits)
            start_percent = '{:.{n}f}'.format(start_percent, n=num_digits)
            end_percent = '{:.{n}f}'.format(end_percent, n=num_digits)
            self.interval_names.append("{0}-{1}%".format(start_percent, end_percent))

        self.groups = []

        self.integral_func = integral_func
        self.integral_title = integral_title
        self.clamp_image_range = clamp_image_range

        LOGGER.info(f'{type(self)} init done')

    def _get_bins(self, mask_batch):
        batch_size = mask_batch.shape[0]
        area = mask_batch.view(batch_size, -1).mean(dim=-1).detach().cpu().numpy()
        bin_indices = np.clip(np.searchsorted(self.bin_edges, area) - 1, 0, self.bins_num - 1)
        return bin_indices

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Calculate and accumulate metrics for batch. To finalize evaluation and obtain final metrics, call evaluation_end
        :param batch: batch dict with mandatory fields mask, image, inpainted (can be overriden by self.inpainted_key)
        """
        result = {}
        with torch.no_grad():
            image_batch, mask_batch, inpainted_batch = batch[self.image_key].cuda(), batch['mask'].cuda(), batch[self.inpainted_key].cuda()
            if self.clamp_image_range is not None:
                image_batch = torch.clamp(image_batch,
                                          min=self.clamp_image_range[0],
                                          max=self.clamp_image_range[1])
            self.groups.extend(self._get_bins(mask_batch))

            for score_name, score in self.scores.items():
                result[score_name] = score(inpainted_batch, image_batch, mask_batch)
        return result

    def process_batch(self, batch: Dict[str, torch.Tensor]):
        return self(batch)

    def evaluation_end(self, states=None):
        """:return: dict with (score_name, group_type) as keys, where group_type can be either 'overall' or
            name of the particular group arranged by area of mask (e.g. '10-20%')
            and score statistics for the group as values.
        """
        LOGGER.info(f'{type(self)}: evaluation_end called')

        self.groups = np.array(self.groups)

        results = {}
        for score_name, score in self.scores.items():
            LOGGER.info(f'Getting value of {score_name}')
            cur_states = [s[score_name] for s in states] if states is not None else None
            total_results, group_results = score.get_value(groups=self.groups, states=cur_states)
            LOGGER.info(f'Getting value of {score_name} done')
            results[(score_name, 'total')] = total_results

            for group_index, group_values in group_results.items():
                group_name = self.interval_names[group_index]
                results[(score_name, group_name)] = group_values

        if self.integral_func is not None:
            results[(self.integral_title, 'total')] = dict(mean=self.integral_func(results))

        LOGGER.info(f'{type(self)}: reset scores')
        self.groups = []
        for sc in self.scores.values():
            sc.reset()
        LOGGER.info(f'{type(self)}: reset scores done')

        LOGGER.info(f'{type(self)}: evaluation_end done')
        return results


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    LOGGER.info('fid_inception_v3 called')
    inception = torchvision.models.inception_v3(num_classes=1008,
                                    aux_logits=False,
                                    pretrained=False)
    LOGGER.info('models.inception_v3 done')
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    LOGGER.info('fid_inception_v3 patching done')

    state_dict = torch.hub.load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    LOGGER.info('fid_inception_v3 weights downloaded')

    inception.load_state_dict(state_dict)
    LOGGER.info('fid_inception_v3 weights loaded into model')

    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = torchvision.models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def fid_calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(activations_pred, activations_target, eps=1e-6):
    mu1, sigma1 = fid_calculate_activation_statistics(activations_pred)
    mu2, sigma2 = fid_calculate_activation_statistics(activations_target)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        LOGGER.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


class FIDScore(EvaluatorScore):
    def __init__(self, dims=2048, eps=1e-6):
        LOGGER.info("FIDscore init called")
        super().__init__()
        if getattr(FIDScore, '_MODEL', None) is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
        self.model = FIDScore._MODEL
        self.eps = eps
        self.reset()
        LOGGER.info("FIDscore init done")

    def forward(self, pred_batch, target_batch, mask=None):
        activations_pred = self._get_activations(pred_batch)
        activations_target = self._get_activations(target_batch)

        self.activations_pred.append(activations_pred.detach().cpu())
        self.activations_target.append(activations_target.detach().cpu())

        return activations_pred, activations_target

    def get_value(self, groups=None, states=None):
        LOGGER.info("FIDscore get_value called")
        activations_pred, activations_target = zip(*states) if states is not None \
            else (self.activations_pred, self.activations_target)
        activations_pred = torch.cat(activations_pred).cpu().numpy()
        activations_target = torch.cat(activations_target).cpu().numpy()

        total_distance = calculate_frechet_distance(activations_pred, activations_target, eps=self.eps)
        total_results = dict(mean=total_distance)

        if groups is None:
            group_results = None
        else:
            group_results = dict()
            grouping = get_groupings(groups)
            for label, index in grouping.items():
                if len(index) > 1:
                    group_distance = calculate_frechet_distance(activations_pred[index], activations_target[index],
                                                                eps=self.eps)
                    group_results[label] = dict(mean=group_distance)

                else:
                    group_results[label] = dict(mean=float('nan'))

        self.reset()

        LOGGER.info("FIDscore get_value done")

        return total_results, group_results

    def reset(self):
        self.activations_pred = []
        self.activations_target = []

    def _get_activations(self, batch):
        activations = self.model(batch)[0]
        if activations.shape[2] != 1 or activations.shape[3] != 1:
            assert False, \
                'We should not have got here, because Inception always scales inputs to 299x299'
            # activations = F.adaptive_avg_pool2d(activations, output_size=(1, 1))
        activations = activations.squeeze(-1).squeeze(-1)
        return activations


def make_evaluator(kind='default', ssim=True, lpips=True, fid=True, integral_kind=None, lpips_model_path=None, **kwargs):
    logging.info(f'Make evaluator {kind}')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {}
    if ssim:
        metrics['ssim'] = SSIMScore()
    if lpips:
        metrics['lpips'] = LPIPSScore(model_path=lpips_model_path)
    if fid:
        metrics['fid'] = FIDScore().to(device)
        
    if integral_kind is None:
        integral_func = None
    elif integral_kind == 'ssim_fid100_f1':
        integral_func = ssim_fid100_f1
    elif integral_kind == 'lpips_fid100_f1':
        integral_func = lpips_fid100_f1
    else:
        raise ValueError(f'Unexpected integral_kind={integral_kind}')

    if kind == 'default':
        return InpaintingEvaluatorOnline(scores=metrics,
                                         integral_func=integral_func,
                                         integral_title=integral_kind,
                                         **kwargs)


def flatten_dict(dct):
    result = {}
    for k, v in dct.items():
        if isinstance(k, tuple):
            k = "_".join(k)
        if isinstance(v, dict):
            for sub_k, sub_v in flatten_dict(v).items():
                result[f"{k}_{sub_k}"] = sub_v
        else:
            result[k] = v
    return result


class PLBase(lightning.LightningModule):
    def __init__(self, model_config, loss_config, optimizer_config):
        super().__init__()
        LOGGER.info("Lama PLBase init called.")
        self.model_config = model_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        
        self.generator = FFCResNetGenerator(**model_config.generator)
        self.discriminator = NLayerDiscriminator(**model_config.discriminator)
        self.loss_adv_fn = NonSaturatingWithR1(**loss_config.adversarial)
        self.loss_resnet_perceptual_fn = ResNetPL(**loss_config.resnet_pl)

        self.val_evaluator = make_evaluator(**model_config.evaluator)
        self.test_evaluator = make_evaluator(**model_config.evaluator)
        self.val_evaluator.requires_grad_(False)
        self.test_evaluator.requires_grad_(False)

        self.automatic_optimization = False
        self.val_step_outputs = [[], []]

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        
        image = batch['image']
        mask = batch['mask']
        masked_image = image * (1 - mask)
        image_in = torch.cat([masked_image, mask], dim=1)
        output = self.generator(image_in)
        inpainted = masked_image + output * mask

        # optimize generator
        loss, logdict_g = self.generator_loss(image, mask, output)
        optimizer_g.zero_grad()
        self.manual_backward(loss)
        self.clip_gradient(optimizer_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer_g.step()

        # optimize discriminator
        output = self.generator(image_in)
        loss, logdict_d = self.discriminator_loss(image, mask, output.detach())
        optimizer_d.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(optimizer_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer_d.step()
        
        logdict = logdict_d | logdict_g
        logdict = {f'train/{k}': v for k, v in logdict.items()}
        self.log_dict(logdict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=image.size(0), sync_dist=True)
        
        if self.global_step % 1000 == 0 and self.global_rank == 0:
            with torch.inference_mode():
                discr_output_real, _ = self.discriminator(image)
                discr_output_fake, _ = self.discriminator(output)
                image_size = image.shape[2:]
                discr_output_real = F.interpolate(discr_output_real, size=image_size, mode="nearest")
                discr_output_fake = F.interpolate(discr_output_fake, size=image_size, mode="nearest")
            log_image_dict = {
                "image": image,
                "mask": mask,
                "predicted_image": output,
                "discr_output_fake": discr_output_fake,
                "discr_output_real": discr_output_real,
                "inpainted": inpainted,
            }
            self.log_image(log_image_dict, batch_idx, mode='train')

        return loss

    def generator_loss(self, image, mask, output):
        per_pixel_l1 = F.l1_loss(output, image, reduction='none')
        pixel_weights = mask * self.loss_config.l1.weight_missing + (1 - mask) * self.loss_config.l1.weight_known
        loss_l1 = (pixel_weights * per_pixel_l1).mean()

        discr_real_pred, discr_real_features = self.discriminator(image)
        discr_fake_pred, discr_fake_features = self.discriminator(output)
        loss_g = F.softplus(-discr_fake_pred)
        loss_g = loss_g.mean() * self.loss_config.adversarial.weight
        # loss_g = self.loss_adv_fn.generator_loss(
        #     real_batch=image, fake_batch=output, discr_real_pred=discr_real_pred, discr_fake_pred=discr_fake_pred,
        #     mask=mask
        # )
        loss = loss_l1 + loss_g

        if self.loss_config.feature_matching.weight > 0:
            loss_fm = self.feature_matching_loss(discr_fake_features, discr_real_features, mask=None)
            loss_fm = loss_fm * self.loss_config.feature_matching.weight
            loss = loss + loss_fm
        
        if self.loss_config.resnet_pl.weight > 0:
            loss_resnet_pl = self.loss_resnet_perceptual_fn(output, image)
            loss_resnet_pl = loss_resnet_pl * self.loss_config.resnet_pl.weight
            loss = loss + loss_resnet_pl
        
        logdict = {
            'loss_l1': loss_l1.item(),
            'loss_g': loss_g.item(),
            'loss_fm': loss_fm.item(),
            'loss_resnet_pl': loss_resnet_pl.item(),
        }
        return loss, logdict

    def feature_matching_loss(self, fake_features: List[torch.Tensor], target_features: List[torch.Tensor], mask=None):
        if mask is None:
            res = torch.stack([F.mse_loss(fake_feat, target_feat)
                            for fake_feat, target_feat in zip(fake_features, target_features)]).mean()
        else:
            res = 0
            norm = 0
            for fake_feat, target_feat in zip(fake_features, target_features):
                cur_mask = F.interpolate(mask, size=fake_feat.shape[-2:], mode='bilinear', align_corners=False)
                error_weights = 1 - cur_mask
                cur_val = ((fake_feat - target_feat).pow(2) * error_weights).mean()
                res = res + cur_val
                norm += 1
            res = res / norm
        return res
    
    def discriminator_loss(self, image, mask, output):
        image.requires_grad = True    # ?
        discr_real_pred, discr_real_features = self.discriminator(image)
        discr_fake_pred, discr_fake_features = self.discriminator(output)
        loss_d, logdict = self.loss_adv_fn.discriminator_loss(
            real_batch=image, fake_batch=output, discr_real_pred=discr_real_pred, discr_fake_pred=discr_fake_pred, mask=mask
        )
        logdict['loss_d'] = loss_d.item()
        return loss_d, logdict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image = batch['image']
        mask = batch['mask']
        masked_image = image * (1 - mask)
        image_in = torch.cat([masked_image, batch['mask']], dim=1)
        output = self.generator(image_in)
        inpainted = masked_image + output * mask
        batch["predicted_image"] = output.detach()
        batch["inpainted"] = inpainted.detach()

        loss_g, logdict1 = self.generator_loss(image, mask, output)
        loss_d, logdict2 = self.discriminator_loss(image, mask, output)
        logdict = logdict1 | logdict2
        if dataloader_idx == 0:
            logdict = {f'val/{k}': v for k, v in logdict.items()}
        else:
            logdict = {f'test/{k}': v for k, v in logdict.items()}
        self.log_dict(logdict, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=image.size(0), sync_dist=True, add_dataloader_idx=False)
        
        if batch_idx % 100 == 0 and self.global_rank == 0:
            with torch.inference_mode():
                discr_output_real, _ = self.discriminator(image)
                discr_output_fake, _ = self.discriminator(output)
                image_size = image.shape[2:]
                discr_output_real = F.interpolate(discr_output_real, size=image_size, mode="nearest")
                discr_output_fake = F.interpolate(discr_output_fake, size=image_size, mode="nearest")
            log_image_dict = {
                "image": image,
                "mask": mask,
                "predicted_image": output,
                "discr_output_fake": discr_output_fake,
                "discr_output_real": discr_output_real,
                "inpainted": inpainted,
            }
            self.log_image(log_image_dict, batch_idx, mode='val')
        
        if dataloader_idx == 0:
            logdict['val/evaluator_state'] = self.val_evaluator(batch)
        elif dataloader_idx == 1:
            logdict['test/evaluator_state'] = self.test_evaluator(batch)
        self.val_step_outputs[dataloader_idx].append(logdict)
        return logdict

    def on_validation_epoch_end(self):
        outputs = [step_out for out_group in self.val_step_outputs for step_out in out_group]

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        val_evaluator_states = [s['val/evaluator_state'] for s in outputs if 'val/evaluator_state' in s]
        val_evaluator_res = self.val_evaluator.evaluation_end(states=val_evaluator_states)
        val_evaluator_res_df = pd.DataFrame(val_evaluator_res).stack(1).unstack(0)
        val_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
        LOGGER.info(f'Validation metrics after epoch #{self.current_epoch}, '
                    f'total {self.global_step} iterations:\n{val_evaluator_res_df}')
        for k, v in flatten_dict(val_evaluator_res).items():
            self.log(f'val/{k}', v, add_dataloader_idx=False)
        
        test_evaluator_states = [s['test/evaluator_state'] for s in outputs if 'test/evaluator_state' in s]
        test_evaluator_res = self.test_evaluator.evaluation_end(states=test_evaluator_states)
        test_evaluator_res_df = pd.DataFrame(test_evaluator_res).stack(1).unstack(0)
        test_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
        LOGGER.info(f'Test metrics after epoch #{self.current_epoch}, '
                    f'total {self.global_step} iterations:\n{test_evaluator_res_df}')
        for k, v in flatten_dict(test_evaluator_res).items():
            self.log(f'test/{k}', v, add_dataloader_idx=False)
    
    def configure_optimizers(self):
        return [
            {'optimizer': torch.optim.Adam(self.generator.parameters(), lr=0.001)},
            {'optimizer': torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)}
        ]

    @torch.no_grad()
    def log_image(self, batch, batch_idx, mode):
        keys = ['image', 'predicted_image', 'discr_output_fake', 'discr_output_real', 'inpainted']
        actual_min = batch['image'].min()
        actual_max = batch['image'].max()
        if actual_min < 0 or actual_max > 1:
            warnings.warn(f"""DirectoryVisualizer target image must be in 0..1 range,
                           but it ranges {actual_min}..{actual_max}""")
        batch = {k: tens.detach().cpu().numpy() for k, tens in batch.items() if k in keys or k == 'mask'}
        batch_size = batch['image'].shape[0]
        items_to_vis = min(batch_size, 10)
        result = []
        for i in range(items_to_vis):
            cur_dct = {k: tens[i] for k, tens in batch.items()}
            result.append(
                visualize_mask_and_images(
                    cur_dct, keys, last_without_mask=True,
                    rescale_keys=['discr_output_fake', 'discr_output_real']
                )
            )
        vis_img = np.concatenate(result, axis=0)

        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')

        curoutdir = os.path.join(self.trainer.default_root_dir, 'log_images', f'epoch{self.current_epoch:04d}_{mode}')
        os.makedirs(curoutdir, exist_ok=True)
        out_fname = os.path.join(curoutdir, f'batch{batch_idx:07d}.jpg')

        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis_img)


def visualize_mask_and_images(images_dict: Dict[str, np.ndarray], keys: List[str],
                              last_without_mask=True, rescale_keys=None, mask_only_first=None,
                              black_mask=False) -> np.ndarray:
    COLORS, _ = generate_colors(151) # 151 - max classes for semantic segmentation

    mask = images_dict['mask'] > 0.5
    result = []
    for i, k in enumerate(keys):
        img = images_dict[k]
        img = np.transpose(img, (1, 2, 0))

        if rescale_keys is not None and k in rescale_keys:
            img = img - img.min()
            img /= img.max() + 1e-5
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif (img.shape[2] > 3):
            img_classes = img.argmax(2)
            img = skimage.color.label2rgb(img_classes, colors=COLORS)

        if mask_only_first:
            need_mark_boundaries = i == 0
        else:
            need_mark_boundaries = i < len(keys) - 1 or not last_without_mask

        if need_mark_boundaries:
            if black_mask:
                img = img * (1 - mask[0][..., None])
            img = skimage.segmentation.mark_boundaries(img,
                                  mask[0],
                                  color=(1., 0., 0.),
                                  outline_color=(1., 1., 1.),
                                  mode='thick')
        result.append(img)
    return np.concatenate(result, axis=1)


def generate_colors(nlabels, type='bright', first_color_black=False, last_color_black=True, verbose=False):
    # https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return randRGBcolors, random_colormap


class Trainer():
    def __init__(self, model_config, loss_config, optimizer_config, device):
        self.model_config = model_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.device = device

        generator = FFCResNetGenerator(**model_config.generator)
        # Don't forget! And it still doesn't work for 1 gpu
        generator = nn.SyncBatchNorm.convert_sync_batchnorm(generator)
        generator = generator.cuda()
        # generator = torch.compile(generator)
        self.generator = DistributedDataParallel(generator, device_ids=[device])
        discriminator = NLayerDiscriminator(**model_config.discriminator)
        discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        discriminator = discriminator.cuda()
        # discriminator = torch.compile(discriminator)
        self.discriminator = DistributedDataParallel(discriminator, device_ids=[device])

        self.loss_adv_fn = NonSaturatingWithR1(**loss_config.adversarial)
        self.loss_resnet_perceptual_fn = ResNetPL(**loss_config.resnet_pl).cuda()
        
        self.val_evaluator = make_evaluator(**model_config.evaluator).cuda()
        self.val_evaluator.requires_grad_(False)

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=optimizer_config.generator_lr)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=optimizer_config.discriminator_lr)

    def on_train_epoch_start(self):
        self.generator.train()
        self.discriminator.train()

    def train_step(self, batch, global_step, epoch, batch_idx, logdir):        
        image = batch['image'].cuda()
        mask = batch['mask'].cuda()
        masked_image = image * (1 - mask)
        image_in = torch.cat([masked_image, mask], dim=1)
        output = self.generator(image_in)
        inpainted = masked_image + output * mask

        # optimize generator
        loss, logdict_g = self.generator_loss(image, mask, output)
        self.optimizer_g.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.optimizer_g.step()

        # optimize discriminator
        output = self.generator(image_in).detach()
        image.requires_grad = True    # ?
        discr_real_pred, discr_real_features = self.discriminator(image)
        discr_fake_pred, discr_fake_features = self.discriminator(output)
        loss_d, logdict_d = self.loss_adv_fn.discriminator_loss(
            real_batch=image, fake_batch=output, discr_real_pred=discr_real_pred, discr_fake_pred=discr_fake_pred, mask=mask
        )
        logdict_d['loss_d'] = loss_d.item()
        self.optimizer_d.zero_grad()
        loss_d.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.optimizer_d.step()
        
        self.logdict = logdict_d | logdict_g
        
        if global_step % 1000 == 0 and dist.get_rank() == 0:
            dirname = os.path.join(logdir, "log_images", "train")
            os.makedirs(dirname, exist_ok=True)
            with torch.inference_mode():
                image_size = image.shape[2:]
                discr_output_real = F.interpolate(discr_real_pred, size=image_size, mode="nearest")
                discr_output_fake = F.interpolate(discr_fake_pred, size=image_size, mode="nearest")
            log_image_dict = {
                "image": image,
                "mask": mask,
                "predicted_image": output,
                "discr_output_fake": discr_output_fake,
                "discr_output_real": discr_output_real,
                "inpainted": inpainted,
            }
            self.log_image(dirname, global_step, epoch, log_image_dict)

    def generator_loss(self, image, mask, output):
        per_pixel_l1 = F.l1_loss(output, image, reduction='none')
        pixel_weights = mask * self.loss_config.l1.weight_missing + (1 - mask) * self.loss_config.l1.weight_known
        loss_l1 = (pixel_weights * per_pixel_l1).mean()

        discr_real_pred, discr_real_features = self.discriminator(image)
        discr_fake_pred, discr_fake_features = self.discriminator(output)
        loss_g = F.softplus(-discr_fake_pred)
        loss_g = loss_g.mean() * self.loss_config.adversarial.weight
        loss = loss_l1 + loss_g

        if self.loss_config.feature_matching.weight > 0:
            loss_fm = self.feature_matching_loss(discr_fake_features, discr_real_features, mask=None)
            loss_fm = loss_fm * self.loss_config.feature_matching.weight
            loss = loss + loss_fm
        
        if self.loss_config.resnet_pl.weight > 0:
            loss_resnet_pl = self.loss_resnet_perceptual_fn(output, image)
            loss_resnet_pl = loss_resnet_pl * self.loss_config.resnet_pl.weight
            loss = loss + loss_resnet_pl
        
        logdict = {
            'loss_l1': loss_l1.item(),
            'loss_g': loss_g.item(),
            'loss_fm': loss_fm.item(),
            'loss_resnet_pl': loss_resnet_pl.item(),
        }
        return loss, logdict

    def feature_matching_loss(self, fake_features: List[torch.Tensor], target_features: List[torch.Tensor], mask=None):
        if mask is None:
            res = torch.stack([F.mse_loss(fake_feat, target_feat)
                            for fake_feat, target_feat in zip(fake_features, target_features)]).mean()
        else:
            res = 0
            norm = 0
            for fake_feat, target_feat in zip(fake_features, target_features):
                cur_mask = F.interpolate(mask, size=fake_feat.shape[-2:], mode='bilinear', align_corners=False)
                error_weights = 1 - cur_mask
                cur_val = ((fake_feat - target_feat).pow(2) * error_weights).mean()
                res = res + cur_val
                norm += 1
            res = res / norm
        return res
    
    def discriminator_loss(self, image, mask, output):
        image.requires_grad = True    # ?
        discr_real_pred, discr_real_features = self.discriminator(image)
        discr_fake_pred, discr_fake_features = self.discriminator(output)
        loss_d, logdict = self.loss_adv_fn.discriminator_loss(
            real_batch=image, fake_batch=output, discr_real_pred=discr_real_pred, discr_fake_pred=discr_fake_pred, mask=mask
        )
        logdict['loss_d'] = loss_d.item()
        return loss_d, logdict

    def on_train_epoch_end(self, epoch):
        return dict()

    def on_val_epoch_start(self):
        self.generator.eval()
        self.discriminator.eval()
        self.val_step_logs = []
        self.val_step_states = []

    @torch.no_grad()
    def val_step(self, batch, global_step, epoch, batch_idx, logdir):
        image = batch['image'].cuda()
        mask = batch['mask'].cuda()
        masked_image = image * (1 - mask)
        image_in = torch.cat([masked_image, mask], dim=1)
        output = self.generator(image_in)
        inpainted = masked_image + output * mask
        batch["predicted_image"] = output.detach()
        batch["inpainted"] = inpainted.detach()

        loss_g, logdict1 = self.generator_loss(image, mask, output)
        loss_d, logdict2 = self.discriminator_loss(image, mask, output)
        logdict = logdict1 | logdict2
        
        if batch_idx % 100 == 0 and dist.get_rank() == 0:
            dirname = os.path.join(logdir, "log_images", "val")
            os.makedirs(dirname, exist_ok=True)
            discr_output_real, _ = self.discriminator(image)
            discr_output_fake, _ = self.discriminator(output)
            image_size = image.shape[2:]
            discr_output_real = F.interpolate(discr_output_real, size=image_size, mode="nearest")
            discr_output_fake = F.interpolate(discr_output_fake, size=image_size, mode="nearest")
            log_image_dict = {
                "image": image,
                "mask": mask,
                "predicted_image": output,
                "discr_output_fake": discr_output_fake,
                "discr_output_real": discr_output_real,
                "inpainted": inpainted,
            }
            self.log_image(dirname, global_step, epoch, log_image_dict)
        
        self.val_step_logs.append(logdict)
        self.val_step_states.append(self.val_evaluator(batch))


    def on_val_epoch_end(self, dataset_name, dataset, logdir):
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        val_evaluator_res = self.val_evaluator.evaluation_end(states=self.val_step_states)
        val_evaluator_res_df = pd.DataFrame(val_evaluator_res).stack(1).unstack(0)
        val_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
        return flatten_dict(val_evaluator_res)

    def log_step(self, logdir, global_step, epoch, batch_idx):
        return self.logdict

    @torch.no_grad()
    def log_image(self, logdir, global_step, epoch, batch):
        keys = ['image', 'predicted_image', 'discr_output_fake', 'discr_output_real', 'inpainted']
        actual_min = batch['image'].min()
        actual_max = batch['image'].max()
        if actual_min < 0 or actual_max > 1:
            warnings.warn(f"""DirectoryVisualizer target image must be in 0..1 range,
                           but it ranges {actual_min}..{actual_max}""")
        batch = {k: tens.detach().cpu().numpy() for k, tens in batch.items() if k in keys or k == 'mask'}
        batch_size = batch['image'].shape[0]
        items_to_vis = min(batch_size, 8)
        result = []
        for i in range(items_to_vis):
            cur_dct = {k: tens[i] for k, tens in batch.items()}
            result.append(
                visualize_mask_and_images(
                    cur_dct, keys, last_without_mask=True,
                    rescale_keys=['discr_output_fake', 'discr_output_real']
                )
            )
        vis_img = np.concatenate(result, axis=0)
        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')

        out_fname = os.path.join(logdir, f'gs{global_step}-e{epoch}_pred.png')
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis_img)
    
    def get_model_state_dict(self):
        return {
            "generator": self.generator.module.state_dict(),
            "discriminator": self.discriminator.module.state_dict(),
        }
    
    def get_optimizer_state_dict(self):
        return {
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }
    
    def get_lr_scheduler_state_dict(self):
        return self.lr_scheduler.state_dict()