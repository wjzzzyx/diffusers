import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from torch_utils import replace_substring_in_state_dict_if_present
from diffusers.model.stable_diffusion_stabilityai import StableDiffusion_StabilityAI

suffix_conversion = {
    'conv1': 'in_layers_2',
    'conv2': 'out_layers_3',
    'norm1': 'in_layers_0',
    'norm2': 'out_layers_0',
    'time_emb_proj': 'emb_layers_1',
    'conv_shortcut': 'skip_connection',
}

def convert_lora_module_name(key):
    m = re.match(r"lora_unet_conv_in(.*)", key)
    if m:
        return f'diffusion_model_input_blocks_0_0{m.group(1)}'
    m = re.match(r"lora_unet_conv_out(.*)", key)
    if m:
        return f'diffusion_model_out_2{m.group(1)}'
    m = re.match(r"lora_unet_time_embedding_linear_(\d+)(.*)", key)
    if m:
        return f'diffusion_model_time_embed_{int(m.group(1)) * 2 - 2}{m.group(2)}'
    m = re.match(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)", key)
    if m:
        return f"diffusion_model_input_blocks_{1 + int(m.group(1)) * 3 + int(m.group(2))}_1_{m.group(3)}"
    m = re.match(r"lora_unet_down_blocks_(\d+)_resnets_(\d+)_(.+)", key)
    if m:
        suffix = suffix_conversion.get(m.group(3), m.group(3))
        return f"diffusion_model_input_blocks_{1 + int(m.group(1)) * 3 + int(m.group(2))}_0_{suffix}"
    m = re.match(r"lora_unet_mid_block_resnets_(\d+)_(.+)", key)
    if m:
        suffix = suffix_conversion.get(m.group(2), m.group(2))
        return f"diffusion_model_middle_block_{int(m.group(1)) * 2}_{suffix}"
    m = re.match(r"lora_unet_mid_block_attentions_(\d+)_(.+)", key)
    if m:
        return f"diffusion_model_middle_block_1_{m.group(2)}"
    m = re.match(r"lora_unet_up_blocks_(\d+)_resnets_(\d+)_(.+)", key)
    if m:
        suffix = suffix_conversion.get(m.group(3), m.group(3))
        return f"diffusion_model_output_blocks_{int(m.group(1)) * 3 + int(m.group(2))}_0_{suffix}"
    m = re.match(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)", key)
    if m:
        return f"diffusion_model_output_blocks_{int(m.group(1)) * 3 + int(m.group(2))}_1_{m.group(3)}"
    m = re.match(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv", key)
    if m:
        return f"diffusion_model_input_blocks_{3 + int(m.group(1)) * 3}_0_op"
    m = re.match(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv", key)
    if m:
        return f"diffusion_model_output_blocks_{2 + int(m.group(1)) * 3}_{2 if int(m.group(1)) > 0 else 1}_conv"
    m = re.match(r"lora_te_text_model_encoder_layers_(\d+)_(.+)", key)
    if m:
        return f"cond_stage_model_transformer_text_model_encoder_layers_{m.group(1)}_{m.group(2)}"
    raise ValueError('Unmatched Lora module name {key}')
    

def convert_lora_module_names(state_dict):
    keys = list(state_dict.keys())
    for key in keys:
        new_key = convert_lora_module_name(key)
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
    return state_dict


class LoRABase(nn.Module):
    """ replace forward method in the original module """
    def __init__(
        self, org_module, lora_dim, alpha,
        dropout_module=0.0, dropout=0.0, dropout_rank=0.0
    ):
        super().__init__()
        self.lora_dim = lora_dim
        self.dropout_module = dropout_module
        self.dropout = dropout
        self.dropout_rank = dropout_rank
        self.org_forward = org_module.forward
        org_module.forward = self.forward
        self.register_buffer('alpha', torch.tensor(alpha))
        self.scale = alpha / self.lora_dim
    
    def forward(self, x):
        org_out = self.org_forward(x)

        # dropout the whole module
        if self.training:
            if self.dropout_module > 0 and torch.rand(1) < self.dropout_module:
                return org_out

        x = self.lora_down(x)
        
        if self.training:
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout)
            if self.dropout_rank > 0:
                mask = torch.rand((x.size(0), self.lora_dim, *x.shape[2:]), device=x.device) > self.dropout_rank
                x = x * mask
                scale = self.scale * (1.0 / (1.0 - self.dropout_rank))
            else:
                scale = self.scale
        
        x = self.lora_up(x)
        return org_out + scale * x


class LoRAMLP(LoRABase):
    def __init__(
        self, org_module, lora_dim, alpha,
        dropout_module=0.0, dropout=0.0, dropout_rank=0.0
    ):
        super().__init__(org_module, lora_dim, alpha, dropout_module, dropout, dropout_rank)
        self.lora_down = nn.Linear(org_module.in_features, lora_dim, bias=False)
        self.lora_up = nn.Linear(lora_dim, org_module.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


class LoRAConv(LoRABase):
    def __init__(
        self, org_module, lora_dim, alpha,
        dropout_module=0.0, dropout=0.0, dropout_rank=0.0
    ):
        super().__init__(org_module, lora_dim, alpha, dropout_module, dropout, dropout_rank)
        self.lora_down = nn.Conv2d(
            org_module.in_channels, lora_dim, org_module.kernel_size,
            org_module.stride, org_module.padding, bias=False
        )
        self.lora_up = nn.Conv2d(lora_dim, org_module.out_channels, 1, 1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


class StableDiffusion_Lora(StableDiffusion_StabilityAI):
    def __init__(self, model_config):
        super().__init__(model_config)
        if 'pretrained' in model_config:
            checkpoint = torch.load(model_config.pretrained, map_location='cpu')
            state_dict = checkpoint['state_dict']
            replace_substring_in_state_dict_if_present(state_dict, 'model.diffusion_model', 'diffusion_model')
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
    
        self.add_lora_modules(model_config)
        if 'pretrained_lora' in model_config:
            checkpoint = torch.load(model_config.pretrained_lora, map_location='cpu')
            state_dict = convert_lora_module_names(checkpoint)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
    
    def add_lora_modules(self, model_config):
        self.lora_module_names = list()
        for name, module in self.diffusion_model.named_modules():
            if module.__class__.__name__ == 'SpatialTransformer':
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ['Linear', 'Conv2d']:
                        lora_name = f'lora_unet.{name}.{child_name}'
                        lora_name = lora_name.replace('.', '_')
                        lora_module_cls = LoRAMLP if child_module.__class__.__name__ == 'Linear' else LoRAConv
                        lora_module = lora_module_cls(
                            child_module, model_config.lora_dim, model_config.lora_alpha,
                            model_config.dropout_module, model_config.dropout, model_config.dropout_rank
                        )
                        self.add_module(lora_name, lora_module)
                        self.lora_module_names.append(lora_name)
        
        for name, module in self.cond_stage_model.named_modules():
            if module.__class__.__name__ in ['CLIPAttention', 'CLIPMLP']:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ['Linear', 'Conv2d']:
                        lora_name = f'lora_te.{name}.{child_name}'
                        lora_name = lora_name.replace('.', '_')
                        lora_module_cls = LoRAMLP if child_module.__class__.__name__ == 'Linear' else LoRAConv
                        lora_module = lora_module_cls(
                            child_module, model_config.lora_dim, model_config.lora_alpha,
                            model_config.dropout_module, model_config.dropout, model_config.dropout_rank
                        )
                        self.add_module(lora_name, lora_module)
                        self.lora_module_names.append(lora_name)
        