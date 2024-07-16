import safetensors
import torch
import torch.nn as nn

from diffusers.model.stable_diffusion_stabilityai import StableDiffusion_StabilityAI
from diffusers.model.lora import LoraNetwork, convert_lora_module_names


class StableDiffusion_TI_Lora(StableDiffusion_StabilityAI):
    """A custom version of Stable Diffusion with Textual Inversion and Lora"""
    def __init__(self, model_config):
        super().__init__(model_config)
        # freeze ae and unet
        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)
        
        # support additional tokens and embeddings
        if 'pretrained_ti' in model_config:
            names = list()
            embeddings = list()
            for cfg in model_config.pretrained_ti:
                names.append(cfg.name)
                checkpoint = torch.load(cfg.path, map_location='cpu')
                embeddings.append(checkpoint['string_to_param']['*'])
            self.cond_stage_model.expand_vocab_from_weight(names, embeddings)
        else:
            self.cond_stage_model.expand_vocab()
        
        # lora modules
        self.lora_networks = nn.ModuleList()
        if 'pretrained_lora' in model_config:
            for cfg in model_config.pretrained_lora:
                if cfg.path.endswith('safetensors'):
                    checkpoint = safetensors.torch.load_file(cfg.path, device='cpu')
                else:
                    checkpoint = torch.load(cfg.path, map_location='cpu')
                state_dict = convert_lora_module_names(checkpoint)
                network = LoraNetwork(cfg.multiplier)
                network.add_lora_modules_from_weight(state_dict, self.diffusion_model, self.cond_stage_model)
                missing, unexpected = network.load_state_dict(state_dict, strict=False)
                self.lora_networks.append(network)
        else:
            network = LoraNetwork(1.0)
            network.add_lora_modules(
                model_config.lora_dim, model_config.lora_alpha, self.diffusion_model, self.cond_stage_model,
                model_config.dropout_module, model_config.dropout, model_config.dropout_rank
            )
            self.lora_networks.append(network)