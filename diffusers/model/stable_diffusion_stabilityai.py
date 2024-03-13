import torch
import torch.nn as nn

import utils
from diffusers.model.vae import DiagonalGaussianDistribution


class StableDiffusion_StabilityAI(nn.Module):
    """A custom version of StabilityAI Stable Diffusion Model"""
    def __init__(self, model_config):
        super().__init__()
        self.prediction_type = model_config.prediction_type
        self.scale_factor = model_config.scale_factor

        self.first_stage_model = utils.instantiate_from_config(model_config.first_stage_config)
        
        if model_config.cond_stage_config:
            self.cond_stage_model = utils.instantiate_from_config(model_config.cond_stage_config)
        
        self.diffusion_model = utils.instantiate_from_config(model_config.unet_config)
        # self.diffusion_model.log_var = 

        # freeze ae and text encoder
        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)
        self.cond_stage_model.eval()
        self.cond_stage_model.requires_grad_(False)
    
    @property
    def device(self):
        return self.diffusion_model.time_embed[0].weight.device
    
    def forward_diffusion_model(self, xt, t, cond_prompt):
        output = self.diffusion_model(xt, t, context=cond_prompt)
        return output
    
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    def get_learned_conditioning(self, cond):
        # cond: list of text
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            c = self.cond_stage_model.encode(cond)
            if isinstance(c, DiagonalGaussianDistribution):
                c = c.mode()
        else:
            c = self.cond_stage_model(cond)
        return c
    
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)


class StableDiffusionImagingCondition_StabilityAI(StableDiffusion_StabilityAI):
    def forward_diffusion_model(self, xt, t, cond_prompt, cond_imaging):
        xt = torch.cat((xt, cond_imaging), dim=1)
        output = self.diffusion_model(xt, t, context=cond_prompt)
        return output