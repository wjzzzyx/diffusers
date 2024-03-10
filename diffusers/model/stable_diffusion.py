import collections
import lightning
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List

import utils
from diffusers.model import ema
from diffusers.model.modules.unet_2d_condition import UNet2DConditionModel
from diffusers.model.modules.autoencoder_kl import AutoencoderKL


def _convert_deprecated_attention_blocks(self, state_dict: collections.OrderedDict):
    for key in list(state_dict.keys()):
        if 'attention' in key and 'query' in key:
            state_dict[key.replace('.query.', '.to_q.')] = state_dict.pop(key)
        if 'attention' in key and 'key' in key:
            state_dict[key.replace('.key.', '.to_k.')] = state_dict.pop(key)
        if 'attention' in key and 'value' in key:
            state_dict[key.replace('.value.', '.to_v.')] = state_dict.pop(key)
        if 'attention' in key and 'proj_attn' in key:
            state_dict[key.replace('.proj_attn.', '.to_out.0.')] = state_dict.pop(key)


class StableDiffusion(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

        self.unet = UNet2DConditionModel(**model_config.unet)
        self.vae = AutoencoderKL(**model_config.vae)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_config.pretrained, subfolder='tokenizer', local_files_only=True)
        self.text_encoder = CLIPTextModel.from_pretrained(model_config.pretrained, subfolder='text_encoder', local_files_only=True)
        
        ckpt = torch.load(os.path.join(model_config.pretrained, 'unet', 'diffusion_pytorch_model.bin'), map_location='cpu')
        self.unet.load_state_dict(ckpt)
        ckpt = torch.load(os.path.join(model_config.pretrained, 'vae', 'diffusion_pytorch_model.bin'), map_location='cpu')
        _convert_deprecated_attention_blocks(ckpt)
        self.vae.load_state_dict(ckpt)
        
        # ckpt = torch.load(model_config.pretrained, map_location='cpu')
        # self._convert_deprecated_attention_blocks(ckpt)
        # self.load_state_dict(ckpt)

        # freeze submodules
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
    
    @property
    def device(self):
        return self.unet.conv_in.weight.device
    
    def trainable_parameters(self):
        return self.unet.parameters()
    
    def forward(self, feeddict):
        batch_size = feeddict['image'].size(0)
        latent = self.encode_image(feeddict['image'])
        noise = torch.randn_like(latent)
        # noise offset
        # input pertubation
        t = torch.randint(0, self.sampler_config.num_train_steps, (batch_size,), dtype=torch.long, device=latent.device)
        noisy_latent = self.sampler.add_noise(latent, noise, t)
        
        prompt_embeds, _ = self.encode_prompt(feeddict['text'])
        
        if self.model_config.prediction_type == 'epsilon':
            target = noise
        elif self.model_config.prediction_type == 'v_prediction':
            target = self.sampler.get_velocity(latent, noise, t)
        else:
            raise ValueError(f'Unsupported prediction type {self.model_config.prediction_type}.')
        model_output = self.unet(noisy_latent, t, prompt_embeds).sample
        
        if self.model_config.snr_gamma is None:
            loss = F.mse_loss(model_output, target, reduction='mean')
        else:
            # Compute loss weights as Section 3.4 of http://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x0, the original formulation is slightly changed.
            # This is discussed in Section 4.2.
            snr = self.computer_snr(t)
            mse_loss_weights = torch.stack([snr, self.model_config.snr_gamma * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
            # We first calculate the original loss. Then we mean over the non-batch dimensions
            # and rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_output.float(), target.float(), reduction='none')
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        
        logdict = {
            'loss': loss.item(),
            'ema_decay': self.ema_helper.decay,
        }
        return loss, logdict
    
    def compute_snr(self, timesteps):
        """
        Computer SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training
        """
        alphas_cumprod = self.sampler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps]
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps]
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        snr = (alpha / sigma) ** 2
        return snr

    def ema_step(self):
        self.ema_helper.ema_step(self.unet, self.unet_ema)
    
    def encode_prompt(self, pos_prompts: List[str], neg_prompts: List[str] = None):
        text_inputs = self.tokenizer(pos_prompts, padding='max_length', truncation=True, return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        pos_prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0]
        if neg_prompts is None:
            neg_prompts = [''] * len(pos_prompts)
        neg_text_inputs = self.tokenizer(neg_prompts, padding='max_length', truncation=True, return_tensors='pt')
        neg_text_input_ids = neg_text_inputs.input_ids
        attention_mask = neg_text_inputs.attention_mask
        neg_prompt_embeds = self.text_encoder(neg_text_input_ids, attention_mask=attention_mask)[0]
        return pos_prompt_embeds, neg_prompt_embeds
    
    def encode_image(self, images: torch.Tensor):
        return self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor

    def decode_image(self, latents: torch.Tensor, generator=None):
        return self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]


class StableDiffusion_StabilityAI(nn.Module):
    """A custom version of StabilityAI Stable Diffusion Model"""
    def __init__(self, model_config):
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


class PLBase(lightning.LightningModule):
    def __init__(self, model_config, sampler_config, ema_config=None, optimizer_config=None, ckpt_path=None):
        super().__init__()
        self.model_config = model_config
        self.sampler_config = sampler_config
        self.ema_config = ema_config
        self.optimizer_config = optimizer_config

        self.model = StableDiffusion(model_config)

        if self.ema_config:
            self.model_ema = StableDiffusion(model_config)
            self.model_ema.requires_grad_(False)
            self.ema_helper = ema.EMAHelper(
                max_decay=ema_config.max_decay,
                use_ema_warmup=ema_config.warmup,
                inv_gamma=ema_config.inv_gamma,
                power=ema_config.power,
            )
        
        self.sampler = utils.instantiate_from_config(sampler_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
    
    def init_from_ckpt(self, path):
        ckpt = torch.load(path, map_location='cpu')
        missing, unexpected = self.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'Restoring from checkpoint {ckpt}.')
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')
    
    def training_step(self, batch, batch_idx):
        loss, logdict = self.model(batch)
        logdict = {f'train/{k}': v for k, v in logdict.items()}
        self.log_dict(logdict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch['image'].size(0))
        return loss
    
    def predict_step(self, batch, batch_idx):
        batch_size = len(batch['fname'])
        in_channels = self.model.unet.config.in_channels
        image_size = self.model.unet.config.sample_size
        
        pos_prompt_embeds, neg_prompt_embeds = self.model.encode_prompt(
            batch['pos_prompt'], batch['neg_prompt']
        )
        if 'image' in batch:
            latents = self.model.encode_image(batch['image'])
        else:
            latents = self.sampler.sample(
                self.model.unet, batch_size, [in_channels, image_size, image_size],
                cond_pos_prompt=pos_prompt_embeds, cond_neg_prompt=neg_prompt_embeds
            )
        samples = self.model.decode_image(latents)
        
        samples = (samples + 1) / 2
        log_image_dict = {
            'image': samples,
            'fname': batch['fname'],
        }
        log_keys = ['image']
        self.log_image(log_image_dict, log_keys, batch_idx, mode='predict')
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_config:
            self.ema_helper.ema_step(self.model, self.model_ema)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.optimizer_config.learning_rate,
            betas=(self.optimizer_config.adam_beta1, self.optimizer_config.adam_beta2),
            weight_decay=self.optimizer_config.adam_weight_decay,
            eps=self.optimizer_config.adam_epsilon,
        )
        lr_scheduler = get_scheduler()
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'},
        }

    @torch.no_grad()
    def log_image(self, batch, keys, batch_idx, mode):
        """
        Image values should be in [0, 1]
        Args:
            batch: dictionary, key: str -> value: tensor
            keys: keys in batch that needs saving
        """
        dirname = os.path.join(self.trainer.default_root_dir, 'log_images', mode)
        os.makedirs(dirname, exist_ok=True)
        for key in keys:
            image_t = batch[key].permute(0, 2, 3, 1).squeeze(-1)
            image_np = image_t.detach().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            for i in range(image_np.shape[0]):
                filename = f"{os.path.splitext(batch['fname'][i])[0]}_{key}.png"
                Image.fromarray(image_np[i]).save(os.path.join(dirname, filename))