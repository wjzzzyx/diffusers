import lightning
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn

import utils
from diffusers.model import ema
from diffusers.model.vae import DiagonalGaussianDistribution
from diffusers.sampler.denoiser import KarrasEpsDenoiser, KarrasVDenoiser, KarrasCFGDenoiser
from torch_utils import replace_substring_in_state_dict_if_present


class StableDiffusion_TextualInversion(nn.Module):
    """A custom version of Stable Diffusion Textual Inversion Model"""
    def __init__(self, model_config):
        super().__init__()
        self.prediction_type = model_config.prediction_type
        self.scale_factor = model_config.scale_factor

        self.first_stage_model = utils.instantiate_from_config(model_config.first_stage_config)
        self.cond_stage_model = utils.instantiate_from_config(model_config.cond_stage_config)
        self.diffusion_model = utils.instantiate_from_config(model_config.unet_config)

        # freeze ae and text encoder
        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)

        if 'pretrained' in model_config:
            checkpoint = torch.load(model_config.pretrained, map_location='cpu')
            state_dict = checkpoint['state_dict']
            replace_substring_in_state_dict_if_present(state_dict, 'model.diffusion_model', 'diffusion_model')
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        # support additional tokens and embeddings
        self.cond_stage_model.expand_vocab()
    
    @property
    def device(self):
        return self.diffusion_model.time_embed[0].weight.device
    
    def trainable_parameters(self):
        return self.cond_stage_model.trainable_parameters()
    
    def forward(self, batch):
        batch_size = batch['images'].size(0)
        latents = self.encode_first_stage(batch['images'])
        cond_prompt = self.cond_stage_model(batch['captions'])
 
        noise = torch.randn_like(latents)
        time = torch.randint(0, batch['num_train_timesteps'], (batch_size,), device=self.device)
        batch['time'] = time
        # TODO noise offset
        alphas_cumprod_t = batch['alphas_cumprod'][time][(...,) + (None,) * 3]    # shape (b, 1, 1, 1)
        xt = torch.sqrt(alphas_cumprod_t) * latents + torch.sqrt(1 - alphas_cumprod_t) * noise
        output = self.diffusion_model(xt, time, cond_prompt)

        # TODO v prediction
        target = noise

        return output, target, xt
    
    def forward_diffusion_model(self, xt, t, cond_prompt):
        output = self.diffusion_model(xt, t, cond_prompt)
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


class PLTextualInversion(lightning.LightningModule):
    def __init__(
        self,
        model_config,
        loss_config,
        optimizer_config,
        sampler_config,
        metric_config=None,
        ema_config=None
    ):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.sampler_config = sampler_config
        self.metric_config = metric_config
        self.ema_config = ema_config

        self.model = StableDiffusion_TextualInversion(model_config)
        self.loss_fn = utils.instantiate_from_config(loss_config)
        self.sampler = utils.instantiate_from_config(sampler_config)
        if self.ema_config:
            self.model_ema = utils.instantiate_from_config(model_config)
            self.model_ema.requires_grad_(False)
            self.ema_helper = ema.EMAHelper(
                max_decay=ema_config.max_decay,
                use_ema_warmup=ema_config.warmup,
                inv_gamma=ema_config.inv_gamma,
                power=ema_config.power,
            )
    
    def training_step(self, batch, batch_idx):
        batch['num_train_timesteps'] = self.sampler.num_train_steps
        batch['alphas_cumprod'] = self.sampler.alphas_cumprod.to(self.device)
        output, target, xt = self.model(batch)
        loss, logdict = self.loss_fn(output, target, batch)
        logdict = {f'train/{k}': v for k, v in logdict.items()}
        self.log_dict(logdict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch['image'])
        
        alphas_cumprod_t = batch['alphas_cumprod'][batch['time']][(...,) + (None,) * 3]    # shape: batch, 1, 1, 1
        pred = (xt - torch.sqrt(1 - alphas_cumprod_t) * output) / torch.sqrt(alphas_cumprod_t)
        pred = self.model.decode_first_stage(pred)
        pred = (pred + 1) / 2
        log_image_dict = {'image': (batch['image'] + 1) / 2, 'pred': pred}
        self.log_image(log_image_dict, ['image', 'pred'], batch_idx, mode='train')
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model.cond_stage_model.freeze_original_embedding()
    
    def validation_step(self, batch, batch_idx):
        alphas_cumprod = self.sampler.alphas_cumprod.to(self.device)
        if self.model.prediction_type == 'epsilon':
            denoiser = KarrasEpsDenoiser(self.model, alphas_cumprod)
        elif self.model.prediction_type == 'v':
            denoiser = KarrasVDenoiser(self.model, alphas_cumprod)
        denoiser = KarrasCFGDenoiser(denoiser, 7)

        batch_size = len(batch['images'])
        cond_pos_prompt = self.model.cond_stage_model(batch['captions'])
        cond_neg_prompt = self.model.cond_stage_model(['' for _ in range(batch_size)])
        denoiser_args = {'cond_pos_prompt': cond_pos_prompt, 'cond_neg_prompt': cond_neg_prompt}
        samples = self.sampler.sample(
            denoiser, batch_size=batch_size, image_shape=(4, 64, 64), denoiser_args=denoiser_args
        )
        samples = self.model.decode_first_stage(samples)
        samples = torch.clamp((samples + 1) / 2, min=0, max=1)
        log_image_dict = {'image': samples}
        log_keys=['image']
        self.log_image(log_image_dict, log_keys, batch_idx, mode='validation')
    
    def on_save_checkpoint(self, checkpoint):
        ti_embeds_dict = self.model.cond_stage_model.get_ti_embedding(checkpoint['state_dict'])
        checkpoint['state_dict'] = ti_embeds_dict
    
    def configure_optimizers(self):
        trainable_params = self.model.trainable_parameters()
        optimizer = utils.get_obj_from_str(self.optimizer_config.optimizer)(
            trainable_params, **self.optimizer_config.optimizer_params
        )
        lr_scheduler = utils.get_obj_from_str(self.optimizer_config.lr_scheduler)(
            optimizer, **self.optimizer_config.lr_scheduler_params
        )
        if self.optimizer_config.warmup:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=self.optimizer_config.warmup
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, lr_scheduler], milestones=[self.optimizer_config.warmup]
            )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step'
            }
        }
    
    @torch.no_grad()
    def log_image(self, batch, keys, batch_idx, mode):
        """
        Args:
            batch: dictionary, key: str -> value: tensor
        """
        dirname = os.path.join(self.trainer.default_root_dir, 'log_images', mode)
        os.makedirs(dirname, exist_ok=True)
        for key in keys:
            image_t = batch[key].permute(0, 2, 3, 1).squeeze(-1)
            image_np = image_t.detach().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            for i in range(image_np.shape[0]):
                filename = f'gs-{self.global_step:04}_e-{self.current_epoch:04}_b-{batch_idx:04}-{i:02}_{key}.png'
                Image.fromarray(image_np[i]).save(os.path.join(dirname, filename))