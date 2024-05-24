import lightning
import torch
import torch.nn as nn

import utils
from diffusers.model import ema
from diffusers.model.vae import DiagonalGaussianDistribution


class StableDiffusion_TextualInversion(nn.Module):
    """A custom version of Stable Diffusion Textual Inversion Model"""
    def __init__(self, model_config):
        super().__init__()
        self.prediction_type = model_config.prediction_type
        self.scale_factor = model_config.scale_factor

        self.first_stage_model = utils.instantiate_from_config(model_config.first_stage_config)
        # support additional tokens and embeddings
        self.cond_stage_model = utils.instantiate_from_config(model_config.cond_stage_config)
        self.diffusion_model = utils.instantiate_from_config(model_config.unet_config)

        # freeze ae and text encoder
        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)
    
    @property
    def device(self):
        return self.diffusion_model.time_embed[0].weight.device
    
    def trainable_parameters(self):
        return self.cond_stage_model.trainable_parameters()
    
    def forward(self, batch):
        batch_size = batch['images'].size(0)
        latents = self.encode_first_stage(batch['images'])
        cond_prompt = self.cond_stage_model(batch['texts'])
 
        noise = torch.randn_like(latents)
        time = torch.randint(0, batch['num_train_timesteps'], (batch_size,), device=self.device)
        # TODO noise offset
        alphas_cumprod_t = batch['alphas_cumprod'][time][(...,) + (None,) * 3]    # shape (b, 1, 1, 1)
        xt = torch.sqrt(alphas_cumprod_t) * batch['image'] + torch.sqrt(1 - alphas_cumprod_t) * noise
        output = self.diffusion_model(xt, time, cond_prompt)

        # TODO v prediction
        target = noise

        return output, target
    
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

        self.model = utils.instantiate_from_config(model_config)
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
        batch['alphas_cumprod'] = self.sampler.alphas_cumprod
        output, target = self.model(batch)
        loss, logdict = self.loss_fn(output, target)
        self.log_dict(logdict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch['image'])
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model.cond_stage_model.freeze_original_embedding()
    
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
                optimizer, start_factor=0, total_iters=self.optimizer_config.warmup
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