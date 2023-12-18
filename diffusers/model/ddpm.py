import lightning
import os
from PIL import Image
import torch
import torch.nn as nn

from . import ema
from .modules import unet_2d
from diffusers.sampler import sample_ddpm
from diffusers.pipeline import uncond_image_gen


class DDPM(nn.Module):
    def __init__(self, model_config):
        self.unet = unet_2d.UNet2D(model_config.unet)
        self.prediction_type == model_config.prediction_type
    
    def forward(self, xt, t, sampler):
        alpha_cumprod_t = self.alphas_cumprod[t]

        output = self.unet(xt, t)
        # epsilon, x0
        if self.prediction_type == 'epsilon':
            epsilon = output
            sample = (xt - torch.sqrt(1 - alpha_cumprod_t) * output) / torch.sqrt(alpha_cumprod_t)
        elif self.prediction_type == 'sample':
            sample = output
            epsilon = (xt - torch.sqrt(alpha_cumprod_t) * output) / torch.sqrt(1 - alpha_cumprod_t)
        elif self.prediction_type == 'v_prediction':
            raise NotImplementedError('v_prediction is not implemented for DDPM.')

        return {
            'epsilon': epsilon,
            'sample': sample,
        }


class PLBase(lightning.LightningModule):
    def __init__(self, model_config, sampler_config, ema_config, optimizer_config, ckpt_path=None):
        super().__init__()
        self.model_config = model_config
        self.sampler_config = sampler_config
        self.ema_config = ema_config
        self.optimizer_config = optimizer_config

        self.model = DDPM(model_config)
        
        if self.ema_config:
            self.model_ema = DDPM(model_config)
            self.model_ema.requires_grad_(False)
            self.ema_helper = ema.EMAHelper(
                max_decay=ema_config.max_decay,
                use_ema_warmup=ema_config.warmup,
                inv_gamma=ema_config.inv_gamma,
                power=ema_config.power,
            )
        
        self.sampler = sample_ddpm.DDPMSampler(
            num_train_steps=sampler_config.num_train_steps,
            beta_start=sampler_config.beta_start,
            beta_end=sampler_config.beta_end,
            beta_schedule=sampler_config.beta_schedule,
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
    
    def init_from_ckpt(self, path):
        ckpt = torch.load(path, map_location='cpu')
        missing, unexpected = self.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'Restoring from checkpoint {ckpt}.')
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')
    
    def predict_step(self, batch, batch_idx):
        batch_size = len(batch['fname'])
        image_shape = self.model.unet.config.sample_size
        samples = uncond_image_gen.pipeline_ddpm(
            self.model, self.sampler, batch_size, image_shape, num_inference_steps=20
        )
        samples = (samples + 1) / 2
        log_image_dict = {
            'image': samples,
            'fname': batch['fname'],
        }
        log_keys = ['image']
        self.log_image(log_image_dict, log_keys, batch_idx, mode='predict')
    
    @torch.no_grad()
    def log_image(self, batch, keys, batch_idx, mode):
        """
        Args:
            batch: dictionary, key: str -> value: tensor
        """
        dirname = os.path.join(self.logger.save_dir, 'log_images', mode)
        os.makedirs(dirname, exist_ok=True)
        for key in keys:
            image_t = batch[key].permute(0, 2, 3, 1).squeeze(-1)
            image_np = image_t.detach().cpu().numpy()
            for i in range(image_np.shape[0]):
                filename = f"{os.path.splitext(batch['fname'][i])[0]}_{key}.png"
                Image.fromarray(image_np[i]).save(os.path.join(dirname, filename))