import collections
import lightning
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn

from . import ema
from .modules import unet_2d
import utils


class DDPM(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.unet = unet_2d.UNet2D(**model_config.unet)
        self.prediction_type = model_config.prediction_type
        if 'pretrained' in model_config:
            ckpt = torch.load(model_config.pretrained, map_location='cpu')
            self._convert_deprecated_attention_blocks(ckpt)
            self.unet.load_state_dict(ckpt)
    
    @property
    def device(self):
        return self.unet.conv_in.weight.device
    
    def forward(self, xt, t):
        output = self.unet(xt, t)
        # epsilon, x0
        # if self.prediction_type == 'epsilon':
        #     epsilon = output
        #     sample = (xt - torch.sqrt(1 - alpha_cumprod_t) * output) / torch.sqrt(alpha_cumprod_t)
        # elif self.prediction_type == 'sample':
        #     sample = output
        #     epsilon = (xt - torch.sqrt(alpha_cumprod_t) * output) / torch.sqrt(1 - alpha_cumprod_t)
        # elif self.prediction_type == 'v_prediction':
        #     raise NotImplementedError('v_prediction is not implemented for DDPM.')

        # return {
        #     'epsilon': epsilon,
        #     'sample': sample,
        # }
        return output

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


class PLBase(lightning.LightningModule):
    def __init__(self, model_config, sampler_config, ema_config=None, optimizer_config=None, ckpt_path=None):
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
        
        self.sampler = utils.instantiate_from_config(sampler_config)

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
        in_channels = self.model.unet.config.in_channels
        image_size = self.model.unet.config.sample_size
        samples = self.sampler.sample(self.model, batch_size, [in_channels, image_size, image_size])
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
        dirname = os.path.join(self.trainer.default_root_dir, 'log_images', mode)
        os.makedirs(dirname, exist_ok=True)
        for key in keys:
            image_t = batch[key].permute(0, 2, 3, 1).squeeze(-1)
            image_np = image_t.detach().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            for i in range(image_np.shape[0]):
                filename = f"{os.path.splitext(batch['fname'][i])[0]}_{key}.png"
                Image.fromarray(image_np[i]).save(os.path.join(dirname, filename))