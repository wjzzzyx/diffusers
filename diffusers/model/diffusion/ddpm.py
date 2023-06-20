from contextlib import contextmanager
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from diffusers.model.ema import LitEma
from diffusers.model.diffusion import noise_schedules
import utils


class DDPM(nn.Module):

    def __init__(self, model_config, schedule_config):
        super().__init__()
        self.model_config = model_config
        self.schedule_config = schedule_config
        self.model = utils.instantiate_from_config(model_config)
        schedule = noise_schedules.get_noise_schedule(schedule_config)
        for k, v in schedule:
            self.register_buffer(k, v)
        if model_config.parameterization == 'eps':
            lvlb_weights = (
                self.betas ** 2 / 
                (2 * self.posterior_variance * self.alphas * (1. - self.alphas_cumprod))
            )
        elif model_config.parameterization == 'x0':
            lvlb_weights = 0.5 * self.sqrt_alphas_cumprod / (2. * 1 - self.alphas_cumprod)    # is it correct???
        lvlb_weights[0] = lvlb_weights[1]    # ?
        assert(not torch.isnan(lvlb_weights).any())
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

        self.logvar = torch.full(fill_value=model_config.logvar_init, size=schedule_config.num_timesteps)
        if model_config.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
    
    def forward(self, x0):
        # t in [0, num_timesteps) -> x1...x999
        t = torch.randint(0, self.model_config.num_timesteps, (x0.size(0),), dtype=torch.long, device=self.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, noise)
        if self.model_config.parameterization == 'eps':
            target = noise
        elif self.model_config.parameterization == 'x0':
            target = x0
        
        pred = self.model(xt, t)

        if self.model_config.loss.loss_type == 'l1':
            loss = (target - pred).abs()
        elif self.model_config.loss.loss_type == 'l2':
            loss = F.mse_loss(target, pred, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])
        loss_simple = loss.mean()
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss = self.model_config.loss.lw_simple * loss_simple + self.model_config.loss.lw_elbo * loss_vlb

        logdict = {
            'loss_simple': loss_simple,
            'loss_vlb': loss_vlb,
            'loss': loss
        }
        return loss, logdict

    def q_sample(self, x0, noise):
        return self.sqrt_alphas_cumprod * x0 + self.sqrt_one_minus_alphas_cumprod * noise

    def p_sample(self, x, t, clip_denoised=True):
        # sample x_{t-1} from P(x_{t-1} | x_t)
        p_mean, p_var, p_logvar = self.p_mean_variance(x, t, clip_denoised=clip_denoised)
        # sample = mu + std * normal_sample
        normal_sample = torch.randn_like(x)
        sample = p_mean + torch.exp(0.5 * p_logvar) * normal_sample
        # if t == 0, return the mean
        sample = torch.where((t == 0).view(-1, *x.shape[1:]), p_mean, sample)
        return sample
    
    def p_mean_variance(self, x, t, clip_denoised):
        # return the mean and variance of P(x_{t-1} | x_t)
        pred = self.model(x, t)
        if self.model_config.parameterization == 'eps':
            x_recon = self.sqrt_recip_alphas_cumprod[t] * x - self.sqrt_recipm1_alphas_cumprod[t] * pred
        elif self.model_config.parameterization == 'x0':
            x_recon = pred
        
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        
        p_mean = self.posterior_mean_coef1[t] * x_recon + self.posterior_mean_coef2[t] * x
        p_var = self.posterior_variance[t]
        p_logvar = self.posterior_log_variance_clipped[t]
        return p_mean, p_var, p_logvar


class PLDDPM(pl.LightningModule):

    def __init__(self, model_config, schedule_config, sampler_config, ckpt_path=None, lr_scheduler_config=None, use_ema=True):
        super().__init__()
        self.model_config = model_config
        self.schedule_config = schedule_config
        self.sampler_config = sampler_config
        self.lrscheduler_config = lr_scheduler_config
        self.use_ema = use_ema
        self.model = DDPM(model_config, schedule_config)
        if use_ema:
            self.model_ema = LitEma(self.model)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        self.sampler = utils.instantiate_from_config(sampler_config)
    
    def init_from_ckpt(self, path):
        ckpt = torch.load(path, map_location='cpu')
        missing, unexpected = self.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'Restored from checkpoint {ckpt}')
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            # store current model parameters and use ema parameters instead
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f'{context}: Switched to EMA weights')
        try:
            yield None
        finally:
            # restore current model parameters
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f'{context}: Restored training weights')
    
    def training_step(self, batch, batch_idx):
        loss, logdict = self.model(batch['image'])
        logdict = {f'train/{k}': v for k, v in logdict.items()}
        self.log_dict(logdict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, logdict = self.model(batch['image'])
        with self.ema_scope():
            _, logdict_ema = self.model(batch['image'])
            logdict_ema = {f'{k}_ema': v for k, v in logdict_ema.items()}
        self.log_dict(logdict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(logdict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    def on_train_batch_end(self, batch, batch_idx):
        if self.use_ema:    # update parameter buffers in the EMA model
            self.model_ema(self.model)

    def configure_optimizers(self):
        params = list(self.model.parameters())
        if self.model_config.learn_logvar:
            params = params + [self.model.logvar]
        optimizer = torch.optim.AdamW(params, lr=self.model_config.learning_rate)
        return optimizer
    
    @torch.no_grad()
    def log_images(self, batch, N=8, log_every_t=200):
        x = batch['image'][:N]
        
        diffusion_row = list()
        for t in range(self.model_config.num_timesteps):
            if t % log_every_t == 0 or t == self.model_config.num_timesteps - 1:
                t = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
                noise = torch.randn_like(x)
                xt = self.model.q_sample(x, noise)
                diffusion_row.append(xt)
        diffusion_row = torch.stack(diffusion_row, 0).transpose(0, 1)
        

        with self.ema_scope():
            samples, denoise_row = self.sampler(
                self.model, batch_size=N, shape=x.shape[1:], return_intermediates=True
            )
            denoise_row = torch.stack(denoise_row, 0).transpose(0, 1)
            denoise_row = denoise_row.view(-1, denoise_row.shape[2:])
        
        logdict = {
            'inputs': x,
            'diffusion_row': torchvision.utils.make_grid(
                diffusion_row.view(-1, *diffusion_row.shape[2:]), nrow=diffusion_row.size(1)
            ),
            'samples': samples,
            'denoise_row': torchvision.utils.make_grid(
                denoise_row.view(-1, *denoise_row.shape[2:]), nrow=denoise_row.size(1)
            )
        }

        return logdict