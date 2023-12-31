import torch
import torch.nn as nn


class DiscreteTimestepsDenoiser(nn.Module):
    def __init__(self, model, alphas_cumprod):
        super().__init__()
        self.inner_model = model
        self.alphas_cumprod = alphas_cumprod
    
    def forward(self, input, timesteps, **kwargs):
        return self.inner_model(input, timesteps, **kwargs)


class DiscreteTimestepsVDenoiser(nn.Module):
    def __init__(self, model, alphas_cumprod):
        self.inner_model = model
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    def predict_eps_from_z_and_v(self, x_t, t, v):
        return self.sqrt_alphas_cumprod[t, None, None, None] * v + self.sqrt_one_minus_alphas_cumprod[t, None, None, None] * x_t

    def forward(self, input, timesteps, **kwargs):
        out = self.inner_model(input, timesteps, **kwargs)
        e_t = self.predict_eps_from_z_and_v(input, timesteps, out)
        return e_t


class CFGDenoiser():
    def __init__(self, denoiser, cfg_scale):
        self.denoiser = denoiser
        self.cfg_scale = cfg_scale

    def __call__(self, x, sigma, cond_pos_prompt, cond_neg_prompt, cond_imaging, image, mask):
        x_in = torch.cat([x, x])
        sigma_in = torch.cat([sigma, sigma])
        cond_imaging_in = torch.cat([cond_imaging, cond_imaging])
        cond_prompt_in = torch.cat([cond_pos_prompt, cond_neg_prompt])

        x_out = self.denoiser(x_in, sigma_in, cond={'c_crossattn': [cond_prompt_in], 'c_concat': [cond_imaging_in]})

        denoised_pos = x_out[:cond_pos_prompt.shape[0]]
        denoised_neg = x_out[-cond_neg_prompt.shape[0]:]
        denoised = denoised_neg + self.cfg_scale * (denoised_pos - denoised_neg)

        denoised = denoised * mask + image * (1 - mask)
        return denoised
