import torch
import torch.nn as nn


class DiscreteTimeEpsDenoiser():
    # from stable-diffusion-webui
    def __init__(self, model, alphas_cumprod):
        self.inner_model = model
        self.alphas_cumprod = alphas_cumprod.to(model.device)
    
    def __call__(self, xt, t, **kwargs):
        # xt: shape (b, c, h, w)
        # t: shape (b)
        eps = self.inner_model.forward_diffusion_model(xt, t, **kwargs)
        return self.predict_x0(xt, eps, t)
    
    def predict_x0(self, xt, eps, t):
        alphas_cumprod_t = self.alphas_cumprod[t][(...,) + (None,) * 3]    # shape (b, 1, 1, 1)
        denoised = (xt - torch.sqrt(1 - alphas_cumprod_t) * eps) / torch.sqrt(alphas_cumprod_t)
        return denoised


class DiscreteTimeVDenoiser():
    # from stable-diffusion-webui
    def __init__(self, model, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
        self.inner_model = model
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
    
    def predict_eps_from_z_and_v(self, x_t, t, v):
        return self.sqrt_alphas_cumprod[t, None, None, None] * v + self.sqrt_one_minus_alphas_cumprod[t, None, None, None] * x_t

    def __call__(self, xt, t, **kwargs):
        # TODO should return denoised
        out = self.inner_model.forward_diffusion_model(xt, t, **kwargs)
        e_t = self.predict_eps_from_z_and_v(xt, t, out)
        return e_t


class DiscreteTimeCFGDenoiser():
    def __init__(self, denoiser, cfg_scale):
        self.denoiser = denoiser
        self.cfg_scale = cfg_scale
    
    def __getattr__(self, name):
        return getattr(self.denoiser, name)

    def __call__(self, xt, t, **kwargs):
        """
        Args:
            xt: (B, C, H, W) input image or latent tensor
            t: (B) time step
            cond_pos_prompt: (B, L, C) positive prompt embedding
            cond_neg_prompt: (B, L, C) negative prompt embedding
            cond_imaging: conditional imaging input for models like inpainting
        """
        x_in = torch.cat([xt, xt])
        t_in = torch.cat([t, t])
        cond_pos_prompt = kwargs.pop('cond_pos_prompt')
        cond_neg_prompt = kwargs.pop('cond_neg_prompt')
        kwargs['cond_prompt'] = torch.cat([cond_pos_prompt, cond_neg_prompt])
        if 'cond_imaging' in kwargs:
            cond_imaging = kwargs.pop('cond_imaging')
            kwargs['cond_imaging'] = torch.cat([cond_imaging, cond_imaging])
        
        x_out = self.denoiser(x_in, t_in, **kwargs)

        denoised_pos = x_out[:cond_pos_prompt.shape[0]]
        denoised_neg = x_out[-cond_neg_prompt.shape[0]:]
        denoised = denoised_pos + self.cfg_scale * (denoised_pos - denoised_neg)

        return denoised


class MaskedDenoiser():
    def __init__(self, denoiser):
        self.denoiser = denoiser
    
    def __getattr__(self, name):
        return getattr(self.denoiser, name)
    
    def __call__(self, image, mask, *args, **kwargs):
        """
        Args:
            image: original image for models like inpainting
            mask: in {0, 1} highlight painting area for models like inpainting
        """
        denoised = self.denoiser(*args, **kwargs)
        denoised = denoised * mask + image * (1 - mask)
        return denoised


class KarrasDenoiser():
    def __init__(self, alphas_cumprod, quantize):
        self.alphas_cumprod = alphas_cumprod
        self.sigmas = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        self.log_sigmas = self.sigmas.log()
        self.quantize = quantize
    
    @property
    def sigma_min(self):
        return self.sigmas[0].item()

    @property
    def sigma_max(self):
        return self.sigmas[-1].item()

    def get_sigmas(self, n=None):
        def append_zero(x):
            return torch.cat([x, x.new_zeros([1])])
        
        if n is None:
            return append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if self.quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()


class KarrasEpsDenoiser(KarrasDenoiser):
    """A wrapper for my StabilityAI diffusion models."""
    def __init__(self, model, alphas_cumprod):
        super().__init__(alphas_cumprod, quantize=False)
        self.inner_model = model
        self.sigma_data = 1.

    def __call__(self, xt, sigma, **kwargs):
        self.sigmas = self.sigmas.to(xt.device)
        self.log_sigmas = self.log_sigmas.to(xt.device)
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        eps = self.inner_model.forward_diffusion_model(
            xt * c_in[..., None, None, None], self.sigma_to_t(sigma), **kwargs
        )
        return xt + eps * c_out[..., None, None, None]


class KarrasVDenoiser(KarrasDenoiser):
    def __init__(self, model, alphas_cumprod):
        super().__init__(alphas_cumprod, quantize=False)
        self.inner_model = model
        self.sigma_data = 1.
    
    def __call__(self, xt, sigma, **kwargs):
        self.sigmas = self.sigmas.to(xt.device)
        self.log_sigmas = self.log_sigmas.to(xt.device)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        pred_v = self.inner_model.forward_diffusion_model(
            xt * c_in[(...,) + (None,) * 3], self.sigma_to_t(sigma), **kwargs
        )
        return xt * c_skip + pred_v * c_out[(...,) + (None,) * 3]


class KarrasCFGDenoiser():
    def __init__(self, denoiser, cfg_scale):
        self.denoiser = denoiser
        self.cfg_scale = cfg_scale
    
    def __getattr__(self, name):
        return getattr(self.denoiser, name)

    def __call__(self, xt, sigma, **kwargs):
        # x_in = torch.cat([xt, xt])
        # sigma_in = torch.cat([sigma, sigma])
        cond_pos_prompt = kwargs.pop('cond_pos_prompt')
        cond_neg_prompt = kwargs.pop('cond_neg_prompt')
        # kwargs['cond_prompt'] = torch.cat([cond_pos_prompt, cond_neg_prompt])
        # if 'cond_imaging' in kwargs:
        #     cond_imaging = kwargs.pop('cond_imaging')
        #     kwargs['cond_imaging'] = torch.cat([cond_imaging, cond_imaging])

        # x_out = self.denoiser(x_in, sigma_in, **kwargs)

        # denoised_pos = x_out[:cond_pos_prompt.shape[0]]
        # denoised_neg = x_out[-cond_neg_prompt.shape[0]:]
        denoised_pos = self.denoiser(xt, sigma, cond_prompt=cond_pos_prompt, **kwargs)
        denoised_neg = self.denoiser(xt, sigma, cond_prompt=cond_neg_prompt, **kwargs)
        
        denoised = denoised_neg + self.cfg_scale * (denoised_pos - denoised_neg)

        return denoised


class CFGDenoiser():
    def __init__(self, denoiser, cfg_scale):
        self.denoiser = denoiser
        self.cfg_scale = cfg_scale
    
    def __call__(self, x, t, cond_pos_prompt, cond_neg_prompt, cond_imaging=None, image=None, mask=None):
        """
        Args:
            x: (B, C, H, W) input image or latent tensor
            cond_pos_prompt: (B, L, C) positive prompt embedding
            cond_neg_prompt: (B, L, C) negative prompt embedding
            cond_imaging: conditional imaging input for models like inpainting
            image: original image for models like inpainting
            mask: in {0, 1} highlight painting area for models like inpainting
        """
        x_in = torch.cat([x, x])
        t_in = torch.cat([t, t])
        cond_prompt_in = torch.cat([cond_pos_prompt, cond_neg_prompt])
        if cond_imaging is not None:
            cond_imaging_in = torch.cat([cond_imaging, cond_imaging])
        
        x_out = self.denoiser(x_in, t_in, encoder_hidden_states=cond_prompt_in)

        denoised_pos = x_out[:cond_pos_prompt.shape[0]]
        denoised_neg = x_out[-cond_neg_prompt.shape[0]:]
        denoised = denoised_neg + self.cfg_scale * (denoised_pos - denoised_neg)

        return denoised
