import torch
from typing import Sequence

from diffusers import k_diffusion
from diffusers.sampler import schedule
from diffusers.sampler.denoiser import K_CFGDenoiser


class EulerSampler():
    def __init__(
        self,
        sampler: str,
        scheduler: str,
        discard_next_to_last_sigma: bool,
        second_order: bool,
        num_train_steps: int,
        num_inference_steps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        initial_noise_multiplier: float = 1.0,
        denoising_strength: float = 1.0,
        cfg_scale: float = 1.0,
    ):
        self.sampler = sampler
        self.scheduler = scheduler
        self.discard_next_to_last_sigma = discard_next_to_last_sigma
        self.second_order = second_order
        self.num_inference_steps = num_inference_steps
        self.betas = schedule.get_betas(beta_start, beta_end, beta_schedule, num_train_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.initial_noise_multiplier = initial_noise_multiplier
        self.denoising_strength = denoising_strength
        self.cfg_scale = cfg_scale
    
    def get_sigmas(self, denoiser):
        steps = self.num_inference_steps + 1 if self.discard_next_to_last_sigma else self.num_inference_steps

        if self.scheduler == 'karras':
            sigma_min, sigma_max = denoiser.sigmas[0].item(), denoiser.sigmas[-1].item()
            sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max)
        elif self.scheduler == 'exponential':
            sigma_min, sigma_max = denoiser.sigmas[0].item(), denoiser.sigmas[-1].item()
            sigmas = k_diffusion.sampling.get_sigmas_exponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max)
        else:
            sigmas = denoiser.get_sigmas(steps)
        
        if self.discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        
        return sigmas
    
    def get_denoiser(self, model):
        if model.prediction_type == 'epsilon':
            denoiser = k_diffusion.external.DiscreteEpsDDPMDenoiser(model, self.alphas_cumprod, quantize=False)
        elif model.prediction_type == 'v_prediction':
            denoiser = k_diffusion.external.DiscreteVDDPMDenoiser(model, self.alphas_cumprod, quantize=False)
        else:
            raise NotImplementedError()
        if self.cfg_scale > 1:
            denoiser = K_CFGDenoiser(denoiser, self.cfg_scale)
        return denoiser
    
    def sample(
        self,
        model,
        batch_size: int,
        image_shape: Sequence,
        cond_pos_prompt: torch.Tensor = None,
        cond_neg_prompt: torch.Tensor = None,
        generator = None
    ) -> torch.Tensor:
        denoiser = self.get_denoiser(model).to(model.device)
        sigmas = self.get_sigmas(denoiser).to(model.device)

        noise = torch.randn((batch_size, *image_shape), generator=generator, device=model.device)
        xi = noise * sigmas[0]

        extra_args = dict()
        if cond_pos_prompt is not None:
            extra_args['cond_pos_prompt'] = cond_pos_prompt
        if cond_neg_prompt is not None:
            extra_args['cond_neg_prompt'] = cond_neg_prompt
        
        sample = k_diffusion.sampling.sample_heun(
            denoiser, xi, sigmas, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., extra_args=extra_args
        )
        
        return sample

    def sample_img2img(
        self,
        model,
        image: torch.Tensor,
        cond_pos_prompt: torch.Tensor = None,
        cond_neg_prompt: torch.Tensor = None,
        cond_imaging: torch.Tensor = None,
        mask: torch.Tensor = None,
        generator = None,
    ) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W) original image or latent image
            noise: (B, C, H, W) noise of same size as image
            cond_pos_prompt: (B, L, embed_dim)
            cond_neg_prompt: (B, L, embed_dim)
            cond_imaging: (B, ?, H, W) imaging conditioning, like inpainting
            mask: (B, H, W) in {0, 1} optional mask for img2img generation.
        """
        denoiser = self.get_denoiser(model)
        t_enc = min(int(self.denoising_strength * self.num_inference_steps), self.num_inference_steps - 1)
        sigmas = self.get_sigmas(denoiser).to(model.device)
        sigma_sched = sigmas[self.num_inference_steps - t_enc - 1:]

        noise = torch.randn(image.shape, device=image.device, generator=generator)
        noise *= self.initial_noise_multiplier

        xi = image + noise * sigma_sched[0]
        # TODO extra noise

        extra_args = {
            'cond_pos_prompt': cond_pos_prompt,
            'cond_neg_prompt': cond_neg_prompt,
            'cond_imaging': cond_imaging,
            'image': image,
            'mask': mask,
        }

        sample = k_diffusion.sampling.sample_heun(
            denoiser, xi, sigma_sched, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., extra_args=extra_args
        )
        
        sample = sample * mask + image * (1 - mask)
        return sample