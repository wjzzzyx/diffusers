import torch
from typing import Callable, Sequence

from diffusers import k_diffusion
from diffusers.sampler import schedule


class DPMSampler():
    def __init__(
        self,
        sampler: str,
        scheduler: str,
        solver_type: str,
        discard_next_to_last_sigma: bool,
        second_order: bool,
        uses_ensd: bool,
        num_train_steps: int,
        num_inference_steps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        initial_noise_multiplier: float = 1.0,
        denoising_strength: float = 1.0,
    ):
        """
        Args:
            sampler: choises [dpmpp_2m, dpmpp_sde, dpmpp_2m_sde, dpmpp_2s_ancestral, dpmpp_3m_sde, dpm_2, dpm_2_ancestral, dpm_fast, dpm_adaptive]
            scheduler: choises [karras, exponential]
            solver_type: choices [midpoint, heun]
        """
        self.sampler = sampler
        self.scheduler = scheduler
        self.solver_type = solver_type
        self.discard_next_to_last_sigma = discard_next_to_last_sigma
        self.second_order = second_order
        self.uses_ensd = uses_ensd
        self.num_inference_steps = num_inference_steps
        self.betas = schedule.get_betas(beta_start, beta_end, beta_schedule, num_train_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.initial_noise_multiplier = initial_noise_multiplier
        self.denoising_strength = denoising_strength

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
    
    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
    
    def sample(
        self,
        denoiser,
        batch_size: int,
        image_shape: Sequence = None,
        image: torch.Tensor = None,
        denoiser_args: dict = {},
        generator = None
    ) -> torch.Tensor:
        t_enc = min(int(self.denoising_strength * self.num_inference_steps), self.num_inference_steps - 1)
        sigmas = self.get_sigmas(denoiser).to(denoiser.inner_model.device)
        sigmas = sigmas[self.num_inference_steps - t_enc - 1:]

        if image is not None:
            noise = torch.randn(image.shape, device=image.device, generator=generator)
            xi = image + noise * sigmas[0]
        else:
            noise = torch.randn((batch_size, *image_shape), generator=generator, device=denoiser.inner_model.device)
            xi = noise * sigmas[0]

        if self.sampler == 'dpmpp_2m':
            sample = k_diffusion.sampling.sample_dpmpp_2m(
                denoiser, xi, sigmas=sigmas, extra_args=denoiser_args
            )
        elif self.sampler == 'dpmpp_sde':
            sample = k_diffusion.sampling.sample_dpmpp_sde(
                denoiser, xi, sigmas=sigmas, eta=1., s_noise=1., extra_args=denoiser_args
            )
        elif self.sampler == 'dpmpp_2m_sde':
            sample = k_diffusion.sampling.sample_dpmpp_2m_sde(
                denoiser, xi, sigmas=sigmas, eta=1., s_noise=1., solver_type=self.solver_type,
                extra_args=denoiser_args
            )
        elif self.sampler == 'dpmpp_2s_ancestral':
            sample = k_diffusion.sampling.sample_dpmpp_2s_ancestral(
                denoiser, xi, sigmas=sigmas, eta=1., s_noise=1., extra_args=denoiser_args
            )
        elif self.sampler == 'dpmpp_3m_sde':
            sample = k_diffusion.sampling.sample_dpmpp_3m_sde(
                denoiser, xi, sigmas=sigmas, eta=1., s_noise=1., extra_args=denoiser_args
            )
        elif self.sampler == 'dpm_2':
            sample = k_diffusion.sampling.sample_dpm_2(
                denoiser, xi, sigmas=sigmas, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                extra_args=denoiser_args
            )
        elif self.sampler == 'dpm_2_ancestral':
            sample = k_diffusion.sampling.sample_dpm_2_ancestral(
                denoiser, xi, sigmas=sigmas, eta=1., s_noise=1., extra_args=denoiser_args
            )
        elif self.sampler == 'dpm_fast':
            # is eta correct?
            sample = k_diffusion.sampling.sample_dpm_fast(
                denoiser, xi, sigma_min=sigmas[-2].item(), sigma_max=sigmas[0].item(), n=len(sigmas) - 1,
                eta=1., s_noise=1., extra_args=denoiser_args
            )
        elif self.sampler == 'dpm_adaptive':
            # is eta correct?
            sample = k_diffusion.sampling.sample_dpm_adaptive(
                denoiser, xi, sigma_min=sigmas[-2].item(), sigma_max=sigmas[0].item(), eta=1., extra_args=denoiser_args
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

        # TODO second order?
        if self.sampler == 'dpmpp_2m':
            sample = k_diffusion.sampling.sample_dpmpp_2m(
                denoiser, xi, sigmas=sigma_sched, extra_args=extra_args
            )
        elif self.sampler == 'dpmpp_sde':
            sample = k_diffusion.sampling.sample_dpmpp_sde(
                denoiser, xi, sigmas=sigma_sched, eta=1., s_noise=1., extra_args=extra_args
            )
        elif self.sampler == 'dpmpp_2m_sde':
            sample = k_diffusion.sampling.sample_dpmpp_2m_sde(
                denoiser, xi, sigmas=sigma_sched, eta=1., s_noise=1., solver_type=self.solver_type,
                extra_args=extra_args
            )
        elif self.sampler == 'dpmpp_2s_ancestral':
            sample = k_diffusion.sampling.sample_dpmpp_2s_ancestral(
                denoiser, xi, sigmas=sigma_sched, eta=1., s_noise=1., extra_args=extra_args
            )
        elif self.sampler == 'dpmpp_3m_sde':
            sample = k_diffusion.sampling.sample_dpmpp_3m_sde(
                denoiser, xi, sigmas=sigma_sched, eta=1., s_noise=1., extra_args=extra_args
            )
        elif self.sampler == 'dpm_2':
            sample = k_diffusion.sampling.sample_dpm_2(
                denoiser, xi, sigmas=sigma_sched, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                extra_args=extra_args
            )
        elif self.sampler == 'dpm_2_ancestral':
            sample = k_diffusion.sampling.sample_dpm_2_ancestral(
                denoiser, xi, sigmas=sigma_sched, eta=1., s_noise=1., extra_args=extra_args
            )
        elif self.sampler == 'dpm_fast':
            # is eta correct?
            sample = k_diffusion.sampling.sample_dpm_fast(
                denoiser, xi, sigma_min=sigma_sched[-2], sigma_max=sigma_sched[0], n=len(sigma_sched) - 1,
                eta=1., s_noise=1., extra_args=extra_args
            )
        elif self.sampler == 'dpm_adaptive':
            # is eta correct?
            sample = k_diffusion.sampling.sample_dpm_adaptive(
                denoiser, xi, sigma_min=sigma_sched[-2], sigma_max=sigma_sched[0], eta=1., extra_args=extra_args
            )
        
        sample = sample * mask + image * (1 - mask)
        return sample
