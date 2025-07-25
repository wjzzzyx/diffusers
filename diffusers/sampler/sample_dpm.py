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
        num_train_timesteps: int,
        num_inference_timesteps: int,
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
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.initial_noise_multiplier = initial_noise_multiplier
        self.denoising_strength = denoising_strength
        self.betas = schedule.get_betas(beta_start, beta_end, beta_schedule, num_train_timesteps).cuda()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)
        self.log_sigmas = self.sigmas.log()

    def get_inference_sigmas(self):
        steps = self.num_inference_timesteps + 1 if self.discard_next_to_last_sigma else self.num_inference_timesteps

        if self.scheduler == 'karras':
            sigmas = k_diffusion.sampling.get_sigmas_karras(
                n=steps, sigma_min=self.sigmas[0].item(), sigma_max=self.sigmas[-1].item()
            )
        elif self.scheduler == 'exponential':
            sigmas = k_diffusion.sampling.get_sigmas_exponential(
                n=steps, sigma_min=self.sigmas[0].item(), sigma_max=self.sigmas[-1].item()
            )
        else:
            t = torch.linspace(self.num_train_timesteps - 1, 0, steps)
            sigmas = torch.cat([self.t_to_sigma(t), t.new_zeros([1])])
        
        if self.discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        
        return sigmas
    
    def sigma_to_t(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
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
    
    def sample(
        self,
        denoiser,
        noise: torch.Tensor,
        image: torch.Tensor = None,
        denoiser_args: dict = {},
        generator = None
    ) -> torch.Tensor:
        t_enc = min(int(self.denoising_strength * self.num_inference_timesteps), self.num_inference_timesteps - 1)
        sigmas = self.get_inference_sigmas().cuda()
        sigmas = sigmas[self.num_inference_timesteps - t_enc - 1:]

        if image is not None:
            xi = image + noise * sigmas[0]
        else:
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
