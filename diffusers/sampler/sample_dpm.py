import torch
from typing import Callable, Sequence

import k_diffusion
from . import schedule


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
        cfg_scale: float = 1.0,
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
            denoiser = CFGDenoiser(denoiser, self.cfg_scale)
        return denoiser
    
    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
    
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

        if self.sampler == 'dpmpp_2m':
            sample = k_diffusion.sampling.sample_dpmpp_2m(
                denoiser, xi, sigmas=sigmas, extra_args=extra_args
            )
        elif self.sampler == 'dpmpp_sde':
            sample = k_diffusion.sampling.sample_dpmpp_sde(
                denoiser, xi, sigmas=sigmas, eta=1., s_noise=1., extra_args=extra_args
            )
        elif self.sampler == 'dpmpp_2m_sde':
            sample = k_diffusion.sampling.sample_dpmpp_2m_sde(
                denoiser, xi, sigmas=sigmas, eta=1., s_noise=1., solver_type=self.solver_type,
                extra_args=extra_args
            )
        elif self.sampler == 'dpmpp_2s_ancestral':
            sample = k_diffusion.sampling.sample_dpmpp_2s_ancestral(
                denoiser, xi, sigmas=sigmas, eta=1., s_noise=1., extra_args=extra_args
            )
        elif self.sampler == 'dpmpp_3m_sde':
            sample = k_diffusion.sampling.sample_dpmpp_3m_sde(
                denoiser, xi, sigmas=sigmas, eta=1., s_noise=1., extra_args=extra_args
            )
        elif self.sampler == 'dpm_2':
            sample = k_diffusion.sampling.sample_dpm_2(
                denoiser, xi, sigmas=sigmas, s_churn=0., s_tmin=0., s_tmax=0., s_noise=1.,
                extra_args=extra_args
            )
        elif self.sampler == 'dpm_2_ancestral':
            sample = k_diffusion.sampling.sample_dpm_2_ancestral(
                denoiser, xi, sigmas=sigmas, eta=1., s_noise=1., extra_args=extra_args
            )
        elif self.sampler == 'dpm_fast':
            # is eta correct?
            sample = k_diffusion.sampling.sample_dpm_fast(
                denoiser, xi, sigma_min=denoiser.sigmas[0].item(), sigma_max=denoiser.sigmas[-1].item(), n=self.num_inference_steps,
                eta=1., s_noise=1., extra_args=extra_args
            )
        elif self.sampler == 'dpm_adaptive':
            # is eta correct?
            sample = k_diffusion.sampling.sample_dpm_adaptive(
                denoiser, xi, sigma_min=denoiser.sigmas[0].item(), sigma_max=denoiser.sigmas[-1].item(), eta=1., extra_args=extra_args
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
                denoiser, xi, sigmas=sigma_sched, s_churn=0., s_tmin=0., s_tmax=0., s_noise=1.,
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
