import math
import torch
from typing import Dict, Sequence

from . import schedule


class DDPMSampler():
    def __init__(
        self,
        num_train_steps: int,
        num_inference_steps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
    ):
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.betas = schedule.get_betas(beta_start, beta_end, beta_schedule, num_train_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.set_timesteps(num_inference_steps)
    
    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        # leading spaceing
        self.timesteps = range(0, self.num_train_steps, self.num_train_steps // self.num_inference_steps)
        self.timesteps = self.timesteps[::-1]

    def step(self, denoised: torch.Tensor, t: int, sample: torch.Tensor, generator = None):
        """
        Args:
            output: model output for the current denoising step
            t: timestep of sample
            sample: sample used as model input to predict the current output
        """
        t_next = t - self.num_train_steps // self.num_inference_steps
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=sample.device)
        alpha_t = alpha_cumprod_t / alpha_cumprod_t_next
        
        pred_x0 = denoised
        # pred_x0 = dynamic_threshold(pred_x0)
        pred_x0 = torch.clamp(pred_x0, -1., 1.)

        coeff1 = torch.sqrt(alpha_cumprod_t_next) * (1 - alpha_t) / (1 - alpha_cumprod_t)
        coeff2 = torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t)
        pred_cur_mean = coeff1 * pred_x0 + coeff2 * sample

        if t > 0:
            # variance type fixed_small
            variance = (1 - alpha_cumprod_t_next) * (1 - alpha_t) / (1 - alpha_cumprod_t)
            variance = torch.clamp(variance, min=1e-20)
            variance = torch.sqrt(variance) * torch.randn(sample.shape, generator=generator, device=sample.device)
        else:
            variance = 0
        sample_next = pred_cur_mean + variance

        return sample_next
    
    @torch.no_grad()
    def sample(
        self,
        denoiser,
        batch_size: int,
        image_shape: Sequence,
        denoiser_args: dict,
        generator=None
    ):
        """
        Args:
            denoiser: takes charge of one step denoising, return denoised image
        """
        image = torch.randn((batch_size, *image_shape), generator=generator, device=denoiser.inner_model.device)

        for t in self.timesteps:
            time_t = torch.full((batch_size,), t, dtype=torch.int64, device=image.device)
            denoised = denoiser(image, time_t, **denoiser_args)
            # if model.prediction_type == 'epsilon':
            #     epsilon = output
            #     sample = (image - torch.sqrt(1 - self.alphas_cumprod[t]) * output) / torch.sqrt(self.alphas_cumprod[t])
            # elif model.prediction_type == 'sample':
            #     sample = output
            #     epsilon = (image - torch.sqrt(self.alphas_cumprod[t]) * output) / torch.sqrt(1 - self.alphas_cumprod[t])
            # elif model.prediction_type == 'v_prediction':
            #     raise NotImplementedError('v_prediction is not implemented for DDPM.')
            image = self.step(denoised, t, image, generator=generator)
        return image


def dynamic_threshold(sample: torch.Tensor):
    "From paper https://arxiv.org/abs/2205.11487."
    batch_size, c, h, w = sample.size()
    sample = sample.reshape(batch_size, -1)
    s = torch.quantile(sample.abs(), 0.995, dim=1, keepdim=True)
    s = torch.clamp(s, min=1, max=1)    # ?
    sample = torch.clamp(sample, -s, s) / s    # ?
    sample = sample.reshape(batch_size, c, h, w)
    return sample