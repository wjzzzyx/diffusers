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
        denoising_strength: float = 1.0,
    ):
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.betas = schedule.get_betas(beta_start, beta_end, beta_schedule, num_train_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.denoising_strength = denoising_strength

        self.set_timesteps()
    
    def set_timesteps(self):
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
        image_shape: Sequence = None,
        image: torch.Tensor = None,
        denoiser_args: dict = {},
        generator=None
    ):
        """
        Args:
            denoiser: takes charge of one step denoising, return denoised image
        """
        num_steps = int(round(self.num_inference_steps * self.denoising_strength))
        start_t = self.timesteps[-num_steps]
        timesteps = self.timesteps[-num_steps:]
        
        if image is None:
            x = torch.randn((batch_size, *image_shape), generator=generator, device=denoiser.inner_model.device)
        else:
            noise = torch.randn(image.shape, generator=generator, device=image.device)
            x = torch.sqrt(self.alphas_cumprod[start_t]) * image + torch.sqrt(1 - self.alphas_cumprod[start_t]) * noise

        for t in timesteps:
            time_t = torch.full((batch_size,), t, dtype=torch.int64, device=x.device)
            denoised = denoiser(x, time_t, **denoiser_args)
            x = self.step(denoised, t, x, generator=generator)
        return x


def dynamic_threshold(sample: torch.Tensor):
    "From paper https://arxiv.org/abs/2205.11487."
    batch_size, c, h, w = sample.size()
    sample = sample.reshape(batch_size, -1)
    s = torch.quantile(sample.abs(), 0.995, dim=1, keepdim=True)
    s = torch.clamp(s, min=1, max=1)    # ?
    sample = torch.clamp(sample, -s, s) / s    # ?
    sample = sample.reshape(batch_size, c, h, w)
    return sample