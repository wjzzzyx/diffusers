import math
import torch
from typing import Dict

from . import schedule


class DDPMSampler():
    def __init__(
        self,
        num_train_steps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
    ):
        self.num_train_steps = num_train_steps
        self.betas = schedule.get_betas(beta_start, beta_end, beta_schedule, num_train_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
    
    def step(self, output: Dict[str, torch.Tensor], t: int, sample: torch.Tensor, generator = None):
        """
        Args:
            output: model output for the current denoising step
            t: timestep of prev_sample
            prev_sample: previous step sample used as model input to predict the current output
        """
        t_next = t - self.num_train_steps // self.num_inference_steps
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else 1.0
        alpha_t = alpha_cumprod_t / alpha_cumprod_t_next
        
        pred_x0 = output['sample']
        # pred_x0 = dynamic_threshold(pred_x0)
        pred_x0 = torch.clamp(pred_x0, -1., 1.)

        coeff1 = torch.sqrt(alpha_cumprod_t_next) * (1 - alpha_t) / (1 - alpha_cumprod_t)
        coeff2 = torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t)
        pred_cur_mean = coeff1 * pred_x0 + coeff2 * sample

        if t > 0:
            variance = (1 - alpha_cumprod_t_next) * (1 - alpha_t) / (1 - alpha_cumprod_t)
            variance = variance * torch.randn(sample.shape, generator=generator)
        else:
            variance = 0
        sample_next = pred_cur_mean + variance

        return sample_next


def dynamic_threshold(sample: torch.Tensor):
    "From paper https://arxiv.org/abs/2205.11487."
    batch_size, c, h, w = sample.size()
    sample = sample.reshape(batch_size, -1)
    s = torch.quantile(sample.abs(), 0.995, dim=1, keepdim=True)
    s = torch.clamp(s, min=1, max=1)    # ?
    sample = torch.clamp(sample, -s, s) / s    # ?
    sample = sample.reshape(batch_size, c, h, w)
    return sample