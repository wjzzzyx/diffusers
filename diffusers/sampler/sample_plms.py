import torch
import tqdm
from typing import Sequence

from diffusers.sampler.denoiser import DiscreteTimestepsDenoiser, DiscreteTimestepsVDenoiser, CFGDenoiser
from diffusers.sampler import schedule


class PLMSSampler():
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
    
    def get_denoiser(self, model):
        if model.prediction_type == 'epsilon':
            denoiser = DiscreteTimestepsDenoiser(model, self.alphas_cumprod)
        elif model.prediction_type == 'v_prediction':
            denoiser = DiscreteTimestepsVDenoiser(model, self.alphas_cumprod)
        else:
            raise NotImplementedError()
        if self.cfg_scale > 1:
            denoiser = CFGDenoiser(denoiser, self.cfg_scale)
        return denoiser

    def get_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        timesteps = range(0, self.num_train_steps, self.num_train_steps // self.num_inference_steps)
        return timesteps
    
    def sample(
        self,
        model,
        batch_size: int,
        image_shape: Sequence,
        cond_pos_prompt: torch.Tensor = None,
        cond_neg_prompt: torch.Tensor = None,
        generator = None
    ) -> torch.Tensor:
        assert(model.prediction_type in ['epsilon', 'v_prediction'])
        denoiser = self.get_denoiser()
        timesteps = self.get_timesteps()

        noise = torch.randn((batch_size, *image_shape), generator=generator, device=model.device)
        xi = noise

        extra_args = dict()
        if cond_pos_prompt is not None:
            extra_args['cond_pos_prompt'] = cond_pos_prompt
        if cond_neg_prompt is not None:
            extra_args['cond_neg_prompt'] = cond_neg_prompt
        
        sample = sample_plms(denoiser, xi, timesteps, extra_args=extra_args)
        return sample


@torch.no_grad()
def sample_plms(model, x, timesteps, extra_args=None, callback=None, disable=None):
    alphas_cumprod = model.alphas_cumprod
    alphas = alphas_cumprod[timesteps]
    alphas_prev = alphas_cumprod[F.pad(timesteps[:-1], pad=(1, 0))]
    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    s_x = x.new_ones((x.shape[0], 1, 1, 1))
    old_eps = []

    def get_x_prev_and_pred_x0(e_t, index):
        # select parameters corresponding to the currently considered timestep
        a_t = alphas[index].item() * s_x
        a_prev = alphas_prev[index].item() * s_x
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].item() * s_x

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1. - a_prev).sqrt() * e_t
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        return x_prev, pred_x0

    for i in tqdm.trange(len(timesteps) - 1, disable=disable):
        index = len(timesteps) - 1 - i
        ts = timesteps[index].item() * s_in
        t_next = timesteps[max(index - 1, 0)].item() * s_in

        e_t = model(x, ts, **extra_args)

        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = model(x_prev, t_next, **extra_args)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        else:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        old_eps.append(e_t)
        if len(old_eps) >= 4:
            old_eps.pop(0)

        x = x_prev

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    return x