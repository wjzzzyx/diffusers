import math
import torch


def get_betas(beta_start, beta_end, beta_schedule, num_steps):
    if beta_schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_steps)
    elif beta_schedule == 'sqrt_linear':
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps) ** 2
    elif beta_schedule == 'squaredcos_cap_v2':    # glide cosine schedule
        steps = range(num_steps + 1) / num_steps
        alphas_cumprod = torch.cos(math.pi / 2 * (steps + 0.008) / 1.008) ** 2
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clamp(betas, max=0.999)
    elif beta_schedule == 'sigmoid':    # geodiff sigmoid schedule
        betas = torch.linspace(-6, 6, num_steps)
        betas = betas.sigmoid() * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f'{beta_schedule} is not implemented.')
    return betas

def rescale_for_zero_terminal_snr():
    ...