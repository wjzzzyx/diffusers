import torch


def get_noise_schedule(schedule_config):
    if schedule_config.type == 'linear':
        betas = torch.linspace(
            schedule_config.linear_start ** 0.5,
            schedule_config.linear_end ** 0.5,
            schedule_config.num_timesteps,
            dtype=torch.float64
        )
        betas = betas ** 2
    elif schedule_config.type == 'cosine':
        timesteps = torch.arange(schedule_config.num_timesteps + 1, dtype=torch.float64)
        alphas = (timesteps + schedule_config.cosine_s) / (1 + schedule_config.cosine_s) * torch.pi / 2
        alphas = torch.cos(alphas) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clip(betas, 0, 0.999)
    elif schedule_config.type == 'sqrt_linear':
        betas = torch.linspace(
            schedule_config.linear_start,
            schedule_config.linear_end,
            schedule_config.num_timesteps,
            dtype=torch.float64
        )
    elif schedule_config.type == 'sqrt':
        betas = torch.linspace(
            schedule_config.linear_start,
            schedule_config.linear_end,
            schedule_config.num_timesteps,
            dtype=torch.float64
        )
        betas = betas ** 0.5
    else:
        raise ValueError(f'Unknown diffusion noise schedule {schedule_config.type}')
    
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat((1., alphas_cumprod[:-1]))

    posterior_variance = (
        (1 - schedule_config.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        + schedule_config.v_posterior * betas
    )
    posterior_log_variance_clipped = torch.log(torch.maximum(posterior_variance, 1e-20))
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    return {
        'betas': betas.type(torch.float32),
        'alphas_cumprod': alphas_cumprod.type(torch.float32),
        'alphas_cumprod_prev': alphas_cumprod_prev.type(torch.float32),
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod).type(torch.float32),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1. - alphas_cumprod).type(torch.float32),
        'log_one_minus_alphas_cumprod': torch.log(1. - alphas_cumprod).type(torch.float32),
        'sqrt_recip_alphas_cumprod': torch.sqrt(1. / alphas_cumprod).type(torch.float32),
        'sqrt_recipm1_alphas_cumprod': torch.sqrt(1. / alphas_cumprod - 1).type(torch.float32),
        'posterior_variance': posterior_variance.type(torch.float32),
        'posterior_log_variance_clipped': posterior_log_variance_clipped.type(torch.float32),
        'posterior_mean_coef1': posterior_mean_coef1.type(torch.float32),
        'posterior_mean_coef2': posterior_mean_coef2.type(torch.float32),
    }