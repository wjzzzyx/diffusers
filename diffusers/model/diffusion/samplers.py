import torch


class VanillaSampler():

    def __init__(self, clip_denoised):
        self.clip_denoised = clip_denoised
    
    @torch.no_grad()
    def __call__(self, model, x, return_intermediates=False, log_every_t=1):
        batch_size = x.size(0)
        intermediates = list()
        # t in [999, ..., 0]
        for t in reversed(range(0, model.schedule_config.num_timesteps)):
            if return_intermediates and t % log_every_t == 0:
                intermediates.append(x.cpu())
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=x.device)
            x = model.p_sample(x, t_tensor, clip_denoised=self.clip_denoised)
        
        if return_intermediates:
            return x, intermediates
        return x


class DDIMSampler():

    def __init__(self, ddim_num_timesteps, eta, clip_denoised):
        self.ddim_num_timesteps = ddim_num_timesteps
        self.eta = eta
        self.clip_denoised = clip_denoised
    
    @torch.no_grad()
    def __call__(self, model, x, return_intermediates=False, log_every_t=1):
        stepsize = model.schedule_config.num_timesteps // self.ddim_num_timesteps
        ddim_timesteps = list(reversed(range(model.schedule_config.num_timesteps - 1, -1, -stepsize)))
        # ddim_timesteps = [t + 1 for t in ddim_timesteps]
        ddim_alphas_cumprod = model.alphas_cumprod[ddim_timesteps]
        ddim_alphas_cumprod_prev = torch.cat((model.alphas_cumprod[0:1], model.alphas_cumprod[ddim_timesteps[:-1]]))
        ddim_sigmas = torch.sqrt((1 - ddim_alphas_cumprod_prev) / (1 - ddim_alphas_cumprod) * (1 - ddim_alphas_cumprod / ddim_alphas_cumprod_prev))
        ddim_sigmas = ddim_sigmas * self.eta

        batch_size = x.size(0)
        intermediates = list()

        for i in range(self.ddim_num_timesteps - 1, -1, -1):
            t = ddim_timesteps[i]
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=x.device)
            pred = model.model(x, t_tensor)
            pred_x0, z = model.get_x0_z_from_pred(x, t, pred, clip_denoised=self.clip_denoised)
            dir_xt = (1. - ddim_alphas_cumprod_prev[i] - ddim_sigmas[i] ** 2).sqrt() * z
            gaussian_sample = torch.randn_like(x)
            x = ddim_alphas_cumprod_prev[i].sqrt() * pred_x0 + dir_xt + ddim_sigmas[i] * gaussian_sample

            if return_intermediates and i % log_every_t == 0:
                intermediates.append(x.cpu())
        
        if return_intermediates:
            return pred_x0, intermediates
        return pred_x0