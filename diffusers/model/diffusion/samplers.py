import torch


class VanillaSampler():

    def __init__(self, sampler_config):
        self.config = sampler_config
    
    @torch.no_grad()
    def __call__(self, model, batch_size, shape, return_intermediates):
        x = torch.randn((batch_size, *shape))
        intermediates = list()
        # t in [0, num_timesteps) -> x1...x999
        for t in reversed(range(0, model.schedule_config.num_timesteps)):
            intermediates.append(x.cpu())
            t_tensor = torch.full((batch_size,), t, dtype=torch.long)
            x = model.p_sample(x, t_tensor, clip_denoised=self.config.clip_denoised)
        
        if return_intermediates:
            return x, intermediates
        return x