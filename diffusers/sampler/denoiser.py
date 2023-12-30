import torch


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
