import torch
from typing import Dict, Sequence

from . import schedule


class DDIMSampler():
    def __init__(
        self,
        num_train_steps: int,
        num_inference_steps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        eta: float = 0.,
        denoising_strength: float = 1.0
    ):
        """
        Args: 
            eta: hyperparameter used in Equation (16)
        """
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.betas = schedule.get_betas(beta_start, beta_end, beta_schedule, num_train_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.eta = eta
        self.denoising_strength = denoising_strength

        self.set_timesteps(num_inference_steps)
    
    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        # leading spacing
        self.timesteps = range(0, self.num_train_steps, self.num_train_steps // self.num_inference_steps)
        self.timesteps = self.timesteps[::-1]
    
    def step(self, denoised: torch.Tensor, xt: torch.Tensor, t: int, generator=None):
        t_next = t - self.num_train_steps // self.num_inference_steps
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=xt.device)
        sigma_t = self.eta * torch.sqrt((1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_next))
        
        pred_x0 = denoised
        pred_z = (xt - torch.sqrt(alpha_cumprod_t) * denoised) / torch.sqrt(1 - alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        cur_mean = torch.sqrt(alpha_cumprod_t_next) * pred_x0 + torch.sqrt(1 - alpha_cumprod_t_next - sigma_t ** 2) * pred_z

        if self.eta > 0 and t > 0:
            variance = sigma_t * torch.randn(pred_x0.shape, generator=generator)
        else:
            variance = 0
        sample_next = cur_mean + variance
        
        return sample_next
    
    @torch.no_grad()
    def sample(
        self,
        denoiser,
        batch_size: int,
        image_shape: Sequence = None,
        image: torch.Tensor = None,
        noise: torch.Tensor = None,
        denoiser_args: dict = {},
        generator=None
    ) -> torch.Tensor:
        """
        Args:
            denoiser: takes charge of one step denoising, return denoised image
            image: initial noised image (e.g. from inverse sampling)
        """
        num_steps = int(round(self.num_inference_steps * self.denoising_strength))
        start_t = self.timesteps[-num_steps]
        timesteps = self.timesteps[-num_steps:]

        if noise is None:
            if image is not None:
                noise = torch.randn((batch_size, *image_shape), generator=generator, device=denoiser.inner_model.device)
            else:
                noise = torch.randn(image.shape, generator=generator, device=image.device)
        if image is None:
            x = noise
        else:
            x = torch.sqrt(self.alphas_cumprod[start_t]) * image + torch.sqrt(1 - self.alphas_cumprod[start_t]) * noise
        
        for t in timesteps:
            time_t = torch.full((batch_size,), t, dtype=torch.int64, device=x.device)
            denoised = denoiser(x, time_t, **denoiser_args)
            # if model.prediction_type == 'epsilon':
            #     epsilon = output
            #     sample = (image - torch.sqrt(1 - self.alphas_cumprod[t]) * output) / torch.sqrt(self.alphas_cumprod[t])
            # elif model.prediction_type == 'sample':
            #     sample = output
            #     epsilon = (image - torch.sqrt(self.alphas_cumprod[t]) * output) / torch.sqrt(1 - self.alphas_cumprod[t])
            # elif model.prediction_type == 'v_prediction':
            #     raise NotImplementedError('v_prediction is not implemented for DDPM.')
            x = self.step(denoised, x, t, generator=generator)
        return x


class DDIMInverseSampler():
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
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.set_timesteps(num_inference_steps)
    
    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        # leading spacing
        self.timesteps = range(0, self.num_train_steps, self.num_train_steps // self.num_inference_steps)
    
    def sample(
        self,
        model,
        image: torch.Tensor,
    ) -> torch.Tensor:
        xt = image
        for it in range(len(self.timesteps)):
            t = self.timesteps[it]
            t_prev = self.timesteps[it-1] if it > 0 else -1
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=model.device)

            output = model(xt, t_prev)
            if model.prediction_type == 'epsilon':
                epsilon = output
                sample = (xt - torch.sqrt(1 - alpha_cumprod_t_prev) * output) / torch.sqrt(alpha_cumprod_t_prev)
            elif model.prediction_type == 'sample':
                sample = output
                epsilon = (xt - torch.sqrt(alpha_cumprod_t_prev) * output) / torch.sqrt(1 - alpha_cumprod_t_prev)
            elif model.prediction_type == 'v_prediction':
                raise NotImplementedError('v_prediction is not implemented for DDPM.')
            
            xt = torch.sqrt(alpha_cumprod_t) * sample + torch.sqrt(1 - alpha_cumprod_t) * epsilon
        
        return xt