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
        eta: float,
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

        self.set_timesteps(num_inference_steps)
    
    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        # leading spacing
        self.timesteps = range(0, self.num_train_steps, self.num_train_steps // self.num_inference_steps)
        self.timesteps = self.timesteps[::-1]
    
    def step(self, output: Dict[str, torch.Tensor], t: int, generator=None):
        t_next = t - self.num_train_steps // self.num_inference_steps
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=output['epsilon'].device)
        sigma_t = self.eta * torch.sqrt((1 - alpha_cumprod_t_next) / alpha_cumprod_t_next)
        
        pred_x0 = output['sample']
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        pred_z = output['epsilon']

        cur_mean = torch.sqrt(alpha_cumprod_t_next) * pred_x0 + torch.sqrt(1 - alpha_cumprod_t_next - sigma_t ** 2) * pred_z

        if self.eta > 0 and t > 0:
            variance = sigma_t * torch.randn(pred_x0.shape, generator=generator)
        else:
            variance = 0
        sample_next = cur_mean + variance
        
        return sample_next
    
    def sample(
        self,
        model,
        batch_size: int,
        image_shape: Sequence,
        image: torch.Tensor = None,
        cond_pos_prompt: torch.Tensor = None,
        cond_neg_prompt: torch.Tensor = None,
        generator=None
    ) -> torch.Tensor:
        """
        Args:
            image: initial noised image (e.g. from inverse sampling)
        """
        if image is None:
            image = torch.randn((batch_size, *image_shape), generator=generator, device=model.device)

        extra_args = dict()
        if cond_pos_prompt is not None:
            extra_args['cond_pos_prompt'] = cond_pos_prompt
        if cond_neg_prompt is not None:
            extra_args['cond_neg_prompt'] = cond_neg_prompt
        
        for t in self.timesteps:
            output = model(image, t, **extra_args)
            if model.prediction_type == 'epsilon':
                epsilon = output
                sample = (image - torch.sqrt(1 - self.alphas_cumprod[t]) * output) / torch.sqrt(self.alphas_cumprod[t])
            elif model.prediction_type == 'sample':
                sample = output
                epsilon = (image - torch.sqrt(self.alphas_cumprod[t]) * output) / torch.sqrt(1 - self.alphas_cumprod[t])
            elif model.prediction_type == 'v_prediction':
                raise NotImplementedError('v_prediction is not implemented for DDPM.')
            image = self.step({'sample': sample, 'epsilon': epsilon}, t, image, generator=generator)
        return image


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
    
    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        # leading spacing
        self.timesteps = range(0, self.num_train_steps, self.num_train_steps // self.num_inference_steps)
    
    def step(self, output, t_next: int):
        alpha_cumprod_t_next = self.alphas_cumprod[t_next]

        pred_x0 = output['sample']
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        pred_z = output['epsilon']

        sample_next = torch.sqrt(alpha_cumprod_t_next) * pred_x0 + torch.sqrt(1 - alpha_cumprod_t_next) * pred_z

        return sample_next
    
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

            output = model(image, t_prev)
            if model.prediction_type == 'epsilon':
                epsilon = output
                sample = (image - torch.sqrt(1 - alpha_cumprod_t_prev) * output) / torch.sqrt(alpha_cumprod_t_prev)
            elif model.prediction_type == 'sample':
                sample = output
                epsilon = (image - torch.sqrt(alpha_cumprod_t_prev) * output) / torch.sqrt(1 - alpha_cumprod_t_prev)
            elif model.prediction_type == 'v_prediction':
                raise NotImplementedError('v_prediction is not implemented for DDPM.')
            
            xt = torch.sqrt(alpha_cumprod_t) * sample + torch.sqrt(1 - alpha_cumprod_t) * epsilon
        
        return xt