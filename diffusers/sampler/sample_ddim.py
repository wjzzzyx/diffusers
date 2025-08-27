import math
import torch

from . import schedule


class DDIMSampler():
    def __init__(
        self,
        num_train_timesteps: int,
        num_inference_timesteps: int,
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
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.betas = schedule.get_betas(beta_start, beta_end, beta_schedule, num_train_timesteps).cuda()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.eta = eta
        self.denoising_strength = denoising_strength

        self.set_timesteps(num_inference_timesteps)
    
    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        # leading spacing
        self.timesteps = torch.arange(0, self.num_train_steps, self.num_train_steps // self.num_inference_steps) + 1
        self.timesteps = self.timesteps.flip(0)
    
    def step(self, denoised: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, pred_xtm1 = None, generator=None):
        tm1 = t - self.num_train_timesteps // self.num_inference_timesteps
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_next = torch.where(tm1 >= 0, self.alphas_cumprod[tm1.clamp(min=0)], self.alhpas_cumprod[0])
        alpha_cumprod_t = alpha_cumprod_t.view(-1, 1, 1, 1)
        alpha_cumprod_t_next = alpha_cumprod_t_next.view(-1, 1, 1, 1)
        sigma_t = self.eta * torch.sqrt((1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_next))
        
        pred_x0 = denoised
        pred_z = (xt - torch.sqrt(alpha_cumprod_t) * denoised) / torch.sqrt(1 - alpha_cumprod_t)
        # pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        pred_xtm1_mean = torch.sqrt(alpha_cumprod_t_next) * pred_x0 + torch.sqrt(1 - alpha_cumprod_t_next - sigma_t ** 2) * pred_z

        if pred_xtm1 is None:
            variance = torch.where(
                (t > 0).view(-1, 1, 1, 1),
                sigma_t * torch.randn(pred_x0.shape, generator=generator, device=xt.device),
                torch.tensor(0., device=xt.device)
            )
            pred_xtm1 = pred_xtm1_mean + variance
        
        logprob = self.get_logprob(pred_xtm1, pred_xtm1_mean, sigma_t)
        
        return pred_xtm1, logprob

    def get_logprob(self, pred, pred_mean, std):
        logprob = (
            -((pred.detach() - pred_mean) ** 2) / (2 * std ** 2)
            - 0.5 * math.log(2 * math.pi)
            - torch.log(std)
        )
        logprob = logprob.mean(dim=[1, 2, 3])
        return logprob
    
    @torch.no_grad()
    def sample(
        self,
        denoiser,
        noise: torch.Tensor,
        image: torch.Tensor = None,
        denoiser_args: dict = {},
        generator=None,
        return_logprob: bool = False
    ) -> torch.Tensor:
        """
        Args:
            denoiser: takes charge of one step denoising, return denoised image
            image: initial noised image (e.g. from inverse sampling)
        """
        num_steps = int(round(self.num_inference_steps * self.denoising_strength))
        start_t = self.timesteps[-num_steps]
        timesteps = self.timesteps[-num_steps:]

        if image is None:
            x = noise
        else:
            x = torch.sqrt(self.alphas_cumprod[start_t]) * image + torch.sqrt(1 - self.alphas_cumprod[start_t]) * noise
        
        xts, logprobs = list(), list()
        for t in timesteps:
            time_t = torch.full((noise.size(0),), t, dtype=torch.int64, device=x.device)
            denoised = denoiser(x, time_t, **denoiser_args)
            x, logprob = self.step(denoised, x, time_t, generator=generator)
            xts.append(x)
            logprobs.append(logprob)
        
        if return_logprob:
            return xts, logprobs
        else:
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