import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(
        self, loss_type, prediction_type, base_huber_c = 0.1,
        min_snr_gamma = 0, scale_v_pred_loss_like_noise_pred = False, v_pred_like_loss = 0,
        debiased_estimation_loss = False
    ):
        self.loss_type = loss_type
        self.prediction_type = prediction_type
        self.base_huber_c = base_huber_c
        self.min_snr_gamma = min_snr_gamma
        self.scale_v_pred_loss_like_noise_pred = scale_v_pred_loss_like_noise_pred
        self.v_pred_like_loss = v_pred_like_loss
        self.debiased_estimation_loss = debiased_estimation_loss
    
    def forward(self, pred, target, batch, time):
        if self.loss_type == 'l2':
            loss = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == 'huber':
            huber_c = self.get_timestep_huber_c(batch, time)
            loss = 2 * huber_c * (torch.sqrt((pred - target) ** 2 + huber_c ** 2) - huber_c)
        elif self.loss_type == 'smooth_l1':
            huber_c = self.get_timestep_huber_c(batch, time)
            loss = 2 * (torch.sqrt((pred - target) ** 2 + huber_c ** 2) - huber_c)
        else:
            raise NotImplementedError()

        loss = loss.mean([1, 2, 3])

        if self.min_snr_gamma:
            snr_t = batch['snrs'][time]
            min_snr_gamma = torch.clamp(snr_t, max=self.min_snr_gamma)
            if self.prediction_type == 'v':
                snr_weight = torch.div(min_snr_gamma, snr_t + 1)
            else:
                snr_weight = torch.div(min_snr_gamma, snr_t)
            loss = loss * snr_weight
        
        if self.scale_v_pred_loss_like_noise_pred:
            snr_t = batch['snrs'][time]
            snr_t = torch.clamp(snr_t, max=1000)    # when t == 0, snr_t is inf
            scale = snr_t / (snr_t + 1)
            loss = loss * scale
        
        if self.v_pred_like_loss:
            snr_t = batch['snrs'][time]
            snr_t = torch.clamp(snr_t, max=1000)
            scale = snr_t / (snr_t + 1)
            loss = loss + self.v_pred_like_loss / scale * loss
        
        if self.debiased_estimation_loss:
            snr_t = batch['snrs'][time]
            snr_t = torch.clamp(snr_t, 1000)
            weight = 1 / torch.sqrt(snr_t)
            loss = loss * weight
        
        loss = loss.mean()
        return loss
        
    def get_timestep_huber_c(self, batch, time):
        if self.huber_schedule == 'exponential':
            huber_c = math.exp(math.log(self.base_huber_c) * time / batch['num_train_timesteps'])
        elif self.huber_schedule == 'snr':
            alphas_cumprod_t = batch['alphas_cumprod']
            sigmas = torch.sqrt((1.0 - alphas_cumprod_t) / alphas_cumprod_t)
            huber_c = (1 - self.base_huber_c) / (1 + sigmas) ** 2 + self.base_huber_c
        elif self.huber_schedule == 'constant':
            huber_c = self.base_huber_c
        else:
            raise NotImplementedError()
        return huber_c