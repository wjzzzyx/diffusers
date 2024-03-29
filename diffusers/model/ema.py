import torch
from torch import nn


class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_updates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        # copy ema parameter to model parameter
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


class EMAHelper():
    def __init__(self, max_decay=0.9999, min_decay=0.0, update_after_step=0, use_ema_warmup=False, inv_gamma=1.0, power=2/3):
        self.max_decay = max_decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.num_updates = 0
        self.decay = 0.0
    
    @torch.no_grad()
    def ema_step(self, model, model_ema):
        # find the current step decay value
        self.num_updates += 1
        step = self.num_updates - self.update_after_step - 1
        if step <= 0:
            decay = 0.0
        else:
            if self.use_ema_warmup:
                decay = 1 - (1 + step / self.inv_gamma) ** (-self.power)
            else:
                decay = (1 + step) / (10 + step)
        decay = min(decay, self.max_decay)
        decay = max(decay, self.min_decay)
        self.decay = decay

        for (name, param), (name_ema, param_ema) in zip(model.named_parameters(), model_ema.named_parameters()):
            assert(name == name_ema)
            if param.requires_grad:
                param_ema.sub_((1 - decay) * (param_ema - param))
            else:
                param_ema.copy_(param)