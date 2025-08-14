import itertools
import lightning
import math
import numpy as np
import os
from PIL import Image
import re
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from torch_utils import replace_substring_in_state_dict_if_present
from diffusers.model.stable_diffusion_stabilityai import StableDiffusion_StabilityAI
from diffusers.sampler.denoiser import KarrasEpsDenoiser, KarrasVDenoiser, KarrasCFGDenoiser

suffix_conversion = {
    'conv1': 'in_layers_2',
    'conv2': 'out_layers_3',
    'norm1': 'in_layers_0',
    'norm2': 'out_layers_0',
    'time_emb_proj': 'emb_layers_1',
    'conv_shortcut': 'skip_connection',
}

def convert_lora_module_name(key):
    m = re.match(r"lora_unet_conv_in(.*)", key)
    if m:
        return f'lora_unet_input_blocks_0_0{m.group(1)}'
    m = re.match(r"lora_unet_conv_out(.*)", key)
    if m:
        return f'lora_unet_out_2{m.group(1)}'
    m = re.match(r"lora_unet_time_embedding_linear_(\d+)(.*)", key)
    if m:
        return f'lora_unet_time_embed_{int(m.group(1)) * 2 - 2}{m.group(2)}'
    m = re.match(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)", key)
    if m:
        return f"lora_unet_input_blocks_{1 + int(m.group(1)) * 3 + int(m.group(2))}_1_{m.group(3)}"
    m = re.match(r"lora_unet_down_blocks_(\d+)_resnets_(\d+)_(.+)", key)
    if m:
        suffix = suffix_conversion.get(m.group(3), m.group(3))
        return f"lora_unet_input_blocks_{1 + int(m.group(1)) * 3 + int(m.group(2))}_0_{suffix}"
    m = re.match(r"lora_unet_mid_block_resnets_(\d+)_(.+)", key)
    if m:
        suffix = suffix_conversion.get(m.group(2), m.group(2))
        return f"lora_unet_middle_block_{int(m.group(1)) * 2}_{suffix}"
    m = re.match(r"lora_unet_mid_block_attentions_(\d+)_(.+)", key)
    if m:
        return f"lora_unet_middle_block_1_{m.group(2)}"
    m = re.match(r"lora_unet_up_blocks_(\d+)_resnets_(\d+)_(.+)", key)
    if m:
        suffix = suffix_conversion.get(m.group(3), m.group(3))
        return f"lora_unet_output_blocks_{int(m.group(1)) * 3 + int(m.group(2))}_0_{suffix}"
    m = re.match(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)", key)
    if m:
        return f"lora_unet_output_blocks_{int(m.group(1)) * 3 + int(m.group(2))}_1_{m.group(3)}"
    m = re.match(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv", key)
    if m:
        return f"lora_unet_input_blocks_{3 + int(m.group(1)) * 3}_0_op"
    m = re.match(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv", key)
    if m:
        return f"lora_unet_output_blocks_{2 + int(m.group(1)) * 3}_{2 if int(m.group(1)) > 0 else 1}_conv"
    m = re.match(r"lora_te_text_model_encoder_layers_(\d+)_(.+)", key)
    if m:
        return f"lora_te_transformer_text_model_encoder_layers_{m.group(1)}_{m.group(2)}"
    raise ValueError('Unmatched Lora module name {key}')
    

def convert_lora_module_names(state_dict):
    keys = list(state_dict.keys())
    for key in keys:
        new_key = convert_lora_module_name(key)
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
    return state_dict


class LoRABase(nn.Module):
    """ replace forward method in the original module """
    def __init__(
        self, org_module, lora_dim, alpha,
        dropout_module=0.0, dropout=0.0, dropout_rank=0.0
    ):
        super().__init__()
        self.lora_dim = lora_dim
        self.dropout_module = dropout_module
        self.dropout = dropout
        self.dropout_rank = dropout_rank
        self.org_forward = org_module.forward
        org_module.forward = self.forward
        self.register_buffer('alpha', torch.tensor(alpha))
        self.scale = alpha / self.lora_dim
    
    def forward(self, x):
        org_out = self.org_forward(x)

        # dropout the whole module
        if self.training:
            if self.dropout_module > 0 and torch.rand(1) < self.dropout_module:
                return org_out

        x = self.lora_down(x)
        
        scale = self.scale
        if self.training:
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout)
            if self.dropout_rank > 0:
                mask = torch.rand((x.size(0), self.lora_dim, *x.shape[2:]), device=x.device) > self.dropout_rank
                x = x * mask
                scale = self.scale * (1.0 / (1.0 - self.dropout_rank))
        
        x = self.lora_up(x)
        return org_out + scale * x


class LoRAMLP(LoRABase):
    def __init__(
        self, org_module, lora_dim, alpha,
        dropout_module=0.0, dropout=0.0, dropout_rank=0.0
    ):
        super().__init__(org_module, lora_dim, alpha, dropout_module, dropout, dropout_rank)
        self.lora_down = nn.Linear(org_module.in_features, lora_dim, bias=False)
        self.lora_up = nn.Linear(lora_dim, org_module.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
    
    @torch.no_grad()
    def merge(self, multiplier):
        self.org_module.weight = (
            self.org_module.weight + multiplier * self.scale * (self.lora_up.weight @ self.lora_down.weight)
        )


class LoRAConv(LoRABase):
    def __init__(
        self, org_module, lora_dim, alpha,
        dropout_module=0.0, dropout=0.0, dropout_rank=0.0
    ):
        super().__init__(org_module, lora_dim, alpha, dropout_module, dropout, dropout_rank)
        self.lora_down = nn.Conv2d(
            org_module.in_channels, lora_dim, org_module.kernel_size,
            org_module.stride, org_module.padding, bias=False
        )
        self.lora_up = nn.Conv2d(lora_dim, org_module.out_channels, 1, 1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
    
    @torch.no_grad()
    def merge(self, multiplier):
        self.org_module.weight = (
            self.org_module.weight
            + multiplier * self.scale * (
                self.lora_up.weight.squeeze(3).squeeze(2) @ self.lora_down.weight.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3)
        )


class LoraNetwork(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def add_lora_modules(
        self, lora_dim, lora_alpha, diffusion_model, cond_stage_model,
        dropout_module=0, dropout=0, dropout_rank=0
    ):
        self.lora_module_names = list()

        def add_a_module(lora_name, child_module):
            lora_module_cls = LoRAMLP if child_module.__class__.__name__ == 'Linear' else LoRAConv
            dim = lora_dim[lora_name] if isinstance(lora_dim, dict) else lora_dim
            alpha = lora_alpha[lora_name] if isinstance(lora_alpha, dict) else lora_alpha
            lora_module = lora_module_cls(
                child_module, dim, alpha,
                dropout_module, dropout, dropout_rank
            )
            self.add_module(lora_name, lora_module)
            self.lora_module_names.append(lora_name)

        for name, module in diffusion_model.named_modules():
            if module.__class__.__name__ == 'SpatialTransformer':
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ['Linear', 'Conv2d']:
                        lora_name = f'lora_unet.{name}.{child_name}'
                        lora_name = lora_name.replace('.', '_')
                        add_a_module(lora_name, child_module)
        
        for name, module in cond_stage_model.named_modules():
            if module.__class__.__name__ in ['CLIPAttention', 'CLIPMLP']:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ['Linear', 'Conv2d']:
                        lora_name = f'lora_te.{name}.{child_name}'
                        lora_name = lora_name.replace('.', '_')
                        add_a_module(lora_name, child_module)
    
    def add_lora_modules_from_weight(self, state_dict, diffusion_model, cond_stage_model):
        loraname2alpha = dict()
        loraname2dim = dict()
        for key, weight in state_dict.items():
            lora_name = key.split('.')[0]
            if 'alpha' in key:
                loraname2alpha[lora_name] = weight.item()
            elif 'lora_down' in key:
                loraname2dim[lora_name] = weight.size(0)
        assert(loraname2alpha.keys() == loraname2dim.keys())

        self.add_lora_modules(loraname2dim, loraname2alpha, diffusion_model, cond_stage_model)
    
    def merge(self):
        for name, child_module in self.named_children():
            child_module.merge(self.multiplier)


class StableDiffusion_Lora(StableDiffusion_StabilityAI):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.lora_networks = nn.ModuleList()
        if 'pretrained_lora' in model_config:
            for cfg in model_config.pretrained_lora:
                if cfg.path.endswith('safetensors'):
                    checkpoint = safetensors.torch.load_file(cfg.path, device='cpu')
                else:
                    checkpoint = torch.load(cfg.path, map_location='cpu')
                state_dict = convert_lora_module_names(checkpoint)
                network = LoraNetwork(cfg.multiplier)
                network.add_lora_modules_from_weight(state_dict, self.diffusion_model, self.cond_stage_model)
                missing, unexpected = network.load_state_dict(state_dict, strict=False)
                self.lora_networks.append(network)
        else:
            network = LoraNetwork(1.0)
            network.add_lora_modules(
                model_config.lora_dim, model_config.lora_alpha, self.diffusion_model, self.cond_stage_model,
                model_config.dropout_module, model_config.dropout, model_config.dropout_rank
            )
            self.lora_networks.append(network)
        
        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)
        self.cond_stage_model.eval()
        self.cond_stage_model.requires_grad_(False)
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)
    
    def trainable_parameters(self):
        return self.lora_networks.parameters()


class PLBase(lightning.LightningModule):
    def __init__(
        self,
        model_config,
        loss_config,
        optimizer_config,
        metric_config,
        sampler_config,
    ):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.metric_config = metric_config
        self.sampler_config = sampler_config

        self.model = StableDiffusion_Lora(model_config)
        self.loss_fn = utils.instantiate_from_config(loss_config)
        self.sampler = utils.instantiate_from_config(sampler_config)
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['images'].size(0)
        time = torch.randint(0, self.sampler.num_train_steps, (batch_size,), device=self.device)
        batch['time'] = time
        latents = self.model.encode_first_stage(batch['images'])
        cond_prompt = self.model.cond_stage_model(batch['captions'])

        noise = torch.randn_like(latents)
        alphas_cumprod = self.sampler.alphas_cumprod.to(self.device)
        alphas_cumprod_t = alphas_cumprod[time][..., None, None, None]
        xt = torch.sqrt(alphas_cumprod_t) * latents + torch.sqrt(1 - alphas_cumprod_t) * noise
        output = self.model.diffusion_model(xt, time, cond_prompt)

        loss, logdict = self.loss_fn(output, noise, batch)
        logdict = {f'train/{k}': v for k, v in logdict.items()}
        self.log_dict(logdict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch_size)

        pred = (xt - torch.sqrt(1 - alphas_cumprod_t) * output) / torch.sqrt(alphas_cumprod_t)
        pred = self.model.decode_first_stage(pred)
        pred = (pred + 1) / 2
        log_image_dict = {'image': (batch['image'] + 1) / 2, 'pred': pred}
        self.log_image(log_image_dict, ['image', 'pred'], batch_idx, mode='train')

        return loss
    
    def validation_step(self, batch, batch_idx):
        alphas_cumprod = self.sampler.alphas_cumprod.to(self.device)
        if self.model.prediction_type == 'epsilon':
            denoiser = KarrasEpsDenoiser(self.model, alphas_cumprod)
        elif self.model.prediction_type == 'v':
            denoiser = KarrasVDenoiser(self.model, alphas_cumprod)
        denoiser = KarrasCFGDenoiser(denoiser, 7)

        batch_size = len(batch['images'])
        cond_pos_prompt = self.model.cond_stage_model(batch['captions'])
        cond_neg_prompt = self.model.cond_stage_model(['' for _ in range(batch_size)])
        denoiser_args = {'cond_pos_prompt': cond_pos_prompt, 'cond_neg_prompt': cond_neg_prompt}
        samples = self.sampler.sample(
            denoiser, batch_size=batch_size, image_shape=(4, 64, 64), denoiser_args=denoiser_args
        )
        samples = self.model.decode_first_stage(samples)
        samples = torch.clamp((samples + 1) / 2, min=0, max=1)
        log_image_dict = {'image': samples}
        log_keys=['image']
        self.log_image(log_image_dict, log_keys, batch_idx, mode='validation')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if k.startswith('model.lora_networks')}

    def configure_optimizers(self):
        trainable_params = self.model.trainable_parameters()
        optimizer = utils.get_obj_from_str(self.optimizer_config.optimizer)(
            trainable_params, **self.optimizer_config.optimizer_params
        )
        lr_scheduler = utils.get_obj_from_str(self.optimizer_config.lr_scheduler)(
            optimizer, **self.optimizer_config.lr_scheduler_params
        )
        if self.optimizer_config.warmup:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=self.optimizer_config.warmup
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, lr_scheduler], milestones=[self.optimizer_config.warmup]
            )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step'
            }
        }

    @torch.no_grad()
    def log_image(self, batch, keys, batch_idx, mode):
        """
        Args:
            batch: dictionary, key: str -> value: tensor
        """
        dirname = os.path.join(self.trainer.default_root_dir, 'log_images', mode)
        os.makedirs(dirname, exist_ok=True)
        for key in keys:
            image_t = batch[key].permute(0, 2, 3, 1).squeeze(-1)
            image_np = image_t.detach().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            for i in range(image_np.shape[0]):
                filename = f'gs-{self.global_step:04}_e-{self.current_epoch:04}_b-{batch_idx:04}-{i:02}_{key}.png'
                Image.fromarray(image_np[i]).save(os.path.join(dirname, filename))


class TrainerDDPO(Trainer):
    def train_step(self, batch, batch_idx, global_step):
        batch_size = len(batch["prompts"])
        height = batch["height"][0]
        width = batch["width"][0]
        mini_batch_size = self.mini_batch_size
        # TODO same prompt on same device? no
        with torch.no_grad():
            all_text_embs = list()
            all_latents = list()
            all_next_latents = list()
            all_log_probs = list()
            all_rewards = list()
            for i in range(0, batch_size, mini_batch_size):
                prompts = batch["prompts"][i:i+mini_batch_size]
                text_embs = self.clip(prompts)
                neg_prompts = ["" for _ in range(mini_batch_size)]
                # sample images
                noise = torch.randn((mini_batch_size, 4, height, width), device=self.device, generator=None)
                latents, log_probs = self.sampler.sample()
                images = latents[-1] / self.vae.scale_factor
                images = self.vae.decode(images)
                images = (images.clamp(-1, 1) + 1) / 2
                rewards = self.reward_fn(images, prompts)
                
                all_text_embs.append(text_embs)
                all_latents.append(torch.stack([noise, *latents[:-1]], dim=1))
                all_next_latents.append(torch.stack(latents, dim=1))
                all_log_probs.append(log_probs)
                all_rewards.append(rewards)

            all_text_embs = torch.cat(all_text_embs, dim=0)
            all_latents = torch.cat(all_latents, dim=0)
            all_next_latents = torch.cat(all_next_latents, dim=0)
            all_log_probs = torch.cat(all_log_probs, dim=0)    # (batch_size, num_timesteps)
            all_rewards = torch.cat(all_rewards, dim=0)    # (batch_size,)
            all_timesteps = self.sampler.timesteps.repeat(batch_size, 1)
            
            # gather all rewards and calculate advantages
            all_advantages = self.rewards_tracker(batch["prompts"], all_rewards)
        
        # training
        # TODO iterate over sample how many times?
        perm = torch.randperm(batch_size, device=self.device)
        all_text_embs = all_text_embs[perm]
        all_latents = all_latents[perm]
        all_next_latents = all_next_latents[perm]
        all_log_probs = all_log_probs[perm]
        all_advantages = all_advantages[perm]
        all_timesteps = all_timesteps[perm]

        perm = torch.stack([
            torch.randperm(self.sampler.num_inference_timesteps, device=self.device)
            for _ in range(batch_size)
        ])
        all_timesteps = all_timesteps[torch.arange(batch_size, device=self.device)[:, None], perm]
        all_latents = all_latents[torch.arange(batch_size, device=self.device)[:, None], perm]
        all_next_latents = all_next_latents[torch.arange(batch_size, device=self.device)[:, None], perm]
        all_log_probs = all_log_probs[torch.arange(batch_size, device=self.device)[:, None], perm]

        for i in range(0, batch_size, mini_batch_size):
            timesteps = all_timesteps[i:i+mini_batch_size]
            text_embs = all_text_embs[i:i+mini_batch_size]    # (mini_batch_size, seq, emb)
            latents = all_latents[i:i+mini_batch_size]    # (mini_batch_size, timesteps, c, h, w)
            next_latents = all_next_latents[i:i+mini_batch_size]
            advantages = all_advantages[i:i+mini_batch_size]    # (mini_batch_size,)
            log_probs_old = all_log_probs[i:i+mini_batch_size]    # (mini_batch_size, num_timesteps)
            advantages = torch.clamp(advantages, -self.loss_config.adv_clip_max, self.loss_config.adv_clip_max)
            
            # TODO grad accumulate
            # TODO cfg
            for j in range(self.sampler.num_inference_timesteps):
                outputs = self.unet(latents[:, j], timesteps[:, j], context=text_embs)
                log_probs = self.sampler.step(return_log_probs=True)
                ratio = torch.exp(log_probs - log_probs_old[:, j])
                unclipped_part = - advantages * ratio
                clipped_part = - advantages * torch.clamp(ratio, 1 - self.loss_config.ppo_clip, 1 + self.loss_config.ppo_clip)
                loss_ppo = torch.maximum(unclipped_part, clipped_part)
                loss_ppo = loss_ppo.mean()
                loss_ppo.backward()
                params = itertools.chain(*[x["params"] for x in self.optimizer.param_groups])
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()

                approx_kl = 0.5 * torch.mean((log_probs - log_probs_old[:, t]) ** 2)
                clip_frac = torch.mean((torch.abs(ratio - 1) > self.loss_config.ppo_clip).float())
                self.loss_meters["approx_kl"].update(approx_kl, mini_batch_size)
                self.loss_meters["clip_frac"].update(clip_frac, mini_batch_size)
                self.loss_meters["loss_ppo"].update(loss_ppo, mini_batch_size)