import collections
import contextlib
import itertools
import lightning
import math
import numpy as np
import os
from PIL import Image, ImageDraw
import pyiqa
import re
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import utils
import torch_utils
from diffusers.model.stable_diffusion_stabilityai import StableDiffusion_StabilityAI
from diffusers.sampler.denoiser import KarrasEpsDenoiser, KarrasVDenoiser, KarrasCFGDenoiser
from diffusers.loss.diffusion_loss import TimeWeightedL2Loss

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
        self, module, lora_rank, lora_alpha,
        dropout_module=0.0, dropout=0.0, dropout_rank=0.0
    ):
        super().__init__()
        self.origin_module = module
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scale = lora_alpha / lora_rank
        self.dropout_module = dropout_module
        self.dropout = dropout
        self.dropout_rank = dropout_rank
    
    def forward(self, x):
        orig_out = self.origin_module(x)
        # dropout the whole module
        if self.training:
            if self.dropout_module > 0 and torch.rand(1) < self.dropout_module:
                return orig_out
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
        return orig_out + scale * x


class LoRALinear(LoRABase):
    def __init__(
        self, module, lora_rank, lora_alpha,
        dropout_module=0.0, dropout=0.0, dropout_rank=0.0
    ):
        super().__init__(module, lora_rank, lora_alpha, dropout_module, dropout, dropout_rank)
        self.lora_down = nn.Linear(module.in_features, lora_rank, bias=False)
        self.lora_up = nn.Linear(lora_rank, module.out_features, bias=False)
        # nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.normal_(self.lora_down.weight, std=1 / lora_rank)    # this is important
        nn.init.zeros_(self.lora_up.weight)
    
    @torch.no_grad()
    def merge(self, multiplier):
        self.origin_module.weight = (
            self.origin_module.weight + multiplier * self.scale * (self.lora_up.weight @ self.lora_down.weight)
        )


class LoRAConv2d(LoRABase):
    def __init__(
        self, module, lora_rank, lora_alpha,
        dropout_module=0.0, dropout=0.0, dropout_rank=0.0
    ):
        super().__init__(module, lora_rank, lora_alpha, dropout_module, dropout, dropout_rank)
        self.lora_down = nn.Conv2d(
            module.in_channels, lora_rank, module.kernel_size,
            module.stride, module.padding, bias=False
        )
        self.lora_up = nn.Conv2d(lora_rank, module.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
    
    @torch.no_grad()
    def merge(self, multiplier):
        self.origin_module.weight = (
            self.origin_module.weight
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


def attach_lora_to_sd(model, lora_rank, lora_alpha):
    lora_model = nn.ModuleDict()
    for name, module in model.named_modules():
        if module.__class__.__name__ == "CrossAttention":
            for sub_name, sub_module in module.named_modules():
                if not sub_module.__class__.__name__ in ["Linear", "Conv2d"]:
                    continue
                full_name = f"{name}.{sub_name}"
                lora_name = f"lora_unet.{full_name}"
                lora_name = lora_name.replace(".", "_")
                if sub_module.__class__.__name__ == "Linear":
                    lora_module = LoRALinear(sub_module, lora_rank, lora_alpha)
                elif sub_module.__class__.__name__ == "Conv2d":
                    lora_module = LoRAConv2d(sub_module, lora_rank, lora_alpha)
                parent_name = ".".join(full_name.split(".")[:-1])
                parent = model.get_submodule(parent_name)
                child_name = full_name.split(".")[-1]
                setattr(parent, child_name, lora_module)
                lora_model[lora_name] = lora_module
    return lora_model


class Trainer():
    def __init__(self, model_config, loss_config, optimizer_config, device):
        self.model_config = model_config
        self.loss_config = loss_config
        self.device = device
        
        self.unet = utils.instantiate_from_config(model_config.unet_config).cuda()
        self.unet.eval()
        self.unet.requires_grad_(False)
        # self.unet = self.unet.to(memory_format=torch.channels_last)
        self.vae = utils.instantiate_from_config(model_config.first_stage_config).cuda()
        self.vae.eval()
        self.vae.requires_grad_(False)
        # self.vae = self.vae.to(memory_format=torch.channels_last)
        self.clip = utils.instantiate_from_config(model_config.cond_stage_config).cuda()
        self.clip.eval()
        self.clip.requires_grad_(False)
        self.lora_model = attach_lora_to_sd(self.unet, model_config.lora.lora_rank, model_config.lora.lora_alpha)
        self.lora_model.cuda()
        self.unet = DistributedDataParallel(self.unet, device_ids=[device])
        self.sampler = utils.instantiate_from_config(model_config.sampler_config)

        # prepare loss
        # self.diffusion_loss_fn = TimeWeightedL2Loss(
        #     loss_config.kind, loss_config.prediction_type, alphas_cumprod=self.sampler.alphas_cumprod
        # )

        # prepare optimizer
        self.optimizer = utils.get_obj_from_str(optimizer_config.optimizer)(
            self.lora_model.parameters(), **optimizer_config.optimizer_params
        )
        optimizer_config.lr_scheduler_params.T_max = optimizer_config.num_training_steps
        self.lr_scheduler = utils.get_obj_from_str(optimizer_config.lr_scheduler)(
            self.optimizer, **optimizer_config.lr_scheduler_params
        )

        # prepare metrics
        self.loss_meters = {}
    
    def on_train_epoch_start(self):
        self.lora_model.train()
        for key in self.loss_meters:
            self.loss_meters[key].reset()

    def train_step(self, batch, batch_idx, global_step):
        with torch.autocast("cuda", torch.bfloat16):
            batch_size = batch['image'].shape[0]
            images = batch['image'].cuda(non_blocking=True)
            images = images.to(memory_format=torch.channels_last)
            images = images * 2 - 1
            latents = self.vae.encode(images).mean
            latents = latents * self.vae.scale_factor
            text_emb = self.clip(batch['text'])
            
            noise = torch.randn_like(latents)
            time = torch.randint(0, self.sampler.num_train_timesteps, (batch_size,), device=self.device)
            alphas_cumprod_t = self.sampler.alphas_cumprod[time][(...,) + (None,) * 3]
            xt = torch.sqrt(alphas_cumprod_t) * latents + torch.sqrt(1 - alphas_cumprod_t) * noise
            outputs = self.unet(xt, time, context=text_emb)
            loss = self.diffusion_loss_fn(outputs, xt, noise, time)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.loss_meters["loss_l2"].update(loss, batch_size)

    def on_train_epoch_end(self, epoch):
        return dict()
    
    @torch.no_grad()
    def log_step(self, logdir, global_step, epoch, batch_idx):
        logdict = dict()
        for key in self.loss_meters.keys():
            val = self.loss_meters[key].compute()
            logdict[key] = val.item()
            self.loss_meters[key].reset()
        logdict["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return logdict

    def on_val_epoch_start(self):
        self.lora_model.eval()
        self.metrics_dict = collections.defaultdict(list)
        self.psnr = pyiqa.create_metric('psnr')
        self.ssim = pyiqa.create_metric('ssim')
        self.lpips = pyiqa.create_metric('lpips', version='0.1', net='vgg', device=self.device)
        self.fid = pyiqa.create_metric("fid", device=self.device)
    
    @torch.no_grad()
    @torch.autocast("cuda", torch.bfloat16)
    def val_step(self, batch, global_step, epoch, batch_idx, logdir):
        batch_size = len(batch["text"])
        height, width = batch["height"][0], batch["width"][0]
        text_emb = self.clip(batch["text"])
        generator=torch.Generator(self.device)
        noise = torch.randn((batch_size, 4, height // 8, width // 8), device=self.device, generator=generator)
        
        preds = self.sampler.sample(
            self.denoiser,
            noise,
            denoiser_args={"cond_pos_prompt": text_emb},
            generator=generator
        )
        preds = preds / self.vae.scale_factor
        preds = self.vae.decode(preds)
        preds = (preds.clamp(-1, 1) + 1) / 2
        
        if batch_idx % 100 == 0:
            dirname = os.path.join(logdir, "log_images", "val")
            os.makedirs(dirname, exist_ok=True)
            log_image_dict = {
                "pred": preds,
                "text": batch["text"],
                "fpath": batch["fpath"],
            }
            self.log_image(dirname, global_step, epoch, log_image_dict)
    
    def denoiser(self, xt, time, cond_prompt):
        alphas_cumprod_t = self.sampler.alphas_cumprod[time][(...,) + (None,) * 3]
        outputs = self.unet(xt, time, context=cond_prompt)
        denoised = (xt - torch.sqrt(1 - alphas_cumprod_t) * outputs) / torch.sqrt(alphas_cumprod_t)
        return denoised
    
    def on_val_epoch_end(self, dataset_name, dataset, logdir):
        if dist.get_rank() == 0:
            self.metrics_dict["fid"] = self.fid(
                os.path.join(logdir, "val_preds"), os.path.join(dataset.root_dir, "val", "images")
            )
        self.fid.cpu()
        return self.metrics_dict

    def log_image(self, logdir, global_step, epoch, log_image_dict):
        preds = log_image_dict["pred"].permute(0, 2, 3, 1).cpu().numpy()
        text = log_image_dict["text"]
        fpaths = log_image_dict["fpath"]
        for i in range(len(fpaths)):
            fname = os.path.basename(fpaths[i])
            pred_fname = os.path.splitext(fname)[0] + f"_gs{global_step}_e{epoch}_pred.png"
            pred = Image.fromarray((preds[i] * 255).astype("uint8"))
            new_pred = Image.new(pred.mode, (pred.width, pred.height + 20), "white")
            new_pred.paste(pred, (0, 0))
            ImageDraw.Draw(new_pred).text((10, pred.height), text[i], fill=(0, 0, 0))
            new_pred.save(os.path.join(logdir, pred_fname))

    def get_model_state_dict(self):
        return self.lora_model.module.state_dict()
    
    def get_optimizer_state_dict(self):
        return self.optimizer.state_dict()
    
    def get_lr_scheduler_state_dict(self):
        return self.lr_scheduler.state_dict()
    
    def load_model_state_dict(self, state_dict):
        self.lora_model.module.load_state_dict(state_dict)
    
    def load_optimizer_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    
    def load_lr_scheduler_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)


class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}
    
    def update(self, prompts, rewards):
        world_size = dist.get_world_size()
        gathered_rewards = [torch.zeros_like(rewards) for _ in range(world_size)]
        dist.all_gather(gathered_rewards, rewards)
        rewards = torch.cat(gathered_rewards)
        gathered_prompts = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_prompts, prompts)
        prompts = list(itertools.chain(*gathered_prompts))
        prompts = np.array(prompts)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = collections.deque(maxlen=self.buffer_size)
            self.stats[prompt].append(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        
        advantages = torch.as_tensor(advantages)
        advantages = advantages.view(world_size, -1)[dist.get_rank()].cuda()
        return advantages
    
    def get_stats(self):
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.stats.items()
        }


class TrainerDDPO(Trainer):
    def __init__(self, model_config, loss_config, optimizer_config, device):
        super().__init__(model_config, loss_config, optimizer_config, device)
        self.reward_fn = utils.get_obj_from_str(loss_config.target)
        self.rewards_tracker = PerPromptStatTracker(16, 16)
        self.mini_batch_size = 8

        self.loss_meters = {
            "loss_ppo": torch_utils.RunningStatistic(device),
            "approx_kl": torch_utils.RunningStatistic(device),
            "clip_frac": torch_utils.RunningStatistic(device),
            "reward": torch_utils.RunningStatistic(device),
            "reward_std": torch_utils.RunningStatistic(device),
        }
    
    def train_step(self, batch, global_step, epoch, batch_idx, logdir):
        batch_size = len(batch["text"])
        height = batch["height"][0]
        width = batch["width"][0]
        mini_batch_size = self.mini_batch_size
        # TODO same prompt on same device? no
        self.unet.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            all_text_embs = list()
            all_latents = list()
            all_next_latents = list()
            all_logprobs = list()
            all_rewards = list()
            all_images = list()
            for i in tqdm(range(0, batch_size, mini_batch_size), desc="Sampling", disable=dist.get_rank() != 0):
                prompts = batch["text"][i:i+mini_batch_size]
                text_embs = self.clip(prompts)
                neg_prompts = ["" for _ in range(mini_batch_size)]
                # sample images
                noise = torch.randn((mini_batch_size, 4, height // 8, width // 8), device=self.device, generator=None)
                latents, log_probs = self.sampler.sample(
                    self.denoiser, noise, denoiser_args={"cond_prompt": text_embs}, return_logprob=True
                )
                images = latents[-1] / self.vae.scale_factor
                images = self.vae.decode(images)
                images = (images.clamp(-1, 1) + 1) / 2
                rewards = self.reward_fn(images, prompts)
                
                all_text_embs.append(text_embs)
                all_latents.append(torch.stack([noise, *latents[:-1]], dim=1))
                all_next_latents.append(torch.stack(latents, dim=1))
                all_logprobs.append(torch.stack(log_probs, dim=1))
                all_rewards.append(rewards)
                all_images.append(images)

            all_text_embs = torch.cat(all_text_embs, dim=0)    # (batch_size, seq, dim)
            all_latents = torch.cat(all_latents, dim=0)    # (batch_size, num_timesteps, c, h, w)
            all_next_latents = torch.cat(all_next_latents, dim=0)    # (batch_size, num_timesteps, c, h, w)
            all_logprobs = torch.cat(all_logprobs, dim=0)    # (batch_size, num_timesteps)
            all_rewards = torch.cat(all_rewards, dim=0)    # (batch_size,)
            all_timesteps = self.sampler.timesteps.cuda().repeat(batch_size, 1)
            all_images = torch.cat(all_images, dim=0)
            
            # gather all rewards and calculate advantages
            all_advantages = self.rewards_tracker(batch["text"], all_rewards)

            self.loss_meters["reward"].update(all_rewards.mean(), batch_size)
            self.loss_meters["reward_std"].update(all_rewards.std(), batch_size)
        
        if global_step % 1 == 0 and dist.get_rank() == 0:
            dirname = os.path.join(logdir, "log_images", "train")
            os.makedirs(dirname, exist_ok=True)
            log_image_dict = {
                "samples": all_images,
                "texts": batch["text"],
                "fpath": batch["fpath"],
                "rewards": all_rewards,
            }
            self.log_image(dirname, global_step, epoch, log_image_dict)
        
        # training
        # TODO iterate over sample how many times?
        perm = torch.randperm(batch_size, device=self.device)
        all_text_embs = all_text_embs[perm]
        all_latents = all_latents[perm]
        all_next_latents = all_next_latents[perm]
        all_logprobs = all_logprobs[perm]
        all_advantages = all_advantages[perm]
        all_timesteps = all_timesteps[perm]

        perm = torch.stack([
            torch.randperm(self.sampler.num_inference_timesteps, device=self.device)
            for _ in range(batch_size)
        ])
        all_timesteps = all_timesteps[torch.arange(batch_size, device=self.device)[:, None], perm]
        all_latents = all_latents[torch.arange(batch_size, device=self.device)[:, None], perm]
        all_next_latents = all_next_latents[torch.arange(batch_size, device=self.device)[:, None], perm]
        all_logprobs = all_logprobs[torch.arange(batch_size, device=self.device)[:, None], perm]

        mini_batch_size = 4
        grad_acc = 2
        self.unet.train()
        for i in tqdm(range(0, batch_size, mini_batch_size), desc="Training", disable=dist.get_rank() != 0):
            timesteps = all_timesteps[i:i+mini_batch_size]
            text_embs = all_text_embs[i:i+mini_batch_size]    # (mini_batch_size, seq, emb)
            latents = all_latents[i:i+mini_batch_size]    # (mini_batch_size, timesteps, c, h, w)
            next_latents = all_next_latents[i:i+mini_batch_size]
            advantages = all_advantages[i:i+mini_batch_size]    # (mini_batch_size,)
            log_probs_old = all_logprobs[i:i+mini_batch_size]    # (mini_batch_size, num_timesteps)
            advantages = torch.clamp(advantages, -self.loss_config.adv_clip_max, self.loss_config.adv_clip_max)
            
            # TODO grad accumulate
            # TODO cfg
            for j in tqdm(range(self.sampler.num_inference_timesteps), disable=dist.get_rank() != 0):
                should_sync = (i // mini_batch_size + 1) % grad_acc == 0 and j == self.sampler.num_inference_timesteps - 1
                if should_sync:
                    context = contextlib.nullcontext()
                else:
                    context = self.unet.no_sync() 
                with context:
                    with torch.autocast("cuda", torch.bfloat16):
                        denoised = self.denoiser(latents[:, j], timesteps[:, j], cond_prompt=text_embs)
                    _, log_probs = self.sampler.step(denoised, latents[:, j], timesteps[:, j], pred_xtm1=next_latents[:, j], generator=None)
                    ratio = torch.exp(log_probs - log_probs_old[:, j])
                    unclipped_part = - advantages * ratio
                    clipped_part = - advantages * torch.clamp(ratio, 1 - self.loss_config.ppo_clip, 1 + self.loss_config.ppo_clip)
                    loss_ppo = torch.maximum(unclipped_part, clipped_part)
                    loss_ppo = loss_ppo.mean()
                    loss_ppo.backward()
                if shuold_sync:
                    params = itertools.chain(*[x["params"] for x in self.optimizer.param_groups])
                    nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()

                approx_kl = 0.5 * torch.mean((log_probs - log_probs_old[:, j]) ** 2)
                clip_frac = torch.mean((torch.abs(ratio - 1) > self.loss_config.ppo_clip).float())
                self.loss_meters["approx_kl"].update(approx_kl, mini_batch_size)
                self.loss_meters["clip_frac"].update(clip_frac, mini_batch_size)
                self.loss_meters["loss_ppo"].update(loss_ppo, mini_batch_size)
    
    @torch.no_grad()
    def log_step(self, logdir, global_step, epoch, batch_idx):
        logdict = dict()
        for key in self.loss_meters.keys():
            val = self.loss_meters[key].compute()
            logdict[key] = val.item()
            self.loss_meters[key].reset()
        logdict["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        stats = self.rewards_tracker.get_stats()
        for k in stats:
            mean = stats[k]["mean"]
            std = stats[k]["std"]
            logdict[f"rewards/{k}_mean"] = mean
            logdict[f"rewards/{k}_std"] = std
        return logdict

    def log_image(self, logdir, global_step, epoch, log_image_dict):
        images = (log_image_dict["samples"].permute(0, 2, 3, 1) * 255).type(torch.uint8).cpu().numpy()
        prompts = log_image_dict["prompts"]
        rewards = log_image_dict["rewards"]
        for i, (image, prompt, reward) in enumerate(zip(images, prompts, rewards)):
            pil = Image.fromarray(image)
            new_pil = Image.new(pil.mode, (512, 512 + 20), "white")
            new_pil.paste(pil, (0, 0))
            draw = ImageDraw.Draw(new_pil)
            text_position = (10, pil.height)  # 10 pixels padding from bottom of original image
            caption = f"{prompt:.50} | {reward:.2f}"
            draw.text(text_position, caption, fill=(0, 0, 0))
            new_pil.save(os.path.join(logdir, f"gs{global_step}_{i}.png"))