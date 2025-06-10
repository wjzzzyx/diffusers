import collections
import os
from PIL import Image

import einops
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from diffusers.model.flux import AutoEncoder, Flux, sample_training_timestep, sample_inference_timestep
from diffusers.model.lora import LoRAMLP, LoRAConv
import torch_utils
import utils


class FluxLora(nn.Module):
    def __init__(self, flux, config):
        super().__init__()
        self.multiplier = config.multiplier
        self.lora_modules = nn.ModuleDict()
        self.add_lora_modules(
            "double_blocks", flux.double_blocks, config.lora_dim, config.lora_alpha,
            config.dropout_module, config.dropout, config.dropout_rank
        )
        self.add_lora_modules(
            "single_blocks", flux.single_blocks, config.lora_dim, config.lora_alpha,
            config.dropout_module, config.dropout, config.dropout_rank
        )
    
    def add_lora_modules(self, name, module, lora_dim, lora_alpha, dropout_module, dropout, dropout_rank):
        for child_name, child_module in module.named_modules():
            # TODO split qkv?
            if child_module.__class__.__name__ == "Linear":
                lora_name = f"lora_unet.{name}.{child_name}"
                lora_name = lora_name.replace(".", "_")
                self.lora_modules[lora_name] = LoRAMLP(
                    child_module,
                    lora_dim=lora_dim[lora_name] if isinstance(lora_dim, dict) else lora_dim,
                    alpha=lora_alpha[lora_name] if isinstance(lora_alpha, dict) else lora_alpha,
                    dropout_module=dropout_module,
                    dropout=dropout,
                    dropout_rank=dropout_rank
                )
            elif child_module.__class__.__name__ == "Conv2d":
                lora_name = f"lora_unet.{name}.{child_name}"
                lora_name = lora_name.replace(".", "_")
                self.lora_modules[lora_name] = LoRAConv(
                    child_module,
                    lora_dim=lora_dim[lora_name] if isinstance(lora_dim, dict) else lora_dim,
                    alpha=lora_alpha[lora_name] if isinstance(lora_alpha, dict) else lora_alpha,
                    dropout_module=dropout_module,
                    dropout=dropout,
                    dropout_rank=dropout_rank
                )


class Trainer():
    def __init__(self, model_config, loss_config, optimizer_config, device):
        self.device = device
        self.dtype = torch.bfloat16

        flow_model = Flux(model_config.flow)
        if "pretrained" in model_config.flow:
            checkpoint = safetensors.torch.load_file(model_config.flow.pretrained, device="cpu")
            missing, unexpected = flow_model.load_state_dict(checkpoint, strict=False, assign=True)
        flow_model.requires_grad_(False)
        flow_model.to(self.dtype)
        flow_model.cuda()
        self.flow_model = flow_model

        self.ae = AutoEncoder(model_config.ae)
        if "pretrained" in model_config.ae:
            checkpoint = safetensors.torch.load_file(model_config.ae.pretrained, device="cpu")
            missing, unexpected = self.ae.load_state_dict(checkpoint, strict=False, assign=True)
        self.ae.required_grad_(False)
        self.ae.to(self.dtype)
        self.ae.cuda()

        lora_model = FluxLora(flow_model, model_config.lora).to(torch.float32)
        lora_model.cuda()
        self.lora_model = DistributedDataParallel(lora_model, device_ids=[device])

        # prepare optimizers
        self.optimizer = utils.get_obj_from_str(optimizer_config.optimizer)(
            self.lora_model.parameters(), optimizer_config.base_lr
        )
        optimizer_config.lr_scheduler_params.T_max = optimizer_config.num_training_steps
        self.lr_scheduler = utils.get_obj_from_str(optimizer_config.lr_scheduler)(
            self.optimizer, **optimizer_config.lr_scheduler_params
        )

        # prepare metrics
        self.loss_meters = {
            "loss_mse": torch_utils.RunningStatistic(device)
        }

    def on_train_epoch_start(self):
        # TODO should flow model in train state?
        self.flow_model.train()
        self.lora_model.train()
        for key in self.loss_meters:
            self.loss_meters[key].reset()

    def train_step(self, batch, batch_idx, global_step):
        with torch.autocast("cuda", torch.bfloat16):
            latents = batch["latent"].cuda()
            batch_size = latents.size(0)
            t5_emb = batch["text_emb"]["t5_emb"].cuda()
            txt_ids = torch.zeros(t5_emb.size(0), t5_emb.size(1), 3, device=self.device)
            clip_emb = batch["text_emb"]["clip_emb"].cuda()
            noise = torch.randn_like(latents)
            time = sample_training_timestep("sigmoid_normal", batch_size, self.device)
            xt = (1.0 - time.view(-1, 1, 1, 1)) * latents + time.view(-1, 1, 1, 1) * noise
            xt = einops.rearrange(xt, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            h, w = latents.size(2) // 2, latents.size(3) // 2
            img_ids = torch.zeros(h, w, 3, device=self.device)
            img_ids[..., 1] = torch.arange(h, device=self.device)[:, None]
            img_ids[..., 2] = torch.arange(w, device=self.device)[None, :]
            img_ids = img_ids.view(h * w, 3).repeat(batch_size, 1, 1)
            guidance_vec = torch.full((batch_size,), 3.5, device=self.device)
            outputs = self.flow_model(xt, img_ids, t5_emb, txt_ids, y=clip_emb, timesteps=time, guidance=guidance_vec)
            xt = einops.rearrange(xt, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)
            outputs = einops.rearrange(outputs, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)

            target = noise - latents
            loss = F.mse_loss(outputs, target, reduction="mean")

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # self.lr_scheduler.step()
        self.loss_meters["loss_mse"].update(loss.detach(), batch_size)

        return {"xt": xt, "outputs": outputs, "time": time}
    
    def on_train_epoch_end(self, epoch):
        return dict()
    
    def on_val_epoch_start(self):
        self.flow_model.eval()
        self.lora_model.eval()
        self.metrics_dict = collections.defaultdict(list)
    
    @torch.no_grad()
    @torch.autocast("cuda", torch.bfloat16)
    def val_step(self, batch, global_step, epoch, batch_idx, logdir):
        dirname = os.path.join(logdir, "log_images", "val")
        os.makedirs(dirname, exist_ok=True)

        latents = batch["latent"].cuda()
        batch_size = latents.size()
        h, w = latents.size(2) // 2, latents.size(3) // 2
        img_ids = torch.zeros(h, w, 3, device=self.device)
        img_ids[..., 1] = torch.arange(h, device=self.device)[:, None]
        img_ids[..., 2] = torch.arange(w, device=self.device)[None, :]
        img_ids = img_ids.view(h * w, 3).repeat(batch_size, 1, 1)
        t5_emb = batch["text_emb"]["t5_emb"].cuda()
        txt_ids = torch.zeros(t5_emb.size(0), t5_emb.size(1), 3, device=self.device)
        clip_emb = batch["text_emb"]["clip_emb"].cuda()
        noise = torch.randn(latents.size(), generator=torch.Generator(self.device), device=self.device)
        timesteps = torch.linspace(0.9, 0.1, steps=4, device=self.device)
        guidance_vec = torch.full((batch_size,), 3.5, device=self.device)
        preds_alltime = list()
        for time in timesteps:
            xt = (1.0 - time) * latents + time * noise
            xt = einops.rearrange(xt, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            outputs = self.flow_model(xt, img_ids, t5_emb, txt_ids, y=clip_emb, timesteps=time.repeat(batch_size), guidance=guidance_vec)
            outputs = einops.rearrange(outputs, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)
            xt = einops.rearrange(xt, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)
            preds = xt - time * outputs
            preds = self.ae.decode(preds)
            preds = (preds.clamp(-1, 1) + 1) / 2
            preds_alltime.append(preds)
            self.metrics_dict[f"loss_mse_time{time.item():.2f}"].append(
                F.mse_loss(outputs, noise - latents, reduction="mean")
            )
        height, width = preds.size(2), preds.size(3)
        margin = 10
        vis = torch.zeros(batch_size, 3, 2 * (height + margin), 2 * (width + margin), device=self.device)
        for i_time in range(len(timesteps)):
            row = i_time // 2
            col = i_time // 2
            vis[...,
                row * (height + margin) : row * (height + margin) + height,
                col * (width + margin) : col * (width + margin) + width
            ] = preds_alltime[i_time]
        self.log_image(dirname, global_step, epoch, batch["fpath"], preds=vis)
    
    def on_val_epoch_end(self, dataset_name, dataset, logdir):
        for key, val in self.metrics_dict.items():
            self.metrics_dict[key] = sum(val) / len(val)
        return self.metrics_dict

    @torch.no_grad()
    @torch.autocast("cuda", torch.bfloat16)
    def test_step(self, batch, global_step, epoch, batch_idx, logdir):
        t5_emb = batch["text_emb"]["t5_emb"].cuda()
        txt_ids = torch.zeros(t5_emb.size(0), t5_emb.size(1), 3, device=self.device)
        clip_emb = batch["text_emb"]["clip_emb"].cuda()
        batch_size = len(batch["text"])
        width, height = batch["width"][0], batch["height"][0]
        image = torch.randn(
            batch_size, 16, height // 8, width // 8,
            generator=torch.Generator(self.device), device=self.device
        )
        c, h, w = image.shape[1:]
        image = einops.rearrange(image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        img_ids = torch.zeros(h // 2, w // 2, 3, device=self.device)
        img_ids[..., 1] = torch.arange(h // 2, device=self.device)[:, None]
        img_ids[..., 2] = torch.arange(w // 2, device=self.device)[None, :]
        img_ids = img_ids.view(h * w, 3).repeat(batch_size, 1, 1)

        timesteps = sample_inference_timestep(10, image.size(1), shift=False)
        guidance_vec = torch.full((batch_size,), 3.5, device=self.device)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((batch_size,), t_curr, device=self.device)
            pred = self.flow_model(image, img_ids, t5_emb, txt_ids, y=clip_emb, timesteps=t_vec, guidance=guidance_vec)
            image = image + (t_prev - t_curr) * pred
        image = einops.rearrange(image, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h // 2, w=w // 2, ph=2, pw=2)
        image = self.ae.decode(image)
        image = (image.clamp(-1, 1) + 1) / 2

        dirname = os.path.join(logdir, "log_images", "test")
        os.makedirs(dirname, exist_ok=True)
        self.log_image(dirname, global_step, epoch, batch["fname"], preds=image)

    @torch.no_grad()
    def log_step(self, batch, outputs, logdir, global_step, epoch, batch_idx):
        logdict = dict()
        for key in self.loss_meters.keys():
            val = self.loss_meters[key].compute()
            logdict[key] = val.item()
            self.loss_meters[key].reset()
        logdict["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        
        with torch.autocast("cuda", torch.bfloat16):
            pred_x0 = outputs["xt"] + outputs["time"].view(-1, 1, 1, 1) * outputs["outputs"]
            pred = self.ae.decode(pred_x0)
            pred = (pred.clamp(-1, 1) + 1) / 2
        
        dirname = os.path.join(logdir, "log_images", "train")
        os.makedirs(dirname, exist_ok=True)
        self.log_image(dirname, global_step, epoch, batch["fpath"], batch["image"], pred)

        return logdict
    
    def log_image(self, logdir, global_step, epoch, fpaths, images=None, preds=None):
        if images is not None:
            images = (images * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        preds = (preds * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for i in range(len(fpaths)):
            fname = os.path.basename(fpaths[i])
            pred_fname = os.path.splitext(fname)[0] + f"_gs{global_step}_e{epoch}_pred.png"
            if images is not None:
                Image.fromarray(images[i]).save(os.path.join(logdir, fname))
            Image.fromarray(preds[i]).save(os.path.join(logdir, pred_fname))
    
    def get_model_state_dict(self):
        return self.lora_model.module.state_dict()
    
    def get_optimizer_state_dict(self):
        return self.optimizer.state_dict()
    
    def get_lr_scheduler_state_dict(self):
        return self.lr_scheduler.state_dict()