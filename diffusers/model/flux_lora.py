import os
from PIL import Image

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from diffusers.model.flux import Flux, sample_training_timestep
from diffusers.model.lora import LoRAMLP, LoRAConv
import torch_utils
import utils


class FluxLora(nn.Module):
    def __init__(self, flux, config):
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
        flow_model = Flux(model_config.flow).to(torch.bfloat16)
        flow_model.requires_grad_(False)
        lora_model = FluxLora(flow_model, model_config.lora).to(torch.bfloat16)
        flow_model.cuda()
        lora_model.cuda()
        # self.flow_model = DistributedDataParallel(flow_model, device_ids=[device])
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
            "loss_mse": torch_utils.RunningStatistic()
        }

        self.device = device
    
    def on_train_epoch_start(self):
        self.flow_model.train()
        self.lora_model.train()
        for key in self.loss_meters:
            self.loss_meters[key].reset()

    def train_step(self, batch, batch_idx, global_step):
        latents = batch["latents"].cuda()
        batch_size = latents.size(0)
        t5_emb, txt_ids, clip_emb = batch["text_embs"]
        t5_emb, txt_ids, clip_emb = t5_emb.cuda(), txt_ids.cuda(), clip_emb.cuda()
        noise = torch.randn_like(latents)
        time = sample_training_timestep("sigmoid", batch_size, self.device)
        time = time.view(-1, 1, 1, 1)
        xt = (1.0 - time) * latents + time * noise
        xt = einops.rearrange(xt, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        h, w = latents.size(2) // 2, latents.size(3) // 2
        img_ids = torch.zeros(h, w, 3, device=self.device)
        img_ids[..., 1] = torch.arange(h, device=self.device)[:, None]
        img_ids[..., 2] = torch.arange(w, device=self.device)[None, :]
        img_ids = img_ids.view(h * w, 3).repeat(batch_size, 1, 1)
        guidance_vec = torch.full((batch_size,), 1.0, device=self.device)
        outputs = self.flow_model(xt, img_ids, t5_emb, txt_ids, y=clip_emb, timesteps=time, guidance=guidance_vec)
        outputs = einops.rearrange(outputs, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)

        target = noise - latents
        loss = F.mse_loss(outputs, target, reduction="mean")
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.loss_meters["loss_mse"].update(loss.detach(), batch_size)

        return {"xt": xt, "outputs": outputs, "time": time}
    
    def on_train_epoch_end(self, epoch):
        return dict()
    
    @torch.no_grad()
    def log_step(self, batch, outputs, logdir, global_step, epoch, batch_idx):
        logdict = dict()
        for key in self.loss_meters.keys():
            val = self.loss_meters[key].compute()
            logdict[key] = val.item()
            self.loss_meters[key].reset()
        
        pred_x0 = outputs["xt"] + outputs["time"] * outputs["outputs"]
        pred_x0 = einops.rearrange(pred_x0, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=pred_x0.size(2), w=pred_x0.size(3), ph=2, pw=2)
        pred = self.ae.decode(pred_x0)
        pred = (pred.clamp(-1, 1) + 1) / 2
        dirname = os.path.join(logdir, "log_images", "train")
        os.makedirs(dirname, exist_ok=True)
        self.log_image(dirname, global_step, epoch, batch["fnames"], batch["images"], pred)

        return logdict
    
    def log_image(self, logdir, global_step, epoch, fnames, images, preds):
        images = (images * 255).type("uint8").permute(0, 2, 3, 1).cpu().numpy()
        preds = (preds * 255).type("uint8").permute(0, 2, 3, 1).cpu().numpy()
        for i in range(images.size(0)):
            pred_fname = os.path.splitext(fnames[i])[0] + "_pred.png"
            Image.fromarray(images[i]).save(os.path.join(logdir, fnames[i]))
            Image.fromarray(pred_fname).save(os.path.join(logdir, pred_fname))