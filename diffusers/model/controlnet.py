import os
import collections
import numpy as np
from PIL import Image
import pyiqa
import skimage.segmentation
import safetensors
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from diffusers.model.stable_diffusion_stabilityai import (
    StableDiffusion_StabilityAI, timestep_embedding, UNetModel, TimestepEmbedSequential, ResBlock,
    Downsample, SpatialTransformer, AttentionBlock, conv_nd, zero_module
)
import torch_utils
import utils


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps, context=None, control=None, only_mid_control=False):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNetEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ])
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if disable_self_attentions is not None:
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if num_attention_blocks is None or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlNet(StableDiffusion_StabilityAI):
    def __init__(self, model_config):
        super().__init__(model_config)
        # freeze ae and unet
        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)

        self.control_stage_model = ControlNetEncoder(**model_config.control_stage_config)
        self.control_scales = [1.0] * (len(self.control_stage_model.input_blocks) + 1)
        if 'pretrained_cn' in model_config:
            if model_config.pretrained_cn.endswith('safetensors'):
                checkpoint = safetensors.torch.load_file(model_config.pretrained_cn, device='cpu')
            else:
                checkpoint = torch.load(model_config.pretrained_cn, map_location='cpu')
            missing, unexpected = self.control_stage_model.load_state_dict(checkpoint)

    def forward_diffusion_model(self, xt, t, cond_prompt, cond_control):
        control_hs = self.control_stage_model(xt, cond_control, t, context=cond_prompt)
        control_hs = [c * scale for c, scale in zip(control_hs, self.control_scales)]
        output = self.diffusion_model(xt, t, context=cond_prompt, control=control_hs)
        return output


class Trainer():
    def __init__(self, model_config, loss_config, optimizer_config, device):
        self.device = device
        
        self.unet = utils.instantiate_from_config(model_config.unet_config).cuda()
        self.unet.eval()
        self.unet.requires_grad_(False)
        self.vae = utils.instantiate_from_config(model_config.first_stage_config).cuda()
        self.vae.eval()
        self.vae.requires_grad_(False)
        clip = utils.instantiate_from_config(model_config.cond_stage_config).cuda()
        self.cond_prompt = clip("best quality, high resolution")
        del clip
        controlnet = ControlNetEncoder(**model_config.control_stage_config).cuda()
        self.controlnet = DistributedDataParallel(controlnet, device_ids=[device])
        self.sampler = utils.instantiate_from_config(model_config.sampler_config)
        self.control_scales = [1.0] * (len(controlnet.input_blocks) + 1)

        # prepare loss
        self.diffusion_loss_fn = utils.instantiate_from_config(loss_config)

        # prepare optimizer
        self.optimizer = utils.get_obj_from_str(optimizer_config.optimizer)(
            self.controlnet.parameters(), optimizer_config.base_lr
        )
        optimizer_config.lr_scheduler_params.T_max = optimizer_config.num_training_steps
        self.lr_scheduler = utils.get_obj_from_str(optimizer_config.lr_scheduler)(
            self.optimizer, **optimizer_config.lr_scheduler_params
        )

        # prepare metrics
        self.loss_meters = {
            "loss_mse": torch_utils.RunningStatistic(device),
            "loss_adv_g": torch_utils.RunningStatistic(device),
            "loss_adv_d": torch_utils.RunningStatistic(device),
        }
    
    def on_train_epoch_start(self):
        self.controlnet.train()
        for key in self.loss_meters:
            self.loss_meters[key].reset()

    def train_step(self, batch, global_step, epoch, batch_idx, logdir):
        batch_size = batch['image'].shape[0]
        images = batch['image'].cuda()
        masks = batch['mask'].cuda()
        masked_images = images * (1 - masks)
        cond_images = torch.cat([masked_images, masks], dim=1)
        with torch.no_grad():
            latents = self.vae.encode(images).sample()
        noise = torch.randn_like(latents)
        time = torch.randint(0, self.sampler.num_train_timesteps, (images.shape[0],), device=self.device)
        alphas_cumprod_t = self.sampler.alphas_cumprod[time][(...,) + (None,) * 3]
        xt = torch.sqrt(alphas_cumprod_t) * latents + torch.sqrt(1 - alphas_cumprod_t) * noise
        cond_feats = self.controlnet(xt, cond_images, time, self.cond_prompt)
        output = self.unet(xt, time, context=self.cond_prompt, control=cond_feats)
        loss_l2 = self.diffusion_loss_fn(output, xt, noise, time)
        loss_l2.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.loss_meters["loss_l2"].update(loss_l2.item(), batch_size)

        if global_step % 1000 == 0 and dist.get_rank() == 0:
            dirname = os.path.join(logdir, "log_images", "train")
            os.makedirs(dirname, exist_ok=True)
            with torch.no_grad():
                pred_x0 = xt + time.view(-1, 1, 1, 1) * output
                pred_x0 = pred_x0 / self.vae.scale_factor
                pred = self.vae.decode(pred_x0)
                pred = (pred.clamp(-1, 1) + 1) / 2
            log_image_dict = {
                "image": images,
                "mask": masks,
                "pred": pred,
                "fpath": batch["fpath"]
            }
            self.log_image(dirname, global_step, epoch, log_image_dict)
    
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
        self.controlnet.eval()
        self.val_metrics = collections.defaultdict(list)
        self.psnr = pyiqa.create_metric('psnr')
        self.ssim = pyiqa.create_metric('ssim')
        self.lpips = pyiqa.create_metric('lpips', version='0.1', net='vgg', device=self.device)
        self.fid = pyiqa.create_metric("fid", device=self.device)

    @torch.no_grad()
    def val_step(self, batch, global_step, epoch, batch_idx, logdir):
        images = batch['image'].cuda()
        masks = batch['mask'].cuda()
        masked_images = images * (1 - masks)
        cond_images = torch.cat([masked_images, masks], dim=1)
        latents = self.vae.encode(images).sample()
        latents = self.vae.scale_factor * latents
        generator = torch.Generator(self.device).manual_seed(0)
        noise = torch.randn(latents.shape, device=self.device, generator=generator)

        preds = self.sampler.sample(
            self.denoiser,
            noise,
            denoiser_args={"cond_prompt": self.cond_prompt, "cond_control": cond_images},
            generator=generator
        )
        preds = preds / self.vae.scale_factor
        preds = self.vae.decode(preds)
        preds = (preds.clamp(-1, 1) + 1) / 2
        preds_np = (preds * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for pred_np, fpath in zip(preds_np, batch["fpath"]):
            fname = os.path.basename(fpath)
            Image.fromarray(pred_np).save(os.path.join(logdir, "val_preds", fname))
        
        self.val_metrics["psnr"].append(self.psnr(preds, images))
        self.val_metrics["ssim"].append(self.ssim(preds, images))
        self.val_metrics["lpips"].append(self.lpips(preds, images))

        if batch_idx % 100 == 0:
            dirname = os.path.join(logdir, "log_images", "val")
            os.makedirs(dirname, exist_ok=True)
            log_image_dict = {
                "image": images,
                "mask": masks,
                "pred": preds,
                "fpath": batch["fpath"]
            }
            self.log_image(dirname, global_step, epoch, log_image_dict)
    
    def denoiser(self, xt, sigma, cond_prompt, cond_control):
        def forward_diffusion_model(xt, t, cond_prompt, cond_control):
            control_hs = self.controlnet(xt, cond_control, t, context=cond_prompt)
            control_hs = [c * scale for c, scale in zip(control_hs, self.control_scales)]
            output = self.unet(xt, t, context=cond_prompt, control=control_hs)
            return output

        sigma_data = 1.0
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
        eps = forward_diffusion_model(
            xt * c_in[..., None, None, None], self.sampler.sigma_to_t(sigma), cond_prompt, cond_control
        )
        return xt + eps * c_out[..., None, None, None]
    
    def on_val_epoch_end(self, dataset_name, dataset, logdir):
        for key, val in self.val_metrics.items():
            self.val_metrics[key] = sum(val) / len(val)
        self.lpips.cpu()
        self.val_metrics["fid"] = self.fid(
            os.path.join(logdir, "val_preds"), os.path.join(dataset.root_dir, "val", "images")
        )
        self.fid.cpu()
        return self.val_metrics
    
    def log_image(self, logdir, global_step, epoch, log_image_dict):
        images = (log_image_dict["image"] * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        masks = (log_image_dict["mask"] * 255).type(torch.uint8).cpu().numpy()
        preds = (log_image_dict["pred"] * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        fpaths = log_image_dict["fpath"]
        margin = 4
        for i in range(len(fpaths)):
            fname = os.path.basename(fpaths[i])
            pred_fname = os.path.splitext(fname)[0] + f"_gs{global_step}_e{epoch}_pred.png"
            image = skimage.segmentation.mark_boundaries(images[i], masks[i], color=(1., 0., 0.), mode='thick')
            pred = skimage.segmentation.mark_boundaries(preds[i], masks[i], color=(1., 0., 0.), mode='thick')
            vis = np.zeros((image.shape[0], image.shape[1] + margin + pred.shape[1], 3), dtype=np.uint8)
            vis[:, :image.shape[1]] = image
            vis[:, -pred.shape[1]:] = pred
            Image.fromarray(vis).save(os.path.join(logdir, pred_fname))