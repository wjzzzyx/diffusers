from functools import partial
import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex
import lightning
from typing import Any, Type, Optional, Dict, List, Tuple

from .sam import PatchEmbed, Block, LayerNorm2d, MLP, PromptEncoder, TwoWayTransformer
from .sam_hq import ImageEncoderViT
from segmentors import metrics
from segmentors.loss import pointwise_bce_dice, focal_loss


class MaskDecoderNoPrompt(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int = 1024,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a transformer architecture.
        Args:
            transformer_dim: the channel dimension of the transformer
            transformer: the transformer used to predict masks
            num_multimask_outputs: the number of masks to predict when disambiguating masks
            activation: the type of activation to use when upscaling masks
            iou_head_depth: the depth of the MLP used to predict mask quality
            iou_head_hidden_dim: the hidden dimension of the MLP used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.transformer = transformer
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # freeze sam parameters
        self.iou_token.requires_grad_(False)
        self.mask_tokens.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.output_upscaling.requires_grad_(False)
        self.output_hypernetworks_mlps.requires_grad_(False)
        self.iou_prediction_head.requires_grad_(False)

        # HQ SAM parameters
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        # three conv fusion layers for obtaining HQ feature
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
        )
    
    def trainable_parameters(self):
        return (
            list(self.hf_token.parameters())
            + list(self.hf_mlp.parameters())
            + list(self.compress_vit_feat.parameters())
            + list(self.embedding_encoder.parameters())
            + list(self.embedding_maskfeature.parameters())
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        multimask_output: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embeddings: the embeddings from the ViT image encoder
            image_pe: positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings: the embeddings of the points and boxes
            dense_prompt_embeddings: the embeddings of the mask inputs
            multimask_output: whether to return multiple masks or a single mask
        Returns:
            torch.Tensor: batched predicted masks
            torch.Tensor: batched predictions of mask quality
        """
        vit_features = interm_embeddings.permute(0, 3, 1, 2)    # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        masks, iou_preds = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            hq_features=hq_features,
        )

        # select the correct mask of masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            # single mask output, default
            mask_slice = slice(0, 1)
            iou_preds = iou_preds[:, mask_slice]
            masks_sam = masks[:, mask_slice]
        
        masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens)]

        return masks_sam, masks_hq, iou_preds
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        hq_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # concatenate output tokens
        batch_size = image_embeddings.size(0)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = output_tokens

        # expand per-image data in bath direction to be per-mask
        image_pe = torch.repeat_interleave(image_pe, batch_size, dim=0)
        b, c, h, w = image_embeddings.shape

        hs, src = self.transformer(image_embeddings, image_pe, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1+self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features

        hyper_in_list: List[torch.Tensor] = list()
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
        
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:, :self.num_mask_tokens-1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_sam_hq = (hyper_in[:, self.num_mask_tokens-1:] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam, masks_sam_hq], dim=1)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class SamNoPrompt(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = 'RGB'

    def __init__(
        self,
        model_config,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.model_config = model_config
        image_embedding_size = model_config.image_size // model_config.vit_patch_size
        self.image_encoder = ImageEncoderViT(
            depth=model_config.encoder_depth,
            embed_dim=model_config.encoder_embed_dim,
            img_size=model_config.image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=model_config.encoder_num_heads,
            patch_size=model_config.vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=model_config.encoder_global_attn_indexes,
            window_size=14,
            out_chans=model_config.prompt_embed_dim,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=model_config.prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(model_config.image_size, model_config.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoderNoPrompt(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=model_config.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=model_config.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=model_config.encoder_embed_dim,
        )
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.image_encoder.requires_grad_(False)
        self.prompt_encoder.requires_grad_(False)

        if 'pretrained' in model_config:
            state_dict = torch.load(model_config.pretrained, map_location='cpu')
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f'Load pretrained model {model_config.pretrained}')
            print(f'Missing keys: {missing}')
            print(f'Unexpected keys: {unexpected}')
        
    @ property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def trainable_parameters(self):
        return self.mask_decoder.trainable_parameters()
    
    def forward(
        self,
        feeddict: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Args:

        """
        input_images = self.pad_if_needed(self.normalize(feeddict['image']))
        image_embeddings, interm_embeddings = self.image_encoder(input_images)
        interm_embeddings = interm_embeddings[0]

        low_res_logits_sam, low_res_logits_hq, iou_preds = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            multimask_output=multimask_output,
            interm_embeddings=interm_embeddings,
        )
        logits = F.interpolate(
            low_res_logits_hq,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode='bilinear',
            align_corners=False,
        )
        return {
            'mask_logits': logits,
            'iou_preds': iou_preds,
            'low_res_logits_sam': low_res_logits_sam,
            'low_res_logits_hq': low_res_logits_hq,
        }
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        return x
    
    def pad_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        assert(h <= self.image_encoder.img_size and w <= self.image_encoder.img_size)
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        if padh > 0 or padw > 0:
            x = F.pad(x, (0, padw, 0, padh))
        return x


class PLBase(lightning.LightningModule):
    def __init__(self, model_config, loss_config, optimizer_config, metric_config, ckpt_path=None):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.metric_config = metric_config

        self.model = SamNoPrompt(model_config)
        self.loss_fn1 = pointwise_bce_dice.PointwiseBCEDiceLoss(
            loss_config.oversample_ratio, loss_config.importance_sample_ratio
        )
        self.loss_fn2 = focal_loss.BinaryFocalLoss(
            loss_config.gamma, loss_config.pos_weight
        )
        self.metric_ious = nn.ModuleList([BinaryJaccardIndex(threshold=0.5) for _ in range(metric_config.num_valsets)])
        self.metric_boundary_ious = nn.ModuleList([metrics.BoundaryJaccardIndex() for _ in range(metric_config.num_valsets)])

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
    
    def init_from_ckpt(self, path):
        ckpt = torch.load(path, map_location='cpu')
        missing, unexpected = self.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'Restored from checkpoint {ckpt}')
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(batch, multimask_output=False)
        pred_masks = outputs['low_res_logits_hq']
        label_masks = batch['mask']
        loss1, logdict1 = self.loss_fn1(pred_masks, label_masks)
        loss2, logdict2 = self.loss_fn2(outputs['mask_logits'], label_masks)
        loss = loss1 + loss2 * self.loss_config.weight_focal
        logdict = logdict1 | logdict2
        logdict = {f'train/{k}': v for k, v in logdict.items()}
        self.log_dict(logdict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch['image'].size(0))

        if self.global_step % 100 == 0 and self.global_rank == 0:
            self.log_image(
                {'image': batch['image'], 'mask': batch['mask'].type(torch.bool), 'pred': (outputs['mask_logits'] > 0)},
                batch_idx,
                mode='train'
            )

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(batch, multimask_output=False)
        pred_masks = (outputs['mask_logits'] > 0).type(torch.uint8)
        label_masks = batch['mask']
        self.metric_ious[dataloader_idx].update(pred_masks, label_masks)
        self.metric_boundary_ious[dataloader_idx].update(pred_masks, label_masks)
        self.log('val/iou', self.metric_ious[dataloader_idx])
        self.log('val/boundary_iou', self.metric_boundary_ious[dataloader_idx])
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(batch, multimask_output=False)
        pred_masks = (outputs['mask_logits'] > 0).type(torch.uint8)
        label_masks = batch['mask']
        self.metric_ious[dataloader_idx].update(pred_masks, label_masks)
        self.metric_boundary_ious[dataloader_idx].update(pred_masks, label_masks)
        self.log('test/iou', self.metric_ious[dataloader_idx])
        self.log('test/boundary_iou', self.metric_boundary_ious[dataloader_idx])

        if batch_idx % 10 == 0 and self.global_rank == 0:
            self.log_image(
                {'image': batch['image'], 'mask': batch['mask'].type(torch.bool), 'pred': pred_masks.type(torch.bool)},
                batch_idx,
                mode='test'
            )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.trainable_parameters(),
            lr=self.optimizer_config.learning_rate,
            betas=(self.optimizer_config.adam_beta1, self.optimizer_config.adam_beta2),
            weight_decay=self.optimizer_config.adam_weight_decay,
            eps=self.optimizer_config.adam_epsilon,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            self.optimizer_config.lr_drop_epoch,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
            }
        }

    @torch.no_grad()
    def log_image(self, batch, batch_idx, mode):
        dirname = os.path.join(self.logger.save_dir, 'log_images', mode)
        os.makedirs(dirname, exist_ok=True)
        for key in batch.keys():
            image_t = batch[key].permute(0, 2, 3, 1).squeeze(-1)
            image_np = image_t.detach().cpu().numpy()
            for i in range(image_np.shape[0]):
                filename = f'gs-{self.global_step:06}_e-{self.current_epoch:06}_b-{batch_idx:06}-{i:02}_{key}.png'
                Image.fromarray(image_np[i]).save(os.path.join(dirname, filename))