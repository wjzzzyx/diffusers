import collections
import lightning
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex
from typing import Sequence

from classifiers.model.resnet import build_resnet
from segmentors import metrics


class DeepLabV3(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.backbone = build_resnet('resnet50', replace_stride_with_dilation=[False, True, True], return_feature_only=True)
        self.classifier = DeepLabHead(2048, model_config.num_classes)
        if model_config.use_aux_classifier:
            self.aux_classifier = FCNHead(1024, model_config.num_classes)
        else:
            self.aux_classifier = None
        
        if 'pretrained' in model_config:
            state_dict = torch.load(model_config.pretrained, map_location='cpu')
            state_dict.pop('classifier.4.weight')
            state_dict.pop('classifier.4.bias')
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f'Load pretrained model {model_config.pretrained}')
            print(f'Missing keys: {missing}')
            print(f'Unexpected keys: {unexpected}')
    
    def forward(self, x: torch.Tensor):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = collections.OrderedDict()
        x = features
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result['out'] = x

        if self.aux_classifier is not None:
            x = features['aux']
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result['aux'] = x
        
        return result


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, atrous_rates: Sequence[int] = (12, 24, 36)) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


class PLBase(lightning.LightningModule):

    def __init__(self, model_config, loss_config, optimizer_config, metric_config, ckpt_path=None):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.metric_config = metric_config
    
        self.model = DeepLabV3(model_config)
        self.loss_fn = nn.BCEWithLogitsLoss()
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
        outputs = self.model(batch['image'])
        logits = outputs['out']
        loss = self.loss_fn(logits, batch['mask'].float())
        logdict = {'train/loss_bce': loss.item()}
        self.log_dict(logdict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch['image'].size(0))
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(batch['image'])
        pred_masks = (outputs['out'] > 0).type(torch.uint8)
        label_masks = batch['mask']
        self.metric_ious[dataloader_idx].update(pred_masks, label_masks)
        self.metric_boundary_ious[dataloader_idx].update(pred_masks, label_masks)
        self.log(f'val/iou', self.metric_ious[dataloader_idx])
        self.log(f'val/boundary_iou', self.metric_boundary_ious[dataloader_idx])
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(batch['image'])
        pred_masks = (outputs['out'] > 0).type(torch.uint8)
        label_masks = batch['mask']
        self.metric_ious[dataloader_idx].update(pred_masks, label_masks)
        self.metric_boundary_ious[dataloader_idx].update(pred_masks, label_masks)
        self.log(f'test/iou', self.metric_ious[dataloader_idx])
        self.log(f'test/boundary_iou', self.metric_boundary_ious[dataloader_idx])
    
    def predict_step(self, batch, batch_idx):
        outputs = self.model(batch['image'])
        pred_masks = (outputs['out'] > 0).type(torch.uint8)
        log_image_dict = {
            'image': batch['image'],
            'pred': pred_masks.type(torch.bool),
            'pred_image': torch.cat((batch['image'], pred_masks * 255), dim=1),
            'image_fname': batch['image_fname'],
        }
        self.log_image(log_image_dict, batch_idx, mode='perfect')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
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
            if key in ['image_fname']:
                continue
            image_t = batch[key].permute(0, 2, 3, 1).squeeze(-1)
            image_np = image_t.detach().cpu().numpy()
            for i in range(image_np.shape[0]):
                if 'image_fname' in batch:
                    filename = f"{batch['image_fname'][i][:-4]}_{key}.png"
                else:
                    filename = f"gs-{self.global_step:06}_e-{self.current_epoch:06}_b-{batch_idx:06}_{key}.png"
                Image.fromarray(image_np[i]).save(os.path.join(dirname, filename))