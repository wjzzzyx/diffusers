import lightning
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex

from classifiers.model.resnet import build_resnet
from segmentors import metrics


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024):
        super().__init__()
        self.feats = build_resnet('resnet50', return_feature_only=True)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )

    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        # auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(p)# , self.classifier(auxiliary)


class PLBase(lightning.LightningModule):

    def __init__(self, model_config, loss_config, optimizer_config, metric_config, ckpt_path=None):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.metric_config = metric_config
    
        self.model = PSPNet(model_config)
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
        logdict = {'train/loss_ce': loss.item()}
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