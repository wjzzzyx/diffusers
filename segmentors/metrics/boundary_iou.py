import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchmetrics import Metric


class BoundaryJaccardIndex(Metric):
    def __init__(self, dilation_ratio=0.02):
        super().__init__()
        self.dilation_ratio = dilation_ratio
        self.add_state('iou', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')
    
    def update(self, preds, targets):
        preds_bdry = self.mask_to_boundary(preds)
        targets_bdry = self.mask_to_boundary(targets)
        intersection = (preds_bdry * targets_bdry).sum(dim=[1, 2, 3])
        union = (preds_bdry + targets_bdry).sum(dim=[1, 2, 3])
        iou = intersection / union

        self.iou += iou.sum()
        self.total += len(preds)
    
    def compute(self):
        return self.iou / self.total
    
    def mask_to_boundary(self, mask: torch.Tensor):
        h, w = mask.shape[-2:]
        # img_diag = np.sqrt(h ** 2 + w ** 2)
        # dilation = int(round(self.dilation_ratio * img_diag))
        # dilation = 1 if dilation < 1 else dilation
        dilation = 3
        new_mask = mask.type(torch.float)
        kernel = torch.ones((1, 1, 3, 3), device=mask.device)
        for i in range(dilation):
            new_mask = F.conv2d(new_mask, kernel, padding=1)
            new_mask[new_mask < 9] = 0
            new_mask[new_mask == 9] = 1
        return mask - new_mask