import torch
import torch.nn as nn


def binary_iou_loss(logits, targets, reduction = 'mean'):
    batch_size = logits.size(0)
    preds = logits.sigmoid().view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    numerator = (preds + targets - torch.abs(preds - targets)).sum(-1)
    denominator = (preds + targets + torch.abs(preds - targets)).sum(-1)
    iou = numerator / denominator
    loss = 1 - iou

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


def iou_loss(logits, targets, class_weight = None, reduction = 'mean'):
    batch_size, num_class = logits.size(0), logits.size(1)
    preds = logits.sigmoid().view(batch_size, num_class, -1)
    targets = targets.view(batch_size, num_class, -1)
    numerator = (preds + targets - torch.abs(preds - targets)).sum(-1)
    denominator = (preds + targets + torch.abs(preds - targets)).sum(-1)
    multiclass_iou = numerator / denominator
    loss = 1 - multiclass_iou

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


class BinaryIoULoss(nn.Module):
    "IoU loss that works for soft labels. According to https://github.com/zifuwanggg/JDTLosses."
    def __init__(self, reduction = 'mean'):
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, H, W) unnormalized logits output from the model
            targets: (B, H, W) soft class indexes in [0, 1]
        """
        return binary_iou_loss(logits, targets, self.reduction)


class IoULoss(nn.Module):
    "Multiclass IoU loss that works for soft labels. According to https://github.com/zifuwanggg/JDTLosses."
    def __init__(self, class_weight = None, reduction = 'mean'):
        self.class_weight = class_weight
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) unnormalized logits output from the model
            targets: (B, C, H, W) soft class indexes in [0, 1]
        """
        return iou_loss(logits, targets, self.class_weight, self.reduction)