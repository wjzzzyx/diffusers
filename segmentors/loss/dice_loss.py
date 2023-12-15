import torch
import torch.nn as nn


def binary_dice_loss(logits, targets, reduction = 'mean'):
    batch_size = logits.size(0)
    preds = logits.sigmoid().view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    numerator = (preds + targets - torch.abs(preds - targets)).sum(-1)
    denominator = (preds + targets).sum(-1)
    dice = numerator / denominator
    loss = 1 - dice

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()
    

def dice_loss(logits, targets, class_weight = None, reduction = 'mean'):
    batch_size, num_class = logits.size(0), logits.size(1)
    preds = logits.sigmoid().view(batch_size, num_class, -1)
    targets = targets.view(batch_size, num_class, -1)
    numerator = (preds + targets - torch.abs(preds - targets)).sum(-1)
    denominator = (preds + targets).sum(-1)
    multiclass_dice = numerator / denominator
    multiclass_dice *= class_weight
    loss = 1 - multiclass_dice

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


class BinaryDiceLoss(nn.Module):
    "Dice loss that works for soft labels. From https://github.com/zifuwanggg/JDTLosses."
    def __init__(self, reduction = 'mean'):
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, H, W) unnormalized logits output from the model
            targets: (B, H, W) soft class indexes in [0, 1]
        """
        return binary_dice_loss(logits, targets, self.reduction)


class DiceLoss(nn.Module):
    "Multiclass Dice loss that works for soft labels. From https://github.com/zifuwanggg/JDTLosses."
    def __init__(self, class_weight = None, reduction = 'mean'):
        self.class_weight = class_weight
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) unnormalized logits output from the model
            targets: (B, C, H, W) soft class indexes in [0, 1]
        """
        return dice_loss(logits, targets, self.class_weight, self.reduction)


def generalized_dice_loss(logits, targets, reduction = 'mean'):
    batch_size, num_class = logits.size(0), logits.size(1)
    preds = logits.sigmoid().view(batch_size, num_class, -1)
    targets = targets.view(batch_size, num_class, -1)
    weights = 1 / targets.sum(dim=-1) ** 2
    numerator = (preds + targets - torch.abs(preds - targets)).sum(-1)
    denominator = (preds + targets).sum(-1)
    weighted_dice = (weights * numerator).sum(dim=-1) / (weights * denominator).sum(dim=-1)
    loss = 1 - weighted_dice    # shape (B,)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


class GeneralizedDiceLoss(nn.Module):
    "Multi-class Dice loss for imbalanced classes. From https://arxiv.org/pdf/1707.03237.pdf."
    def __init__(self, reduction = 'mean'):
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) unnormalized logits output from the model
            targets: (B, C, H, W) soft class indexes in [0, 1]
        """
        return generalized_dice_loss(logits, targets, self.reduction)