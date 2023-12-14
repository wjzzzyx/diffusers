import torch
import torch.nn as nn


def hausdorff_distance_loss(logits, targets, crop=False, crop_margin=0, reduction='mean'):
    preds = logits.sigmoid()
    batch_size, height, width = logits.size()
    loss = list()
    for b in range(batch_size):
        target_coords = torch.nonzero(targets[b])
        if crop:
            x1, x2 = target_coords[:, 1].min(), target_coords[:, 1].max()
            y1, y2 = target_coords[:, 0].min(), target_coords[:, 0].max()
            x1 = max(0, x1 - crop_margin)
            x2 = min(width - 1, x2 + crop_margin)
            y1 = max(0, y1 - crop_margin)
            y2 = min(height - 1, y2 + crop_margin)
            ys, xs = torch.meshgrid(torch.range(y1, y2 + 1), torch.range(x1, x2 + 1), indexing='ij')
            pred_coords = torch.stack((ys, xs), dim=-1).view(-1, 2)
        else:
            ys, xs = torch.meshgrid(torch.range(height), torch.range(width), indexing='ij')
            pred_coords = torch.stack((ys, xs), dim=-1).view(-1, 2)
        
        cropped_pred = preds[b, ys, xs]
        dist = torch.cdist(pred_coords, target_coords)

        hd1 = torch.sum(cropped_pred * dist.min(dim=1)[0]) / torch.sum(cropped_pred)
        pred_at_target = preds[b, target_coords[:, 0], target_coords[:, 1]]
        hd2 = torch.sum(1 - pred_at_target) / len(target_coords)
        hd = hd1 + hd2
        loss.append(hd)
    loss = torch.stack(loss)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


class HausdorffDistanceLoss(nn.Module):
    "Inspiration from paper <Locating Objects Without Bounding Boxes>."
    def __init__(self, crop=False, crop_margin=0, reduction=None):
        self.crop = crop
        self.crop_margin = crop_margin
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, H, W) unnormalized logits output from the model
            targets: (B, H, W) class indexes in {0, 1}
        """
        return hausdorff_distance_loss(logits, targets, self.crop, self.crop_margin, self.reduction)