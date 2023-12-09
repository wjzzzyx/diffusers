import torch
import torch.nn as nn


def binary_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Args: 
        logits: (B, H, W) unnormalized positive logits output from the model
        targets: (B, H, W) class index in {0, 1}
    """
    batch_size = logits.size(0)
    preds = torch.sigmoid(logits).view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    true_pos = torch.sum(preds * targets, dim=1)
    false_pos = torch.sum(preds * (1 - targets), dim=1)
    false_neg = torch.sum((1 - preds) * targets, dim=1)

    loss = 1. - true_pos / (true_pos + alpha * false_pos + beta * false_neg + 0.1)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


class BinaryTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, reduction='mean'):
        """
        Args:
            alpha: controls the penalty for false positives. Larger alpha, fewer false positives.
            beta: controls the penalty for false negatives. Larger beta, fewer false negatives.
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, logits, targets):
        return binary_tversky_loss(logits, targets, self.alpha, self.beta, self.reduction)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, weight=None, reduction='mean'):
        """
        Args:
            alpha: controls the penalty for false positives. Larger alpha, fewer false positives.
            beta: controls the penalty for false negatives. Larger beta, fewer false negatives.
            weight: (C) one-dim tensor, weight for each class.
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        num_class = logits.size(1)
        loss = list()
        for c in range(num_class):
            loss.append(binary_tversky_loss(logits[:, c], (targets == c), self.alpha, self.beta, self.reduction))
        loss = torch.stack(loss, dim=1)
        loss = loss * self.weight

        if self.reduction == 'none':
            return loss    # (B,)
        elif self.reduction =='sum':
            return loss.sum()
        else:
            return loss.mean()