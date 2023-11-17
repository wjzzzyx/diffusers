import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
    weight: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Args:
        logits: (B, C, H, W)
        targets: (B, H, W)
        alpha: (C)
        gamma:
        weight: (C)
        reduction:
    Returns:
        loss
    """
    eps = 1e-4
    prob = F.softmax(logits.permute(0, 2, 3, 1), dim=-1)
    prob_gt = prob[targets]    # (B, H, W)
    logprob_gt = torch.log(prob + eps)
    # alpha_gt = alpha[targets]    # (B, H, W)
    if weight is None:
        weight = torch.ones(targets.shape, dtype=logits.dtype, device=logits.device)
    else:
        weight = weight[targets]
    loss = -alpha * torch.pow(1 - prob_gt, gamma) * logprob_gt

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss


class FocalLoss(nn.Module):
    """
    focal_loss = -alpha * (1 - p)^gamma * logp
    """
    def __init__(self, alpha: float, gamma: float, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        return focal_loss(logits, targets, weight=self.weight, reduction=self.reduction)


if __name__ == '__main__':
    focalloss = FocalLoss(0.25, 2)
    logits = torch.tensor([
        [
            [[1., 2., 3.],
             [1., 2., 3.],
             [1., 2., 3.]],

            [[0., 1., 2.],
             [0., 1., 2.],
             [0., 1., 2.]],
        ]
    ])
    targets = torch.tensor([
        [[0, 1, 2],
         [1, 2, 0],
         [2, 0, 1]]
    ])
    loss = focalloss(logits, targets)