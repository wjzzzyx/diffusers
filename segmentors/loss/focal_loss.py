import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    weight: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Args:
        logits: (B, C, H, W) unnormalized logits output from the model
        targets: (B, H, W) class indexes in the range [0, C)
        gamma: exponent
        weight: (C) one-dim tensor, weight for each class
        reduction: choices in ('mean', 'sum', 'none')
    Returns:
        loss
    """
    eps = 1e-4
    prob = F.softmax(logits, dim=1)
    prob_gt = torch.gather(prob, 1, targets.unsqueeze(1)).squeeze(1)    # (B, H, W)
    logprob_gt = torch.log(prob_gt + eps)
    if weight is None:
        weight = torch.ones(targets.shape, dtype=logits.dtype, device=logits.device)
    else:
        weight = weight[targets]
    loss = - weight * torch.pow(1 - prob_gt, gamma) * logprob_gt

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f'Unsupported reduction {reduction}')


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    pos_weight: float = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Args:
        logits: (B, H, W) unnormalized positive logits output from the model
        targets: (B, H, W) class indexes in {0, 1}
        gamma: exponent
        pos_weight: weight for the positive class between [0, 1]
        reduction: choices in ('mean', 'sum', 'none')
    """
    eps = 1e-4
    prob = F.sigmoid(logits)
    prob = torch.clamp(prob, eps, 1.0 - eps)

    pos_loss = - pos_weight * torch.pow(1 - prob, gamma).detach() * F.logsigmoid(logits)
    neg_loss = - (1 - pos_weight) * torch.pow(prob, gamma).detach() * F.logsigmoid(-logits)
    loss = torch.where(targets, pos_loss, neg_loss)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f'Unsupported reduction {reduction}')


class FocalLoss(nn.Module):
    """
    focal_loss = - weight * (1 - p)^gamma * logp
    """
    def __init__(self, gamma: float, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.gamma = gamma
    
    def forward(self, logits, targets):
        return focal_loss(
            logits, targets, gamma=self.gamma,
            weight=self.weight, reduction=self.reduction
        )


class BinaryFocalLoss(nn.Module):
    """
    binary_focal_loss = - pos_weight * (1 - p)^gamma * logp - (1 - pos_weight) * p^gamma * log(1 - p)
    """
    def __init__(self, gamma: float, pos_weight: float = None, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.gamma = gamma
    
    def forward(self, logits, targets):
        return binary_focal_loss(
            logits, targets, gamma=self.gamma,
            pos_weight=self.pos_weight, reduction=self.reduction
        )


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