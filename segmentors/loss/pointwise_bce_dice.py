import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseBCEDiceLoss(nn.Module):
    def __init__(self, oversample_ratio, importance_sample_ratio):
        super().__init__()
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
    
    def forward(self, pred, target):
        with torch.no_grad():
            point_coords = self.get_uncertain_point_coords_with_randomness(pred, 112 * 112)

            point_labels = self.point_sample(target.float(), point_coords).squeeze(1)
        
        point_logits = self.point_sample(pred, point_coords).squeeze(1)

        loss_bce = self.bce_loss_func(point_logits, point_labels)
        loss_dice = self.dice_loss_func(point_logits, point_labels)
        loss = loss_bce + loss_dice
        logdict = {'loss_bce': loss_bce.item(), 'loss_dice': loss_dice.item()}

        return loss, logdict
    
    def get_uncertain_point_coords_with_randomness(self, logits, num_points):
        """
        Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty.
        See PointRend paper for details.
        Args:
            logits: a tensor of shape (N, C, H, W) or (N, 1, H, W) for class-specific or class-agnostic prediction
            num_points: the number of points to sample
        Returns:
            point_coords: a tensor of shape (N, P, 2) that contains the coordinates of P sampled points
        """
        assert self.oversample_ratio >= 1
        assert 0 <= self.importance_sample_ratio <= 1
        num_boxes = logits.shape[0]
        num_sampled = int(num_points * self.oversample_ratio)
        point_coords = torch.rand(num_boxes, num_sampled, 2, device=logits.device)
        point_logits = self.point_sample(logits, point_coords)
        # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
        # Calculating uncertainties of the coarse predictions first and sampling them for points leads
        # to incorrect results.
        # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
        # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
        # However, if we calculate uncertainties for the coarse predictions first,
        # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
        point_uncertainties = -torch.abs(point_logits)
        num_uncertain_points = int(self.importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(point_uncertainties[:, 0], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
        if num_random_points > 0:
            point_coords = torch.cat(
                [point_coords, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1
            )
        return point_coords
    
    def point_sample(input, point_coords):
        """
        A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
        Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
        [0, 1] x [0, 1] square.
        Args:
            input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
            point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
            [0, 1] x [0, 1] normalized point coordinates.
        Returns:
            output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
                features for points in `point_coords`. The features are obtained via bilinear
                interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
        """
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            point_coords = point_coords.unsqueeze(2)
        output = F.grid_sample(input, 2.0 * point_coords - 1.0, align_corners=False)
        if add_dim:
            output = output.squeeze(3)
        return output

    def bce_loss_func(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return loss.mean()
    
    def dice_loss_func(self, logits, targets):
        preds = logits.sigmoid()
        numerator = 2 * (preds * targets).sum(-1)
        denominator = preds.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()