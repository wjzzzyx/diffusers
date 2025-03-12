from typing import List

import scipy.optimize
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from segmentors.loss.pointwise_bce_dice import point_sample, get_uncertain_point_coords_with_randomness


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bce: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bce = cost_bce
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_bce != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, class_logits, mask_logits, gt_classes, gt_masks):
        """More memory-friendly matching"""
        bs, num_queries = class_logits.shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = class_logits[b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = gt_classes[b]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = mask_logits[b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = gt_masks[b].to(out_mask)

            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask.unsqueeze(1),
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask.unsqueeze(1),
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with torch.amp.autocast("cuda", enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_bce = self.get_bce_cost_matrix(out_mask, tgt_mask)
                # Compute the dice loss betwen masks
                cost_dice = self.get_dice_cost_matrix(out_mask, tgt_mask)
            
            # Final cost matrix
            cost = self.cost_class * cost_class + self.cost_bce * cost_bce + self.cost_dice * cost_dice
            cost = cost.cpu().numpy()
            # linear_sum_assignment returns arryas of row_ind and col_ind
            # in this case, the matched indexes of query and indexes of ground truth
            indices.append(scipy.optimize.linear_sum_assignment(cost))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, class_logits, mask_logits, gt_classes, gt_masks):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(class_logits, mask_logits, gt_classes, gt_masks)
    
    def get_bce_cost_matrix(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: shape (num_queries, num_points)
            targets: shape (num_masks, num_points), value in {0, 1}
        Returns:
            loss: shape (num_queries, num_masks)
        """
        hw = inputs.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )
        loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
        loss = loss / hw

        return loss

    def get_dice_cost_matrix(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: shape (num_queries, num_points)
            targets: shape (num_masks, num_points), value in {0, 1}
        Returns:
            loss: shape (num_queries, num_masks)
        """
        inputs = inputs.sigmoid()
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return (tensor, mask)


class SetClassSegmentLoss(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
        self, num_classes, weight_bg, weight_class, weight_bce, weight_dice,
        num_points, oversample_ratio, importance_sample_ratio
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        class_weights = torch.ones(num_classes + 1)
        class_weights[-1] = weight_bg
        self.register_buffer("class_weights", class_weights)

        self.matcher = HungarianMatcher(
            cost_class=weight_class, cost_bce=weight_bce, cost_dice=weight_dice, num_points=num_points
        )
    
    def forward(self, pred_logits, pred_masks, gt_classes, gt_masks, aux_outputs = None):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t) for t in gt_classes)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=pred_logits.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        num_masks = torch.clamp(num_masks / world_size, min=1).item()

        indices = self.matcher(pred_logits, pred_masks, gt_classes, gt_masks)
        loss_class = self.loss_labels(pred_logits, gt_classes, indices, num_masks)
        loss_bce, loss_dice = self.loss_masks(pred_masks, gt_masks, indices, num_masks)
        losses = {
            "loss_class": self.weight_class * loss_class,
            "loss_bce": self.weight_class * loss_bce,
            "loss_dice": self.weight_dice * loss_dice
        }

        if aux_outputs is not None:
            for i, out_i in enumerate(aux_outputs):
                indices = self.matcher(out_i["pred_logits"], out_i["pred_masks"], gt_classes, gt_masks)
                loss_class = self.loss_labels(out_i["pred_logits"], gt_classes, indices, num_masks)
                loss_bce, loss_dice = self.loss_masks(out_i["pred_masks"], gt_masks, indices, num_masks)
                losses[f"loss_class_{i}"] = self.weight_class * loss_class
                losses[f"loss_bce_{i}"] = self.weight_bce * loss_bce
                losses[f"loss_dice_{i}"] = self.weight_dice * loss_dice

        return losses

    def loss_labels(self, pred_logits, gt_classes, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        batch_size, num_queries = pred_logits.shape[:2]
        src_logits = pred_logits.float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(gt_classes, indices)])
        target_classes = torch.full(
            (batch_size, num_queries), self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return loss_ce
    
    def loss_masks(self, pred_masks, gt_masks, indices, num_masks):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # pred_masks: shape (batch, query, h, w)
        src_masks = pred_masks[src_idx]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(gt_masks)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.unsqueeze(1),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks.unsqueeze(1),
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks.unsqueeze(1),
            point_coords,
            align_corners=False,
        ).squeeze(1)

        loss_bce = self.sigmoid_ce_loss(point_logits, point_labels, num_masks)
        loss_dice = self.dice_loss(point_logits, point_labels, num_masks)

        del src_masks
        del target_masks
        return loss_bce, loss_dice

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def sigmoid_ce_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        return loss.mean(1).sum() / num_masks

    def dice_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)