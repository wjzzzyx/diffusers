import math
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from typing import Sequence


class Compose():
    def __init__(self, augs: Sequence):
        self.augs = augs
    
    def aug(self, batch):
        " The stack of augmentations make in-place changes to batch['image'] "
        for i in range(len(self.augs)):
            batch = self.augs[i].aug(batch)
        return batch
    
    def inference(self, batch, model, *args, **kwargs):
        " We only run the model once at last. "
        batch = self.augs[-1].inference(batch, model, *args, **kwargs)
        return batch
    
    def de_aug(self, batch):
        " The reverse stack of de-augmentations make in-place changes to batch['output'] "
        for i in range(len(self.augs) - 1, -1, -1):
            batch = self.aug[i].de_aug(batch)
        return batch
    
    def __call__(self, batch, model, *args, **kwargs):
        batch = self.aug(batch)
        batch = self.inference(batch, model, *args, **kwargs)
        batch = self.de_aug(batch)
        return batch


class ResizeShortestEdge():
    def __init__(self, size: int, image_key = 'image', output_key = 'output'):
        self.size = size
        self.image_key = image_key
        self.output_key = output_key
    
    def aug(self, batch):
        image = batch[self.image_key]
        self.origin_image = image
        h, w = image.shape[-2:]
        self.origin_size = (h, w)
        scale = self.size / min(h, w)
        if h < w:
            new_h, new_w = self.size, round(w * scale)
        else:
            new_h, new_w = round(h * scale), self.size
        image = F.interpoloate(image.type(torch.float), (new_h, new_w), mode='bilinear')
        batch[self.image_key] = image
        return batch

    def inference(self, batch, model, *args, **kwargs):
        output = model(batch, *args, **kwargs)
        batch[self.output_key] = output
        return batch
    
    def de_aug(self, batch):
        output = F.interpolate(batch['output'], self.origin_size, mode='bilinear')
        batch[self.output_key] = output
        batch[self.image_key] = self.origin_image
        return batch
    
    def __call__(self, batch, model, *args, **kwargs):
        batch = self.aug(batch)
        batch = self.inference(batch, model, *args, **kwargs)
        batch = self.de_aug(batch)
        return batch


class PatchwisePredict():
    def __init__(
        self,
        patch_size: Sequence[int],
        step_size: Sequence[int],
        use_gaussian: bool,
        image_key = 'image',
        output_key = 'output',
    ):
        self.patch_size = patch_size
        self.step_size = step_size
        self.use_gaussian = use_gaussian
        if use_gaussian:
            self._gaussian = self._get_gaussian(patch_size, sigma_scale=1./8)
        self.image_key = image_key
        self.output_keys = output_key
    
    def aug(self, batch):
        return batch
    
    def inference(self, batch, model, *args, **kwargs):
        origin_image = batch.pop(self.image_key)
        batch_size = origin_image.size(0)
        image_size = origin_image.shape[2:]
        num_steps = [math.ceil((i - p) / s) + 1 for i, p, s in zip(image_size, self.patch_size, self.step_size)]
        actual_step_sizes = [(i - p) / (n - 1) if n > 1 else 999 for i, p, n in zip(image_size, self.patch_size, num_steps)]

        if self.use_gaussian:
            gaussian_importance_map = torch.from_numpy(self._gaussian)
            gaussian_importance_map = gaussian_importance_map.to(origin_image.device, origin_image.dtype)
            add_num_out = gaussian_importance_map
        else:
            add_num_out = torch.ones(self.patch_size, device=origin_image.device)
        
        aggregated_out = torch.zeros((batch_size, 1, *image_size), device=origin_image.device)
        aggregated_num_out = torch.zeros((batch_size, 1, *image_size), device=origin_image.device)

        for i in range(num_steps[0]):
            y1 = round(actual_step_sizes[0] * i)
            y2 = min(y1 + self.patch_size[0], image_size[0])
            for j in range(num_steps[1]):
                x1 = round(actual_step_sizes[1] * j)
                x2 = min(x1 + self.patch_size[1], image_size[1])
                batch['image'] = origin_image[:, :, y1:y2, x1:x2]
                out_patch = model(batch, *args, **kwargs)['mask_logits']
                if self.use_gaussian:
                    out_patch *= gaussian_importance_map
                aggregated_out[:, :, y1:y2, x1:x2] += out_patch
                aggregated_num_out[:, :, y1:y2, x1:x2] += add_num_out
        
        aggregated_out /= aggregated_num_out

        batch[self.image_key] = origin_image
        batch[self.output_key] = aggregated_out
        return batch
    
    def de_aug(self, batch):
        return batch

    def __call__(self, batch, model, *args, **kwargs):
        batch = self.aug(batch)
        batch = self.inference(batch, model, *args, **kwargs)
        return self.de_aug(batch)
    
    def _get_gaussian(self, patch_size, sigma):
        center = [i // 2 for i in patch_size]
        sigma = [i * sigma for i in patch_size]
        temp = np.zeros(patch_size)
        temp[center] = 1
        gaussian_importance_map = scipy.ndimage.filters.gaussian_filter(temp, sigma, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map)
        gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[gaussian_importance_map != 0].min()
        return gaussian_importance_map