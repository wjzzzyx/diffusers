import numpy as np
import torch
import torch.nn.functional as F


class LargeScaleJitter():
    """
    Implementation of large scale jitter from copy_paste
    https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """
    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
    
    def __call__(self, sample):
        image, label = sample
        image_size = torch.tensor(image.shape[-2:])

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()
        # the longer side is scaled to scaled_size
        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()

        scaled_image = F.interpolate(image.unsqueeze(0), scaled_size.tolist(), mode='bilinear').squeeze(0)
        scaled_label = F.interpolate(label.unsqueeze(0), scaled_size.tolist(), mode='nearest').squeeze(0)

        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:, crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0, padding_w, 0, padding_h], value=128)
        label = F.pad(scaled_label, [0, padding_w, 0, padding_h], value=0)

        return image, label