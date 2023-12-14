import math
import numpy as np
from PIL import Image
import random
import torch
import torch.nn.functional as F
from typing import Union, Sequence


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



class RandomCrop_FGAware():
    "Crop the image at a random location with given size."
    def __init__(self, size: Union[int, Sequence], fg_to_bg_ratio = 1, pad_if_needed = False):
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.fg_to_bg_ratio = fg_to_bg_ratio
        self.pad_if_needed = pad_if_needed
    
    def __call__(self, image: Image, mask: Image):
        width, height = image.size
        if width < self.size[0] or height < self.size[1]:
            if self.pad_if_needed:
                padded = Image.new(image.mode, self.size, (0, 0, 0))
                padded.paste(image, (0, 0))
                image = padded
                padded = Image.new(mask.mode, self.size, 0)
                padded.paste(mask, (0, 0))
                mask = padded
            else:
                raise RuntimeError('Image size is smaller than the crop size.')
        
        mask_np = np.array(mask)
        fg_coords = np.nonzero(mask_np)

        if len(fg_coords[0]) > 0 and random.random() < self.fg_to_bg_ratio / (self.fg_to_bg_ratio + 1):
            crop_fg = True
        else:
            crop_fg = False
        
        # find suitable crop region
        if crop_fg:
            # make sure a random foreground point in the cropped region
            i = random.randint(0, len(fg_coords[0] - 1))
            fg_y, fg_x = fg_coords[0][i], fg_coords[1][i]
            # fg_x < x1 + crop_w <= width, 0 <= x1 <= fg_x
            x1 = random.randint(
                max(0, fg_x - self.size[0] + 1), min(fg_x, width - self.size[0])
            )
            y1 = random.randint(
                max(0, fg_y - self.size[1] + 1), min(fg_y, height - self.size[1])
            )
        else:
            x1 = random.randint(0, width - self.size[0])
            y1 = random.randint(0, height - self.size[1])
        x2 = x1 + self.size[0]
        y2 = y1 + self.size[1]

        image = image.crop((x1, x2, y1, y2))
        mask = mask.crop((x1, x2, y1, y2))

        return image, mask


class RandomResizedCrop_FGAware():
    """
    Foreground aware random resized crop. Randomly crop a region then resize to the target size.
    Can specify foreground / background ratio.
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), fg_to_bg_ratio=1):
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.fg_to_bg_ratio = fg_to_bg_ratio
    
    def __call__(self, image: Image, mask: Image):
        width, height = image.size
        area = height * width

        if isinstance(mask, Image):
            mask_np = np.array(mask)
        fg_coords = np.nonzero(mask_np)

        if len(fg_coords[0]) > 0 and random.random() < self.fg_to_bg_ratio / (self.fg_to_bg_ratio + 1):
            crop_fg = True
        else:
            crop_fg = False

        # find suitable crop region
        success = False
        for _ in range(10):
            target_area = area * random.uniform(self.scale[0], self.scale[1])
            target_aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            crop_w = int(round(math.sqrt(target_area * target_aspect_ratio)))
            crop_h = int(round(math.sqrt(target_area / target_aspect_ratio)))
            if 0 < crop_w <= width and 0 < crop_h <= height:
                if crop_fg:
                    # make sure a random foreground point in the cropped region
                    i = random.randint(0, len(fg_coords[0]) - 1)
                    fg_y, fg_x = fg_coords[0][i], fg_coords[1][i]
                    # fg_x < x1 + crop_w <= width, 0 <= x1 <= fg_x
                    x1 = random.randint(
                        max(0, fg_x - crop_w + 1), min(fg_x, width - crop_w)
                    )
                    y1 = random.randint(
                        max(0, fg_y - crop_h + 1), min(fg_y, height - crop_h)
                    )
                else:
                    x1 = random.randint(0, w - crop_w)
                    y1 = random.randint(0, h - crop_h)
                success = True
                break
        
        if not success:    # fall back to center crop
            in_ratio = width / height
            if in_ratio < self.ratio[0]:
                crop_w = width
                crop_h = int(round(width / self.ratio[0]))
            elif in_ratio > self.ratio[1]:
                crop_h = height
                crop_w = int(round(height * self.ratio[1]))
            else:    # whole image
                crop_h = height
                crop_w = width
            x1 = (width - crop_w) // 2
            y1 = (height - crop_h) // 2
        
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        image = image.crop((x1, y1, x2, y2))
        image = image.resize(self.size, resample=Image.BICUBIC)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        mask = mask.crop((x1, y1, x2, y2))
        mask = mask.resize(self.size, resample=Image.NEAREST)

        return image, mask