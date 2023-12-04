import math
import numpy as np
from PIL import Image
import random
import shapely.geometry
from typing import Union, List


def instance_in_cropped_region(self, crop_xyxy, instance):
    x1, y1, x2, y2 = crop_xyxy
    crop_region = shapely.geometry.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    if instance.polygon is not None:
        polygon = shapely.geometry.Polygon(instance.polygon)
        iou = polygon.intersection(crop_region).area / polygon.area
        if iou > self.instance_retain_iou:
            return True
        else:
            return False
    elif instance.mask is not None:
        if instance.mask[y1:y2, x1:x2].sum() / instance.mask.sum() > self.instance_retain_iou:
            return True
        else:
            return False
    else:
        return False


class Resize():
    def __init__(self, size: Union[int, List]):
        # size in (width, height)
        if isinstance(size, int):
            size = [size, size]
        self.size = size
    
    def __call__(self, image: Image, instances: List):
        w, h = image.size
        image = image.resize(self.size, resample=Image.BICUBIC)
        scale_w = self.size[0] / w
        scale_h = self.size[1] / h
        for instance in instances:
            if instance.polygon is not None:
                instance.polygon = instance.polygon * np.array([scale_w, scale_h])
            if instance.mask is not None:
                instance.mask = instance.mask.resize(self.size, resample=Image.NEAREST)
        
        return image, instances


class RandomCrop():
    """ Crop an image at a random location with given size. """
    def __init__(self, size: Union[int, List], instance_retain_iou=0.1, pad_if_needed=False):
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.instance_retain_iou = instance_retain_iou
        # the padding will be on the right and bottom if needed
        self.pad_if_needed = pad_if_needed
    
    def __call__(self, image: Image, instances: List):
        w, h = image.size
        if w < self.size[0] or h < self.size[1]:
            if not self.pad_if_needed:
                raise RuntimeError('Image size is smaller than the crop size.')
            else:
                padded = Image.new(image.mode, self.size, (0, 0, 0))
                padded.paste(image, (0, 0))
                image = padded
        
        x1 = random.randint(0, w - self.size[0])
        y1 = random.randint(0, h - self.size[1])
        x2 = x1 + self.size[0]
        y2 = y1 + self.size[0]
        image = image.crop((x1, y1, x2, y2))

        new_instances = list()
        for instance in instances:
            if instance_in_cropped_region((x1, y1, x2, y2), instance):
                if instance.polygon is not None:
                    instance.polygon[:, 0] = np.clip(instance.polygon[:, 0] - x1, 0, self.size[0])
                    instance.polygon[:, 1] = np.clip(instance.polygon[:, 1] - y1, 0, self.size[1])
                if instance.mask is not None:
                    instance.mask = instance.mask[y1:y2, x1:x2]
                new_instances.append(instance)
        
        return image, new_instances


class RandomResizedCrop():
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)):
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, image, instances):
        w, h = image.size
        area = h * w

        # find suitable crop region
        success = False
        for _ in range(10):
            target_area = area * random.uniform(self.scale[0], self.scale[1])
            target_aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            crop_w = int(round(math.sqrt(target_area * target_aspect_ratio)))
            crop_h = int(round(math.sqrt(target_area / target_aspect_ratio)))
            if 0 < crop_w <= w and 0 < crop_h <= h:
                x1 = random.randint(0, w - crop_w)
                y1 = random.randint(0, h - crop_h)
                success = True
                break
        if not success:    # fall back to center crop
            in_ratio = w / h
            if in_ratio < self.ratio[0]:
                crop_w = w
                crop_h = int(round(w / self.ratio[0]))
            elif in_ratio > self.ratio[0]:
                crop_h = h
                crop_w = int(round(h * self.ratio[1]))
            else:    # whole image
                crop_w = w
                crop_h = h
            x1 = (w - crop_w) // 2
            y1 = (h - crop_h) // 2
        
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        image = image.crop((x1, y1, x2, y2))
        image = image.resize(self.size, resample=Image.BICUBIC)
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h

        new_instances = list()
        for instance in instances:
            if instance_in_cropped_region((x1, y1, x2, y2), instance):
                if instance.polygon is not None:
                    instance.polygon[:, 0] = np.clip(instance.polygon[:, 0] - x1, 0, self.size[0])
                    instance.polygon[:, 1] = np.clip(instance.polygon[:, 1] - y1, 0, self.size[1])
                    instance.polygon = instance.polygon * np.array([scale_w, scale_h])
                if instance.mask is not None:
                    instance.mask = instance.mask[y1:y2, x1:x2]
                    instance.mask = instance.mask.resize(self.size, resample=Image.NEAREST)
                new_instances.append(instance)
        
        return image, new_instances
