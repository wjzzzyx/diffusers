import albumentations as A
import cv2
import math
import numpy as np
from PIL import Image, ImageEnhance
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


class ResizeShortestEdge():
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, image, instances):
        w, h = image.size
        scale = self.size / min(h, w)
        if h < w:
            new_h, new_w = self.size, round(w * scale)
        else:
            new_h, new_w = round(h * scale), self.size
        
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        
        for instance in instances:
            if instance.polygon is not None:
                instance.polygon = instance.polygon * np.array([scale, scale])
            if instance.mask is not None:
                instance.mask = instance.mask.resize((new_w, new_h), resample=Image.NEAREST)
        
        return image, instances


class RandomResizeShortestEdge():
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
    
    def __call__(self, image, instances):
        size = random.randint(self.min_size, self.max_size)

        w, h = image.size
        scale = size / min(h, w)
        if h < w:
            new_h, new_w = size, round(w * scale)
        else:
            new_h, new_w = round(h * scale), size
        
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        
        for instance in instances:
            if instance.polygon is not None:
                instance.polygon = instance.polygon * np.array([scale, scale])
            if instance.mask is not None:
                instance.mask = instance.mask.resize((new_w, new_h), resample=Image.NEAREST)
        
        return image, instances


class ResizeLongestEdge():
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, image, instances):
        w, h = image.size
        scale = self.size / max(h, w)
        if h < w:
            new_h, new_w = round(h * scale), self.size
        else:
            new_h, new_w = self.size, round(w * scale)
        
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        
        for instance in instances:
            if instance.polygon is not None:
                instance.polygon = instance.polygon * np.array([scale, scale])
            if instance.mask is not None:
                instance.mask = instance.mask.resize((new_w, new_h), resample=Image.NEAREST)
        
        return image, instances


class RandomResizeLongestEdge():
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
    
    def __call__(self, image, instances):
        size = random.randint(self.min_size, self.max_size)

        w, h = image.size
        scale = size / max(h, w)
        if h < w:
            new_h, new_w = round(h * scale), size
        else:
            new_h, new_w = size, round(w * scale)
        
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        
        for instance in instances:
            if instance.polygon is not None:
                instance.polygon = instance.polygon * np.array([scale, scale])
            if instance.mask is not None:
                instance.mask = instance.mask.resize((new_w, new_h), resample=Image.NEAREST)
        
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
        y2 = y1 + self.size[1]
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


class CenterCrop():
    def __init__(self, size):
        if isinstance(size, int):
            size = [size, size]
        self.size = size
    
    def __call__(self, image, instances):
        w, h = image.size
        x1 = (w - self.size[0]) // 2
        y1 = (h - self.size[1]) // 2
        x2 = x1 + self.size[0]
        y2 = y1 + self.size[1]

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


class RandomRotation():
    def __init__(self, degree_range=None, degree_std=None):
        self.degree_range = degree_range
        self.degree_std = degree_std
    
    def __call__(self, image, instances):
        if self.degree_range:
            degree = random.uniform(self.degree_range[0], self.degree_range[1])
        else:
            degree = random.gauss(sigma=self.degree_std)
        
        w, h = image.size
        center_x, center_y = w / 2.0, h / 2.0
        rotated_w = round(w * math.fabs(math.cos(degree)) + h * math.fabs(math.sin(degree)))
        rotated_h = round(w * math.fabs(math.sin(degree)) + h * math.fabs(math.cos(degree)))

        matrix = cv2.getRotationMatrix2D((center_x, center_y), degree / math.pi * 180, scale=1)
        matrix[0, 2] += int((rotated_w - w) / 2)
        matrix[1, 2] += int((rotated_h - h) / 2)
        image = cv2.warpAffine(image, matrix, (rotated_w, rotated_h))

        for instance in instances:
            if instance.polygon is not None:
                instance.polygon = self.rotate_points(instance.polygon, degree, (center_x, center_y))
                instance.polygon[:, 0] += int((rotated_w - w) / 2)
                instance.polygon[:, 1] += int((rotated_h - h) / 2)
            if instance.mask is not None:
                instance.mask = cv2.warpAffine(instance.mask, matrix, (rotated_w, rotated_h))
        
        return image, instances
    
    def rotate_points(points, degree, center):
        center_x, center_y = center
        x, y = points[:, 0], points[:, 1]
        center_y = -center_y
        y = -y

        x = x - center_x
        y = y - center_y

        x_rot = x * math.cos(degree) - y * math.sin(degree)
        y_rot = x * math.sin(degree) + y * math.cos(degree)

        x_rot = x_rot + center_x
        y_rot = - (y_rot + center_y)

        return np.stack([x_rot, y_rot], axis=1)


class RandomContrast():
    def __init__(self, min_intensity, max_intensity):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
    
    def __call__(self, image, instances):
        w = random.uniform(self.min_intensity, self.max_intensity)

        image_np = np.array(image)
        has_alpha = False
        if image_np.shape[2] == 4:
            has_alpha = True
            alpha = image_np[:, :, -1]
            image_np = image_np[:, :, :-1]
        image_np = image_np.mean() * (1 - w) + image_np * w
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        if has_alpha:
            image_np = np.concatenate((image_np, alpha[:, :, None]), axis=2)

        return Image.fromarray(image_np), instances


class RandomBrightness():
    def __init__(self, min_intensity, max_intensity):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
    
    def __call__(self, image, instances):
        w = random.uniform(self.min_intensity, self.max_intensity)

        image = ImageEnhance.Brightness(image).enhance(w)

        return image, instances


class RandomLighting():
    def __init__(self, std):
        self.std = std
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])
    
    def __call__(self, image, instances):
        weights = np.random.normal(scale=self.std, size=3)

        image_np = np.array(image)
        image_np = image_np + self.eigen_vecs.dot(weights * self.eigen_vals)
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_np)

        return image, instances


class RandomHueSaturationValue():
    def __init__(self, hue_shift_limit, sat_shift_limit, val_shift_limit, p):
        self.aug = A.HueSaturationValue(hue_shift_limit, sat_shift_limit, val_shift_limit, p=p)
    
    def __call__(self, image, instances):
        image_np = np.array(image)
        image_np = self.aug(image=image_np)['image']
        image = Image.fromarray(image_np)
        return image, instances


class RandomBlur():
    def __init__(self, kernel_size, p):
        self.aug = A.OneOf([
            A.Blur(blur_limit=kernel_size, p=1),
            A.MotionBlur(blur_limit=kernel_size, p=1)
        ], p=p)
    
    def __call__(self, image, instances):
        image_np = np.array(image)
        image_np = self.aug(image=image_np)['image']
        image = Image.fromarray(image_np)
        return image, instances