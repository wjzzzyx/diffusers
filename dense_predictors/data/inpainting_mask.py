from enum import Enum
import math
import random

import cv2
import numpy as np

import utils

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part


class RandomIrregularMaskGenerator:
    def __init__(
        self, max_angle=4, max_len=60, max_width=20, min_times=1, max_times=10,
        ramp_kwargs=None, draw_method=DrawMethod.LINE, invert_proba = 0.
    ):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None
        self.invert_proba = invert_proba

    def __call__(self, height, width, iter_i=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)

        mask = np.zeros((height, width), np.float32)
        times = random.randint(self.min_times, cur_max_times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(self.max_angle)
                if i % 2 == 0:
                    angle = 2 * math.pi - angle
                length = 10 + np.random.randint(cur_max_len)
                brush_w = 5 + np.random.randint(cur_max_width)
                end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
                end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
                if self.draw_method == DrawMethod.LINE:
                    cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
                elif self.draw_method == DrawMethod.CIRCLE:
                    cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
                elif self.draw_method == DrawMethod.SQUARE:
                    radius = brush_w // 2
                    mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
                start_x, start_y = end_x, end_y
        
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            mask = 1 - mask
        return mask


class RandomRectangleMaskGenerator:
    def __init__(
        self, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3,
        ramp_kwargs=None, invert_proba=0.
    ):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None
        self.invert_proba = invert_proba

    def __call__(self, height, width, iter_i=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)

        mask = np.zeros((height, width), np.float32)
        bbox_max_size = min(cur_bbox_max_size, height - self.margin * 2, width - self.margin * 2)
        times = np.random.randint(self.min_times, cur_max_times + 1)
        for i in range(times):
            box_width = np.random.randint(self.bbox_min_size, bbox_max_size)
            box_height = np.random.randint(self.bbox_min_size, bbox_max_size)
            start_x = np.random.randint(self.margin, width - self.margin - box_width + 1)
            start_y = np.random.randint(self.margin, height - self.margin - box_height + 1)
            mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
        
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            mask = 1 - mask
        return mask


class RandomSuperresMaskGenerator:
    def __init__(self, min_step=2, max_step=4, min_width=1, max_width=3, invert_proba=0.):
        self.min_step = min_step
        self.max_step = max_step
        self.min_width = min_width
        self.max_width = max_width
        self.invert_proba = invert_proba

    def __call__(self, height, width, iter_i=None):
        mask = np.zeros((height, width), np.float32)
        step_x = np.random.randint(self.min_step, self.max_step + 1)
        width_x = np.random.randint(self.min_width, min(step_x, self.max_width + 1))
        offset_x = np.random.randint(0, step_x)

        step_y = np.random.randint(self.min_step, self.max_step + 1)
        width_y = np.random.randint(self.min_width, min(step_y, self.max_width + 1))
        offset_y = np.random.randint(0, step_y)

        for dy in range(width_y):
            mask[offset_y + dy::step_y] = 1
        for dx in range(width_x):
            mask[:, offset_x + dx::step_x] = 1
        
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            mask = 1 - mask
        return mask


class OutpaintingMaskGenerator:
    def __init__(
        self, min_padding_percent:float=0.04, max_padding_percent:int=0.25,
        left_padding_prob:float=0.5, top_padding_prob:float=0.5, 
        right_padding_prob:float=0.5, bottom_padding_prob:float=0.5,
        is_fixed_randomness:bool=False
    ):
        """
        is_fixed_randomness - get identical paddings for the same image if args are the same
        """
        self.min_padding_percent = min_padding_percent
        self.max_padding_percent = max_padding_percent
        self.probs = [left_padding_prob, top_padding_prob, right_padding_prob, bottom_padding_prob]
        self.is_fixed_randomness = is_fixed_randomness

        assert self.min_padding_percent <= self.max_padding_percent
        assert self.max_padding_percent > 0
        assert len([x for x in [self.min_padding_percent, self.max_padding_percent] if (x>=0 and x<=1)]) == 2, f"Padding percentage should be in [0,1]"
        assert sum(self.probs) > 0, f"At least one of the padding probs should be greater than 0 - {self.probs}"
        assert len([x for x in self.probs if (x >= 0) and (x <= 1)]) == 4, f"At least one of padding probs is not in [0,1] - {self.probs}"
        if len([x for x in self.probs if x > 0]) == 1:
            LOGGER.warning(f"Only one padding prob is greater than zero - {self.probs}. That means that the outpainting masks will be always on the same side")

    def apply_padding(self, mask, coord):
        mask[int(coord[0][0]*self.img_h):int(coord[1][0]*self.img_h),   
             int(coord[0][1]*self.img_w):int(coord[1][1]*self.img_w)] = 1
        return mask

    def get_padding(self, size):
        n1 = int(self.min_padding_percent*size)
        n2 = int(self.max_padding_percent*size)
        return self.rnd.randint(n1, n2) / size

    @staticmethod
    def _img2rs(img):
        arr = np.ascontiguousarray(img.astype(np.uint8))
        str_hash = hashlib.sha1(arr).hexdigest()
        res = hash(str_hash)%(2**32)
        return res

    def __call__(self, height, width, iter_i=None, raw_image=None):
        self.img_h, self.img_w = height, width
        mask = np.zeros((self.img_h, self.img_w), np.float32)
        at_least_one_mask_applied = False

        if self.is_fixed_randomness:
            assert raw_image is not None, f"Cant calculate hash on raw_image=None"
            rs = self._img2rs(raw_image)
            self.rnd = np.random.RandomState(rs)
        else:
            self.rnd = np.random

        coords = [[
                   (0,0), 
                   (1,self.get_padding(size=self.img_h))
                  ],
                  [
                    (0,0),
                    (self.get_padding(size=self.img_w),1)
                  ],
                  [
                    (0,1-self.get_padding(size=self.img_h)),
                    (1,1)
                  ],    
                  [
                    (1-self.get_padding(size=self.img_w),0),
                    (1,1)
                  ]]

        for pp, coord in zip(self.probs, coords):
            if self.rnd.random() < pp:
                at_least_one_mask_applied = True
                mask = self.apply_padding(mask=mask, coord=coord)

        if not at_least_one_mask_applied:
            idx = self.rnd.choice(range(len(coords)), p=np.array(self.probs)/sum(self.probs))
            mask = self.apply_padding(mask=mask, coord=coords[idx])
        return mask


class MixedMaskGenerator:
    def __init__(
        self, generator_configs
    ):
        self.probas = []
        self.gens = []
        for generator_config in generator_configs:
            self.gens.append(
                utils.instantiate_from_config(generator_config)
            )
            self.probas.append(generator_config.probability)

    def __call__(self, height, width, iter_i=None):
        gen = random.choices(self.gens, self.probas)[0]
        mask = gen(height, width, iter_i=iter_i)
        return mask
