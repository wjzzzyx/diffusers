import cv2
import numpy as np
import os
from PIL import Image
import scipy.ndimage
import skimage.draw
import skimage.segmentation
import skimage.measure
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as TF


class TextInstance():
    def __init__(self, text, orient=None, polygon=None, mask=None, h=None, w=None):
        self.text = text
        self.orient = orient
        self.polygon = polygon    # shape (num_point, 2) xy
        self.mask = mask

        if polygon:
            self.polygon[:, 0] = np.clip(self.polygon[:, 0], 1, w - 1)
            self.polygon[:, 1] = np.clip(self.polygon[:, 1], 1, h - 1)
    
    def get_polygon_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, pts=[self.polygon], color=(1,))
        return mask
    
    def sample_boundary_points(self, h, w, num_points):
        mask = self.get_polygon_mask(h, w)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = contours[0]
        interval = len(contour) / num_points
        sampled_index = [round(i * interval) for i in range(num_points)]
        self.sampled_points = contour[sampled_index]
        return self.sampled_points

    def get_polygon_distance(self, h, w):
        mask = self.get_polygon_mask(h, w)
        return
    
    def get_polygon_direction(self, h, w):
        mask = self.get_polygon_mask(h, w)
        _, labels = cv2.distanceTransformWithLabels(
            mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
        )
        index = labels.copy()
        index[mask > 0] = 0
        place = np.argwhere(index > 0)
        nearCord = place[labels - 1, :]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, h, w))
        nearPixel[0, :, :] = x
        nearPixel[1, :, :] = y
        grid = np.indices(mask.shape)
        grid = grid.astype(float)
        diff = nearPixel - grid
        return diff


class TotalTextDataset(Dataset):
    def __init__(self, data_dir, mode, num_instance, num_points):
        self.data_dir = data_dir
        self.mode = mode
        if mode == 'train':
            self.image_dir = os.path.join(data_dir, 'train_image')
            self.label_dir = os.path.join(data_dir, 'train_label_polygon')
        else:
            self.image_dir = os.path.join(data_dir, 'test_image')
            self.label_dir = os.path.join(data_dir, 'test_label_polygon')

        self.image_fnames = os.listdir(data_dir)
        self.train_transform = 
        self.test_transform = 
        self.num_instance = num_instance
        self.num_points = num_points
    
    def __len__(self):
        return len(self.image_fnames)
    
    def __getitem__(self, index):
        image_fname = self.image_fnames[index]
        label_fname = f'poly_gt_{image_fname[:-4]}.mat'
        image = Image.open(os.path.join(self.image_dir, image_fname))
        label = scipy.io.loadmat(os.path.join(self.label_dir, label_fname))['polygt']

        text_instances = list()
        for cell in label:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            orient = cell[5][0] if len(cell[5]) > 0 else 'c'
            polygon = np.stack([x, y], axis=1).astype(np.int32)
            text_instance = TextInstance(text, orient, polygon=polygon)
            text_instances.append(text_instance)
        
        if self.mode == 'train':
            image, text_instances = self.train_transform(image, text_instances)
        else:
            image, text_instances = self.test_transform(image, text_instances)
        
        w, h = image.size
        
        if self.mode == 'train':
            instance_mask = np.zeros((h, w), dtype=np.uint8)
            distance_map = np.zeros((h, w), dtype=float)
            direction_map = np.zeros((2, h, w), dtype=float)
            gt_points = list()
            instance_valid = list()
            for iinst, instance in enumerate(text_instances):
                cv2.fillPoly(instance_mask, pts=[instance.polygon], color=(iinst + 1,))
            
                bin_mask = instance_mask == iinst + 1
                distance = scipy.ndimage.distance_transform_edt(bin_mask)
                if instance.text == '#' or np.max(distance) < 4 or np.sum(bin_mask) < 150:
                    instance_valid.append(0)
                else:
                    instance_valid.append(1)
                
                distance_map = np.maximum(distance_map, distance / (np.max(distance) + 0.001))

                diff = instance.get_polygon_direction(h, w)
                direction_map[:, bin_mask > 0] = diff[:, bin_mask > 0]

                gt_points.append(instance.sample_boundary_points(h, w, self.num_points))

            gt_points = np.stack(gt_points[:self.num_instance], axis=0)
            instance_valid = np.stack(instance_valid[:self.num_instance], axis=0)
            
            return {
                'image': TF.pil_to_tensor(image),
                'instance_mask': torch.as_tensor(instance_mask),
                'distance_map': torch.as_tensor(distance_map),
                'direction_map': torch.as_tensor(direction_map),
                'gt_points': torch.as_tensor(gt_points),
                'instance_valid': torch.as_tensor(instance_valid),
            }
        
        else:
            gt_points = np.zeros((self.num_instance, self.num_points, 2))
            gt_points_len = np.zeros((self.num_instance), dtype=int)
            instance_valid = np.zeros(self.num_instance, dtype=int)
            for iinst, instance in enumerate(text_instances):
                l = instance.polygon.shape[0]
                gt_points[iinst, :l] = instance.polygon
                gt_points_len[iinst] = l
                if instance.text == '#':
                    instance_valid[iinst] = 0
                else:
                    instance_valid[iinst] = 1

            return {
                'image': TF.pil_to_tensor(image),
                'image_fname': image_fname,
                'gt_points': gt_points,
                'gt_points_len': gt_points_len,
                'instance_valid': instance_valid,
                'height': h,
                'width': w,
            }