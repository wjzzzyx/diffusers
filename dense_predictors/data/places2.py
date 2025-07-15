import glob
import os
from PIL import Image

import albumentations as A
import numpy as np
import torch
import torchvision.transforms.v2.functional as TF

import utils


class Places2Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, target_size, maskgen_config=None):
        self.root_dir = root_dir
        self.mode = mode
        if mode == "train":
            self.fpaths = glob.glob(os.path.join(root_dir, "train", "**", "*.jpg"), recursive=True)
            self.mask_generator = utils.instantiate_from_config(maskgen_config)
            self.transform = A.Compose([
                A.Perspective(scale=(0.0, 0.06)),
                A.Affine(scale=(0.7, 1.3), rotate=(-40, 40), shear=(-0.1, 0.1)),
                A.PadIfNeeded(min_height=target_size, min_width=target_size),
                A.OpticalDistortion(),
                A.RandomCrop(height=target_size, width=target_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
                A.ToFloat()
            ])
        elif mode == "val":
            self.fpaths = sorted(glob.glob(os.path.join(root_dir, "val", "images/*")))
            self.mask_fpaths = sorted(glob.glob(os.path.join(root_dir, "val", "masks/*")))
            self.transform = A.ToFloat()
        else:
            self.fpaths = sorted(glob.glob(os.path.join(root_dir, "test", "images/*")))
            self.mask_fpaths = sorted(glob.glob(os.path.join(root_dir, "test", "masks/*")))
            self.transform = A.ToFloat()
    
    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.fpaths[index]).convert("RGB"))
        image = self.transform(image=image)["image"]
        if self.mode == "train":
            mask = self.mask_generator(image.shape[0], image.shape[1])
        else:
            mask = np.array(Image.open(self.mask_fpaths[index]))
            mask = mask.astype(np.float32) / 255.
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return {
            "image": image,
            "mask": mask,
            "fpath": "_".join(self.fpaths[index].rsplit("/", 1))
        }
