import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class CelebA(Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, data_dir, selected_attrs, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = os.path.join(data_dir, 'images')
        self.attr_path = os.path.join(data_dir, 'Anno', 'list_attr_celeba.txt')
        self.selected_attrs = selected_attrs
        self.mode = mode

        # train, val, test files
        self.train_fnames = list()
        self.val_fnames = list()
        self.test_fnames = list()
        with open(os.path.join(data_dir, 'Eval', 'list_eval_partition.txt'), 'r') as f:
            for line in f:
                fname, split = line.split()
                if split == '0':
                    self.train_fnames.append(fname)
                elif split == '1':
                    self.val_fnames.append(fname)
                elif split == '2':
                    self.test_fnames.append(fname)
        
        # labels
        self.fname2label = dict()
        self.attr2idx = dict()
        with open(self.attr_path, 'r') as f:
            lines = f.readlines()
            lines = [l.rstrip() for l in lines]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
        self.selected_attr_idxs = [self.attr2idx[attr] for attr in self.selected_attrs]
        lines = lines[2:]
        for iline, line in enumerate(lines):
            parts = line.split()
            fname = parts[0]
            values = parts[1:]
            label = [values[i] for i in self.selected_attr_idxs]
            label = [l == '1' for l in label]
            # for attr in self.selected_attrs:
            #     idx = self.attr2idx[attr]
            #     label.append(values[idx] == '1')
            self.fname2label[fname] = label

        self.transform = T.Compose([
            T.RandomCrop(178),
            T.Resize(128),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode == 'train':
            fname = self.train_fnames[index]
        elif self.mode == 'val':
            fname = self.val_fnames[index]
        elif self.mode == 'test':
            fname = self.test_fnames[index]
        label = self.fname2label[fname]
        label_text = [' '.join(self.selected_attrs[i].split('_')) for i, l in enumerate(label) if l == 1]
        label_text = ','.join(label_text)
        image = Image.open(os.path.join(self.image_dir, fname))
        return {
            'image': self.transform(image),
            'label': torch.FloatTensor(label),
            'label_text': label_text,
        }

    def __len__(self):
        """Return the number of images."""
        if self.mode == 'train':
            return len(self.train_fnames)
        elif self.mode == 'val':
            return len(self.val_fnames)
        elif self.mode == 'test':
            return len(self.test_fnames)