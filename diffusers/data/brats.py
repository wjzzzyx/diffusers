from glob import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class BRATS_SYN(Dataset):
    """Dataset class for the BRATS dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and Load the BRATS dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.load_data()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def load_data(self):
        """Load BRATS dataset"""
        
        # Load test dataset
        test_neg = glob(os.path.join(self.image_dir, 'test', 'negative', '*jpg'))
        test_pos = glob(os.path.join(self.image_dir, 'test', 'positive', '*jpg'))

        for filename in test_neg:
            self.test_dataset.append([filename, [0]])

        for filename in test_pos:
            self.test_dataset.append([filename, [1]])


        # Load train dataset
        train_neg = glob(os.path.join(self.image_dir, 'train', 'negative', '*jpg'))
        train_pos = glob(os.path.join(self.image_dir, 'train', 'positive', '*jpg'))

        for filename in train_neg:
            self.train_dataset.append([filename, [0]])

        for filename in train_pos:
            self.train_dataset.append([filename, [1]])

        print('Finished loading the BRATS dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(filename)
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images