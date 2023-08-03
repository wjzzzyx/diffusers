import numpy as np
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF


class Cifar10(Dataset):
    """Dataset class for the cifar10 dataset."""
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
    
        with open(os.path.join(data_dir, 'batches.meta'), 'rb') as f:
            meta = pickle.load(f)
            self.id2name = meta['label_names']
        
        train_images = list()
        train_labels = list()
        for batch in [1, 2, 3, 4, 5]:
            with open(os.path.join(data_dir, f'data_batch_{batch}'), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            train_images.append(data[b'data'])
            train_labels.append(data[b'labels'])
        self.train_images = np.concatenate(train_images, axis=0)    # shape (50000, 3072)
        self.train_images = self.train_images.reshape(self.train_images.shape[0], 3, 32, 32)
        self.train_images = self.train_images.transpose(0, 2, 3, 1)
        self.train_labels = np.concatenate(train_labels, axis=0)

        with open(os.path.join(data_dir, f'test_batch'), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        self.test_images = data[b'data']
        self.test_images = self.test_images.reshape(self.test_images.shape[0], 3, 32, 32)
        self.test_images = self.test_images.transpose(0, 2, 3, 1)
        self.test_labels = np.array(data[b'labels'])

        # TODO transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_images)
        else:
            return len(self.test_images)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            image = self.train_images[index]
            label = self.train_labels[index]
        else:
            image = self.test_images[index]
            label = self.test_labels[index]
        image = self.transform(image)
        return {'image': image, 'label': label, 'label_text': self.id2name[label]}


class Cifar10Eval(Dataset):
    """A simple dataset for parallel loading in evaluation."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.fnames = os.listdir(data_dir)
    
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        fname = self.fnames[index]
        image = Image.open(os.path.join(self.data_dir, fname))
        image = TF.to_tensor(image)
        return image