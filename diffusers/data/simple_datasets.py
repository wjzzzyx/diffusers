import os
from PIL import Image
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torch.utils.data import Dataset


class NullImageDataset(Dataset):
    def __init__(self, len: int):
        self.len = len
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return {'index': index}


class ImageFolder(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.fnames = sorted(os.listdir(image_dir))
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        fpath = os.path.join(self.image_dir, self.fnames[index])
        image = Image.open(fpath)
        image = TF.pil_to_tensor(image)
        image = image.float() / 255
        return {'image': image, 'fname': self.fnames[index]}