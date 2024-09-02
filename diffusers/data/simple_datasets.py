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


class FixedPrompts(Dataset):
    def __init__(self, image_size):
        self.prompts = [
            'a laptop on the chair',
            'a dog to the left of a cat',
            'a dog to the right of an elephant',
            'a bycicle leaning to a tree',
            'a man holding a picture',
            'a man holding a picture using the left hand',
            'a man holding a picture up using two hands'
        ]
        self.image_size = image_size
    
    def __len__(self):
        return len(self.prompts) * 4

    def __getitem__(self, index):
        prompt = self.prompts[index // 4]
        return {
            'prompt': prompt,
            'image_size': self.image_size,
        }