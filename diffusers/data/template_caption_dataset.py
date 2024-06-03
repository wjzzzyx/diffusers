import os
from PIL import Image
import random
import torch
import torchvision.transforms.v2 as T


class TemplateCaptionDataset(torch.utils.data.Dataset):
    object_templates_small = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    style_templates_small = [
        "a painting in the style of {}",
        "a rendering in the style of {}",
        "a cropped painting in the style of {}",
        "the painting in the style of {}",
        "a clean painting in the style of {}",
        "a dirty painting in the style of {}",
        "a dark painting in the style of {}",
        "a picture in the style of {}",
        "a cool painting in the style of {}",
        "a close-up painting in the style of {}",
        "a bright painting in the style of {}",
        "a cropped painting in the style of {}",
        "a good painting in the style of {}",
        "a close-up painting in the style of {}",
        "a rendition in the style of {}",
        "a nice painting in the style of {}",
        "a small painting in the style of {}",
        "a weird painting in the style of {}",
        "a large painting in the style of {}",
    ]

    def __init__(self, image_dir, ti_name, mode):
        " "
        self.image_dir = image_dir
        self.fnames = os.listdir(image_dir)
        self.ti_name = ti_name
        self.mode = mode

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((512, 512)),
            T.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        feeddict = dict()
        if self.mode == 'train':
            fname = self.fnames[index]
            image = Image.open(os.path.join(self.image_dir, fname))
            image = self.transform(image)
            feeddict['images'] = image
        caption = random.choice(self.object_templates_small).format(self.ti_name)
        feeddict['captions'] = caption
        return feeddict