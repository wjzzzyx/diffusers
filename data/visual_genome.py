import json
import os
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as TF


class VisualGenomeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        with open(os.path.join(root_dir, 'objects.json')) as f:
            self.all_objects = json.load(f)
        with open(os.path.join(root_dir, 'relationships.json')) as f:
            self.all_relations = json.load(f)
        
        self.openai_clip_mean = (0.48145466, 0.4578275, 0.40821073)
        self.openai_clip_std = (0.26862954, 0.26130258, 0.27577711)
    
    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        objects = self.all_objects[index]
        relations = self.all_relations[index]
        assert(objects['image_id'] == relations['image_id'])
        image_fname = '/'.join(objects['image_url'].split('/')[-2:])
        image = Image.open(os.path.join(self.root_dir, image_fname)).convert('RGB')
        # rel = random.choice(relations)
        image = TF.pil_to_tensor(image)
        h, w = image.shape[-2:]
        maxhw = max(h, w)
        pad_size = (
            (maxhw - w) // 2,
            (maxhw - h) // 2,
            maxhw - (maxhw - w) // 2,
            maxhw - (maxhw - h) // 2
        )
        pad_value = tuple(int(x * 255) for x in self.openai_clip_mean)
        image = TF.pad(pad_size, fill=pad_value)
        return image