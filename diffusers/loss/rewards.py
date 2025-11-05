import io
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from transformers import CLIPModel, CLIPProcessor
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


def jpeg_incompressibility(images: torch.Tensor):
    images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    images = images.transpose(0, 2, 3, 1)    # shape (batch_size, height, width, channels)
    images = [Image.fromarray(image) for image in images]
    buffers = [io.BytesIO() for _ in images]
    for image, buffer in zip(images, buffers):
        image.save(buffer, format="JPEG", quality=95)
    sizes = [buffer.tell() / 1000 for buffer in buffers]
    return torch.tensor(sizes).cuda()


def jpeg_compressibility(images: torch.Tensor):
    return - jpeg_incompressibility(images)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(nn.Module):
    def __init__(self, path="sac+logos+ava1-l14-linearMSE.pth"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        state_dict = torch.load(path, weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.eval()

    def __call__(self, images):
        embed = self.clip.get_image_features(images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def aesthetic_score():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    scorer = AestheticScorer().cuda()

    def _fn(images: torch.Tensor, prompts):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        device = next(scorer.parameters()).device
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return scorer(inputs["pixel_values"])
    
    return _fn


def aesthetic_loss_fn():
    target_size = 224
    normalize = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    scorer = AestheticScorer().cuda()
    scorer.requires_grad_(False)

    def _fn(images, prompts):
        images = torchvision.transforms.Resize(target_size)(images)
        images = normalize(images)
        rewards = scorer(images)
        return rewards
    
    return _fn


def hps_loss_fn(inference_dtype=None, device=None):
    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )    
    
    tokenizer = get_tokenizer(model_name)
    
    link = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
    import os
    import requests
    from tqdm import tqdm

    # Create the directory if it doesn't exist
    os.makedirs(os.path.expanduser('~/.cache/hpsv2'), exist_ok=True)
    checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"

    # Download the file if it doesn't exist
    if not os.path.exists(checkpoint_path):
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(checkpoint_path, 'wb') as file, tqdm(
            desc="Downloading HPS_v2_compressed.pt",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    
    
    # force download of model via score
    hpsv2.score([], "")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def loss_fn(im_pix, prompts):
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return  loss, scores
    
    return loss_fn