import einops
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from diffusers.model.flux import FluxModel, forward_text_model
from diffusers.sampler.sample_flow_match import sample_inference_timestep


seed = 42
device = torch.device('cuda')

prompt = ['']
num_steps: int = 4
guidance: float = 3.5

config = OmegaConf.load('diffusers/config/flux_schnell.yaml')
model = FluxModel(config.model)

batch_size = 1
c = 16
height = 512
width = 512

image = torch.randn(
    batch_size, c, 2 * height // 16, 2 * width // 16,
    dtype=torch.bfloat16, device=device, generator=torch.Generator(device=device).manual_seed(seed)
)
c, h, w = image.shape[1:]
image = einops.rearrange(image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
img_ids = torch.zeros(h // 2, w // 2, 3)
img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
img_ids = einops.repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
img_ids = img_ids.to(device)

model.t5_text_model.to(device)
model.clip_text_model.to(device)
txt, txt_ids, vec = forward_text_model(model.t5_tokenizer, model.t5_text_model, model.clip_tokenizer, model.clip_text_model, prompt, max_length=128)
model.t5_text_model.cpu()
model.clip_text_model.cpu()
torch.cuda.empty_cache()

timesteps = sample_inference_timestep(num_steps, image.size(1), shift=config.shift_schedule)

model.flow_model.to(device)
image = model.sample(image, img_ids, txt, txt_ids, vec, timesteps, guidance)
model.flow_model.cpu()
torch.cuda.empty_cache()

image = image.float()
image = einops.rearrange(image, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=height // 16, w=width // 16, ph=2, pw=2)
model.ae.to(device)
image = model.ae.decode(image)
model.ae.cpu()

image = image.clamp(-1, 1)
image = (image + 1) / 2 * 255
image = image[0].permute(1, 2, 0)
image = image.cpu().numpy().astype(np.uint8)
image = Image.fromarray(image)