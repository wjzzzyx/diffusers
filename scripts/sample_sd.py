import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import safetensors
import torch
import torchvision.transforms.v2.functional as TF

from diffusers.sampler.denoiser import DiscreteTimeEpsDenoiser, DiscreteTimeCFGDenoiser
from diffusers.sampler.denoiser import KarrasEpsDenoiser, KarrasVDenoiser, KarrasCFGDenoiser
from torch_utils import replace_substring_in_state_dict_if_present
import utils


config = OmegaConf.load('infant/diffusers/config/stable-diffusion-v1-inference.yaml')
sd_model = utils.instantiate_from_config(config.model)
checkpoint = safetensors.torch.load_file('infant/diffusers/pretrained/', device='cpu')
state_dict = checkpoint
replace_substring_in_state_dict_if_present(state_dict, 'model.diffusion_model', 'diffusion_model')
missing, unexpected = sd_model.load_state_dict(state_dict, strict=False)
sd_model.eval()
sd_model.cuda()

sampler = utils.instantiate_from_config(config.sampler)
alphas_cumprod = state_dict['alphas_cumprod']
if sd_model.prediction_type == 'epsilon':
    denoiser = KarrasEpsDenoiser(sd_model, alphas_cumprod.cuda())
elif sd_model.prediction_type == 'v':
    denoiser = KarrasVDenoiser(sd_model, alphas_cumprod.cuda())
denoiser = KarrasCFGDenoiser(denoiser, 7)

cond_pos_prompt = ['']
cond_neg_prompt = ['']
cond_pos_prompt = sd_model.get_learned_conditioning(cond_pos_prompt)
cond_neg_prompt = sd_model.get_learned_conditioning(cond_neg_prompt)

denoiser_args = {'cond_pos_prompt': cond_pos_prompt, 'cond_neg_prompt': cond_neg_prompt}

def txt2img():
    samples = sampler.sample(denoiser, batch_size=1, image_shape=(4, 64, 64), denoiser_args=denoiser_args)
    return samples


def img2img(image):
    image = TF.pil_to_tensor(image)
    image = image.float() / 255
    image = image * 2 - 1
    image = image.to(sd_model.device)
    
    z = sd_model.encode_first_stage(image)

    samples = sampler.sample(denoiser, batch_size=1, image=z, denoiser_args=denoiser_args)
    return samples

image = Image.open()
samples = txt2img()

samples = sd_model.decode_first_stage(samples)
samples = torch.clamp((samples + 1) / 2, min=0, max=1)
samples_np = samples.detach().squeeze(0).cpu().numpy()
samples_np = (samples_np * 255).astype(np.uint8)
samples_np = np.moveaxis(samples_np, 0, 2)
samples_pil = Image.fromarray(samples_np)
samples_pil.save()