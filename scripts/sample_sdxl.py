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


config = OmegaConf.load('diffusers/config/stable-diffusion-xl10base_inference.yaml')
sd_model = utils.instantiate_from_config(config.model)
checkpoint = safetensors.torch.load_file('', device='cpu')
state_dict = checkpoint
replace_substring_in_state_dict_if_present(state_dict, 'model.diffusion_model', 'diffusion_model')
missing, unexpected = sd_model.load_state_dict(state_dict, strict=False)
sd_model.eval()
sd_model.cuda()

sampler = utils.instantiate_from_config(config.sampler)

alphas_cumprod = sampler.alphas_cumprod
if sd_model.prediction_type == 'epsilon':
    denoiser = KarrasEpsDenoiser(sd_model, alphas_cumprod)
elif sd_model.prediction_type == 'v':
    denoiser = KarrasVDenoiser(sd_model, alphas_cumprod)
denoiser = KarrasCFGDenoiser(denoiser, 7)

cond_pos_prompt = ['']
cond_neg_prompt = ['']

def txt2img():
    width, height = 1024, 1024
    original_size_as_tuple = torch.tensor([[height, width]]).cuda()
    crop_coords_top_left = torch.tensor([[0, 0]]).cuda()
    target_size_as_tuple = torch.tensor([[height, width]]).cuda()

    batch = {'txt': cond_pos_prompt, 'original_size_as_tuple': original_size_as_tuple, 'crop_coords_top_left': crop_coords_top_left, 'target_size_as_tuple': target_size_as_tuple}
    batch_uc = {'txt': cond_neg_prompt, 'original_size_as_tuple': original_size_as_tuple, 'crop_coords_top_left': crop_coords_top_left, 'target_size_as_tuple': target_size_as_tuple}

    c, uc = sd_model.conditioner.get_unconditional_conditioning(batch, batch_uc)

    denoiser_args = {'cond_pos_prompt': c['crossattn'], 'cond_neg_prompt': uc['crossattn'], 'cond_emb': c['vector']}
    samples = sampler.sample(denoiser, batch_size=1, image_shape=(4, height // 8, width // 8), denoiser_args=denoiser_args)
    return samples

def img2img(image):
    width, height = image.size
    image = TF.pil_to_tensor(image)
    image = image.float() / 255
    image = image * 2 - 1
    image = image.to(sd_model.device)
    image = image.unsqueeze(0)

    original_size_as_tuple = torch.tensor([[height, width]]).cuda()
    crop_coords_top_left = torch.tensor([[0, 0]]).cuda()
    aesthetic_score = torch.tensor([[6.0]]).cuda()
    negative_aesthetic_score = torch.tensor([[2.5]]).cuda()

    batch = {'txt': cond_pos_prompt, 'original_size_as_tuple': original_size_as_tuple, 'crop_coords_top_left': crop_coords_top_left, 'aesthetic_score': aesthetic_score}
    batch_uc = {'txt': cond_neg_prompt, 'original_size_as_tuple': original_size_as_tuple, 'crop_coords_top_left': crop_coords_top_left, 'negative_aesthetic_score': negative_aesthetic_score}
    
    c, uc = sd_model.conditioner.get_unconditional_conditioning(batch, batch_uc)
    z = sd_model.encode_first_stage(image)
    
    denoiser_args = {
        'cond_pos_prompt': c['crossattn'],
        'cond_neg_prompt': uc['crossattn'],
        'cond_emb': c['vector'],
    }
    samples = sampler.sample(denoiser, batch_size=1, image=z, denoiser_args=denoiser_args)
    return samples

# samples = txt2img()
image = Image.open()
samples = img2img(image)

samples = sd_model.decode_first_stage(samples)
samples = torch.clamp((samples + 1) / 2, min=0, max=1)
samples_np = samples.detach().squeeze(0).cpu().numpy()
samples_np = (samples_np * 255).astype(np.uint8)
samples_np = np.moveaxis(samples_np, 0, 2)
samples_pil = Image.fromarray(samples_np)