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


# load model
config = OmegaConf.load('infant/diffusers/config/stable-diffusion-v2_upscale_inference.yaml')
sd_model = utils.instantiate_from_config(config.model)
# checkpoint = safetensors.torch.load_file('infant/diffusers/pretrained/sd-v2-x4-upscaler-ema.ckpt', device='cpu')
checkpoint = torch.load('infant/diffusers/pretrained/sd-v2-x4-upscaler-ema.ckpt', map_location='cpu')
state_dict = checkpoint['state_dict']
replace_substring_in_state_dict_if_present(state_dict, 'model.diffusion_model', 'diffusion_model')
missing, unexpected = sd_model.load_state_dict(state_dict, strict=False)
sd_model.eval()
sd_model.cuda()

# sampler and denoiser
sampler = utils.instantiate_from_config(config.sampler)
alphas_cumprod = state_dict['alphas_cumprod'].cuda()
sqrt_alphas_cumprod = state_dict['sqrt_alphas_cumprod'].cuda()
sqrt_one_minus_alphas_cumprod = state_dict['sqrt_one_minus_alphas_cumprod'].cuda()
if sd_model.prediction_type == 'epsilon':
    denoiser = KarrasEpsDenoiser(sd_model, alphas_cumprod)
elif sd_model.prediction_type == 'v':
    denoiser = KarrasVDenoiser(sd_model, alphas_cumprod)
denoiser = KarrasCFGDenoiser(denoiser, 7)

# condition inputs for model
cond_pos_prompt = ['']
cond_neg_prompt = ['']
cond_pos_prompt = sd_model.get_learned_conditioning(cond_pos_prompt)
cond_neg_prompt = sd_model.get_learned_conditioning(cond_neg_prompt)

lowres_image = Image.open().convert('RGB')
lowres_image = TF.pil_to_tensor(lowres_image)
lowres_image = lowres_image.unsqueeze(0).cuda()
noise_level = torch.tensor([0]).cuda()
noise = torch.randn(lowres_image.size()).cuda()
lowres_aug = sqrt_alphas_cumprod[noise_level] * lowres_image + sqrt_one_minus_alphas_cumprod[noise_level] * noise

denoiser_args = {
    'cond_pos_prompt': cond_pos_prompt,
    'cond_neg_prompt': cond_neg_prompt,
    'cond_imaging': lowres_aug,
    'cond_emb': noise_level
}
samples = sampler.sample(denoiser, batch_size=1, image_shape=(4, 200, 200), denoiser_args=denoiser_args)

samples = sd_model.decode_first_stage(samples)
samples = torch.clamp((samples + 1) / 2, min=0, max=1)
samples_np = samples.detach().squeeze(0).cpu().numpy()
samples_np = (samples_np * 255).astype(np.uint8)
samples_np = np.moveaxis(samples_np, 0, 2)
samples_pil = Image.fromarray(samples_np)
samples_pil.save()