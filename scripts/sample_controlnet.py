import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from diffusers.sampler.sample_dpm import DPMSampler
from diffusers.sampler.denoiser import KarrasCFGDenoiser, KarrasEpsDenoiser, KarrasVDenoiser
from dense_predictors.model import hed
import utils

config = OmegaConf.load('')
sd_model = utils.instantiate_from_config(config.model)
sd_model.eval()
sd_model.cuda()

sampler = DPMSampler(
    sampler='dpmpp_2m',
    scheduler='karras',
    solver_type='none',
    discard_next_to_last_sigma=False,
    second_order=False,
    uses_ensd=False,
    num_train_steps=1000,
    num_inference_steps=20,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule='sqrt_linear',
    denoising_strength=1.0
)
alphas_cumprod = sampler.alphas_cumprod
if sd_model.prediction_type == 'epsilon':
    denoiser = KarrasEpsDenoiser(sd_model, alphas_cumprod)
elif sd_model.prediction_type == 'v':
    denoiser = KarrasVDenoiser(sd_model, alphas_cumprod)
denoiser = KarrasCFGDenoiser(denoiser, 7)

batch_size = 4
cond_pos_prompt = [''] * batch_size
cond_neg_prompt = [''] * batch_size
cond_pos_prompt = sd_model.get_learned_conditioning(cond_pos_prompt)
cond_neg_prompt = sd_model.get_learned_conditioning(cond_neg_prompt)

def canny_ref_image(ref_image):
    ref_image = np.array(ref_image)
    ref_image = cv2.Canny(ref_image, 100, 200)
    Image.fromarray(ref_image).save()
    ref_image = torch.from_numpy(ref_image).float() / 255
    ref_image = ref_image.unsqueeze(-1).repeat(1, 1, 3)
    ref_image = ref_image.permute(2, 0, 1).unsqueeze(0)
    ref_image = ref_image.cuda()
    return ref_image

def hed_ref_image(ref_image):
    hed_config = OmegaConf.create({'pretrained': ''})
    hed_network = hed.HED(hed_config).cuda().eval()
    ref_image = torch.from_numpy(np.array(ref_image)).cuda()
    ref_image = ref_image.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        edge = hed_network(ref_image)
    Image.fromarray((edge[0].cpu().numpy() * 255).astype(np.uint8)).save('')
    edge = edge.unsqueeze(1).expand(-1, 3, -1, -1)
    return edge

ref_image = Image.open()
ref_image = ref_image.resize((512, 512))
ref_image = canny_ref_image(ref_image)

denoiser_args = {
    'cond_pos_prompt': cond_pos_prompt,
    'cond_neg_prompt': cond_neg_prompt,
    'cond_control': ref_image
}
samples = sampler.sample(denoiser, batch_size=batch_size, image_shape=(4, 64, 64), denoiser_args=denoiser_args)
samples = sd_model.decode_first_stage(samples)
samples = torch.clamp((samples + 1) / 2, min=0, max=1)
samples_np = samples.detach().permute(0, 2, 3, 1).cpu().numpy()
samples_np = (samples_np * 255).astype(np.uint8)
for i in range(batch_size):
    samples_pil = Image.fromarray(samples_np[i])
    samples_pil.save()