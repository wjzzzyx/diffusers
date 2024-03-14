import cv2
import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import torch

from diffusers import k_diffusion


def find_suitable_input_size(init_w, init_h):
    if init_w < init_h:
        input_h = 512
        input_w = int(round(init_w / init_h * input_h))
    else:
        input_w = 512
        input_h = int(round(init_h / init_w * input_w))
    return input_w, input_h


def prepare_image_and_mask(image, out_left=0, out_right=0, out_up=0, out_down=0, mask_blur=4):
    width, height = image.size
    target_h = height + out_up + out_down
    target_w = width + out_left + out_right

    extended_image = Image.new('RGB', (target_w, target_h))
    extended_image.paste(image, (out_left, out_up))

    mask = Image.new('L', (target_w, target_h), 'white')
    draw = ImageDraw.Draw(mask)
    draw.rectangle(
        (
            out_left + (mask_blur * 2 if out_left > 0 else 0),
            out_up + (mask_blur * 2 if out_up > 0 else 0),
            target_w - out_right - (mask_blur * 2 if out_right > 0 else 0),
            target_h - out_down - (mask_blur * 2 if out_down > 0 else 0)
        ),
        fill='black'
    )

    latent_mask = Image.new('L', (target_w, target_h), 'white')
    latent_draw = ImageDraw.Draw(latent_mask)
    latent_draw.rectangle(
        (
            out_left + (mask_blur // 2 if out_left > 0 else 0),
            out_up + (mask_blur // 2 if out_up > 0 else 0),
            target_w - out_right - (mask_blur // 2 if out_right > 0 else 0),
            target_h - out_down - (mask_blur // 2 if out_down > 0 else 0)
        ),
        fill='black'
    )

    return extended_image, mask, latent_mask


def fill(image, mask):
    "fills masked regions with colors from image using blur"
    image_mod = Image.new('RGBA', (image.width, image.height))

    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert('RGBA').convert('RGBa'), mask=ImageOps.invert(mask.convert('L')))

    image_masked = image_masked.convert('RGBa')

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)
    
    return image_mod.convert('RGB')


def get_sigmas(steps):
    # karras
    sigma_min, sigma_max = denoiser.sigmas[0].item(), denoiser.sigmas[-1].item()
    sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max)
    return sigmas


def cfg_denoise(x, sigma, cond, uncond, cond_inpaint, cfg_scale, latent, latent_mask_inv, latent_mask):
    x_in = torch.cat([x, x])
    sigma_in = torch.cat([sigma, sigma])
    cond_inpaint_in = torch.cat([cond_inpaint, cond_inpaint])
    cond_in = torch.cat([cond, uncond])

    x_out = denoiser(x_in, sigma_in, cond={'c_crossattn': [cond_in], 'c_concat': [cond_inpaint_in]})

    denoised_cond = x_out[:cond.shape[0]]
    denoised_uncond = x_out[-uncond.shape[0]:]
    denoised = denoised_cond.clone()
    denoised += (denoised_cond - denoised_uncond) * cfg_scale

    denoised = latent * latent_mask_inv + denoised * latent_mask

    return denoised


def sample_img2img(
    latent, noise, cond_pos_prompt, cond_neg_prompt, cond_inpaint,
    latent_mask_inv, latent_mask, num_step, denoising_strength, cfg_scale
):
    # ?
    steps = num_step
    t_enc = int(min(denoising_strength, 0.999) * steps)
    sigmas = get_sigmas(steps)
    sigma_sched = sigmas[steps - t_enc - 1:]

    xi = latent + noise * sigma_sched[0]
    # TODO extra noise

    extra_args = {
        'cond': cond_pos_prompt,
        'uncond': cond_neg_prompt,
        'cond_inpaint': cond_inpaint,
        'cfg_scale': cfg_scale,
        'latent': latent,
        'latent_mask_inv': latent_mask_inv,
        'latent_mask': latent_mask
    }
    sample = k_diffusion.sampling.sample_dpmpp_2m(cfg_denoise, xi, sigmas=sigma_sched, extra_args=extra_args)
    return sample


def process_image(
    init_image, mask, latent_mask, pos_prompt, neg_prompt,
    num_step, denoising_strength, cfg_scale, mask_blur
):
    seed = int(random.randrange(4294967294))
    generator = torch.Generator().manual_seed(seed)
    assert(init_image.mode == 'RGB')
    assert(mask.mode == 'L')
    mask_blur_x, mask_blur_y = mask_blur, mask_blur

    # prepare image and mask for overlay
    mask_np = np.array(mask)
    if mask_blur_x > 0:
        kernel_size = 2 * round(2.5 * mask_blur_x) + 1
        mask_np = cv2.GaussianBlur(mask_np, (kernel_size, 1), mask_blur_x)
    if mask_blur_y > 0:
        kernel_size = 2 * round(2.5 * mask_blur_y) + 1
        mask_np = cv2.GaussianBlur(mask_np, (1, kernel_size), mask_blur_y)
    mask_blurred = Image.fromarray(mask_np)

    mask_np = np.clip(mask_np.astype(np.float32) * 2, 0, 255).astype(np.uint8)
    mask_for_overlay = Image.fromarray(mask_np)

    image_for_overlay = Image.new('RGBa', (init_image.width, init_image.height))
    image_for_overlay.paste(init_image.convert('RGBA').convert('RGBa'), mask=ImageOps.invert(mask_for_overlay))

    image = fill(init_image, latent_mask)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.moveaxis(image_np, 2, 0)
    image_t = torch.from_numpy(image_np).unsqueeze(0)
    image_t = image_t.to(sd_model.device)
    image_t = image_t * 2 - 1
    latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image_t))

    latent_mask = latent_mask.resize((latent.size(3), latent.size(2)))
    latent_mask = np.array(latent_mask).astype(np.float32) / 255.0
    latent_mask = np.around(latent_mask)    # binary latent mask
    latent_mask = np.tile(latent_mask[None], (latent.size(1), 1, 1))
    latent_mask_inv = torch.asarray(1.0 - latent_mask).cuda()
    latent_mask = torch.asarray(latent_mask).cuda()

    cond_mask = np.array(mask_blurred)
    cond_mask = cond_mask.astype(np.float32) / 255.0
    cond_mask = torch.from_numpy(cond_mask[None, None]).cuda()
    cond_mask = torch.round(cond_mask)
    cond_image = image_t * (1.0 - cond_mask)
    cond_image = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(cond_image))
    cond_mask = F.interpolate(cond_mask, (latent.size(2), latent.size(3)))
    cond_inpaint = torch.cat((cond_mask, cond_image), dim=1)

    with torch.no_grad(), sd_model.ema_scope():
        cond_pos_prompt = sd_model.get_learned_conditioning(pos_prompt)
        cond_neg_prompt = sd_model.get_learned_conditioning(neg_prompt)

        noise = torch.randn((4, latent.size(2), latent.size(3)), generator=generator)
        noise = noise.cuda()
        # if initial_noise_multiplier != 1.0:
        #     noise *= initial_noise_multiplier

        sample = sample_img2img(
            latent, noise, cond_pos_prompt, cond_neg_prompt, cond_inpaint,
            latent_mask_inv, latent_mask, num_step, denoising_strength, cfg_scale
        )
        sampe = sample * latent_mask + latent * latent_mask_inv

        sample = sd_model.decode_first_stage(sample)
        sample = torch.clamp((sample + 1) / 2, 0., 1.)
    
    sample_np = sample.squeeze(0).cpu().numpy()
    sample_np = np.moveaxis(sample_np, 0, 2)
    sample_np = (sample_np * 255).astype(np.uint8)
    sample_image = Image.fromarray(sample_np)

    sample_image = sample_image.convert('RGBA')
    sample_image.alpha_composite(image_for_overlay.convert('RGBA'))
    sample_image = sample_image.convert('RGB')
    
    return sample_image