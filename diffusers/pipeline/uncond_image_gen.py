import torch
from typing import Sequence


def pipeline_ddpm(
    model,
    sampler,
    batch_size: int,
    image_shape: Sequence,
    num_inference_steps: int,
    generator = None
):
    image = torch.randn((batch_size, model.in_channels, *image_shape), generator=generator)

    sampler.set_timesteps(num_inference_steps)

    for t in range(sampler.timesteps):
        output = model(image, t)
        image = sampler.step(output, t, image, generator=generator)
    
    return image


def pipeline_ddim(
    model,
    sampler,
    batch_size: int,
    image_shape: Sequence,
    num_inference_steps: int,
    generator = None
):
    image = torch.randn((batch_size, model.in_channels, *image_shape), generator=generator)

    sampler.set_timesteps(num_inference_steps)

    for t in range(sampler.timesteps):
        output = model(image, t)
        image = sampler.step(output, t, image, generator=generator)
    
    return image