import math
import torch

def sample_training_timestep(sampling_method, batch_size, device, sigmoid_scale = 1.0, sigmoid_shift = 1.0):
    if sampling_method == "uniform":
        time = torch.rand((batch_size,), device=device)
    elif sampling_method == "sigmoid_normal":
        time = torch.randn((batch_size,), device=device)
        time = (time * sigmoid_scale).sigmoid()
        time = (sigmoid_shift * time) / (1 + (sigmoid_shift - 1) * time)
    else:
        raise ValueError("Unknown sampling method.")
    return time


def sample_inference_timestep(
    num_steps: int,
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
    shift_terminal: float = 0.0
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = m * image_seq_len + b

        timesteps = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1) ** 1.0)
    
    if shift_terminal:
        one_minus_t = 1 - timesteps[:-1]
        scale_factor = one_minus_t[-1] / (1 - shift_terminal)
        timesteps = torch.cat([1 - (one_minus_t / scale_factor), timesteps[-1:]])
    
    return timesteps.tolist()
