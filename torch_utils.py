import torch
import torch.distributed as dist
from typing import Any, Dict


def replace_substring_in_state_dict_if_present(
    state_dict: Dict[str, Any], substring: str, replace: str
):
    keys = sorted(state_dict.keys())
    for key in keys:
        if substring in key:
            newkey = key.replace(substring, replace)
            state_dict[newkey] = state_dict.pop(key)
    
    if "_metadata" in state_dict:
        metadata = state_dict['_metadata']
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove
            # 'module': for the actual model
            # 'module.xx.xx': for the rest
            if len(key) == 0:
                continue
            newkey = key.replace(substring, replace)
            metadata[newkey] = metadata.pop(key)


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


class RunningStatistic():
    def __init__(self, device):
        self.device = device
        self.reset()
    
    def reset(self):
        self.sum = torch.tensor(0, dtype=torch.float, device=self.device)
        self.count = torch.tensor(0, device=self.device)
        self.val = torch.tensor(0, dtype=torch.float, device=self.device)
    
    def update(self, val, n):
        # val is the mean over n counts
        self.count += n
        self.sum += val * n
        self.val = val
    
    def compute(self):
        if dist.is_available() and dist.is_initialized():
            dist.reduce(self.sum, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(self.count, dst=0, op=dist.ReduceOp.SUM)
        mean = self.sum / self.count
        return mean