import functools
import importlib
import logging
import time


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def scale_learning_rate(config):
    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if isinstance(config.lightning.trainer.devices, int):
        ngpu = config.lightning.trainer.devices
    elif isinstance(config.lightning.trainer.devices, (list, tuple)):
        ngpu = len(config.lightning.trainer.devices)
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in config.lightning.trainer:
        accumulate_grad_batches = config.lightning.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    
    if config.model.scale_lr:
        learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {learning_rate:.2e}")
    return learning_rate


def simple_time_tracker(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            res = func(*args, **kwargs)
        finally:
            duration = time.time() - start_time
        print(f'Simple time tracker: Function {func.__name__} runs for {duration} seconds.')
        return res

    return wrapper


def setup_logging(log_file, rank):
    formatter = logging.Formatter(
        "%(asctime)s - Rank %(rank)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if rank == 0:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.disabled = True
    return logger