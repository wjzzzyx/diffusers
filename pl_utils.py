from functools import partial
import numpy as np
from omegaconf import OmegaConf
import os
from PIL import Image
import time
import torch
from torch.utils.data import DataLoader
import torchvision

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities import rank_zero_info

from utils import instantiate_from_config


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        # is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        # if is_iterable_dataset or self.use_worker_init_fn:
        #     init_fn = worker_init_fn
        # else:
        #     init_fn = None
        init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, #shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        # if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
        #     init_fn = worker_init_fn
        # else:
        #     init_fn = None
        init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    # def on_exception(self, trainer, pl_module, exception):
    #     if trainer.global_rank == 0:
    #         print("Summoning checkpoint.")
    #         ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
    #         trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        # else:
        #     # ModelCheckpoint callback created log directory --- remove it
        #     if not self.resume and os.path.exists(self.logdir):
        #         dst, name = os.path.split(self.logdir)
        #         dst = os.path.join(dst, "child_runs", name)
        #         os.makedirs(os.path.split(dst)[0], exist_ok=True)
        #         try:
        #             os.rename(self.logdir, dst)
        #         except FileNotFoundError:
        #             pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(
                global_step,
                current_epoch,
                batch_idx,
                k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        logger = type(pl_module.logger)

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)

        self.log_local(pl_module.logger.save_dir, split, images,
                        pl_module.global_step, pl_module.current_epoch, batch_idx)

        # logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
        # logger_log_images(pl_module, images, pl_module.global_step, split)

        if is_train:
            pl_module.train()

    def should_log_this_step(self, pl_module, check_idx):
        if self.disabled:
            return False

        if pl_module.global_step == 0 and not self.log_first_step:
            return False
        
        if not hasattr(pl_module, 'log_images') and not callable(pl_module.log_images):
            return False
        
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)):
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device)
        torch.cuda.synchronize(trainer.strategy.root_device)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


def get_logger(opt, lightning_config):
    # default logger configs
    default_logger_cfgs = {
        # "wandb": {
        #     "target": "pytorch_lightning.loggers.WandbLogger",
        #     "params": {
        #         "name": opt.nowname,
        #         "save_dir": opt.logdir,
        #         "offline": False,
        #         "id": opt.nowname,
        #     }
        # },
        "testtube": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "name": "tensorboard",
                "save_dir": opt.logdir,
            }
        },
    }
    default_logger_cfg = default_logger_cfgs["testtube"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return instantiate_from_config(logger_cfg)


def get_callbacks(args, config, lightning_config):
    # add callback which sets up log directory
    default_callbacks_cfg = {
        # "setup_callback": {
        #     "target": "pl_utils.SetupCallback",
        #     "params": {
        #         "resume": args.resume,
        #         "now": args.now,
        #         "logdir": args.logdir,
        #         "ckptdir": args.ckptdir,
        #         "cfgdir": args.cfgdir,
        #         "config": config,
        #         "lightning_config": lightning_config,
        #     }
        # },
        # "image_logger": {
        #     "target": "pl_utils.ImageLogger",
        #     "params": {
        #         "batch_frequency": 750,
        #         "max_images": 4,
        #         "clamp": True
        #     }
        # },
        "learning_rate_logger": {
            "target": "pl_utils.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        },
        "cuda_callback": {
            "target": "pl_utils.CUDACallback"
        },
        "checkpoint_callback": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": args.ckptdir,
                # "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
    }

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint':
                {"target": 'lignthing.pytorch.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(args.ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10000,
                        'save_weights_only': True
                    }
                    }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    # if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
    #     callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    # elif 'ignore_keys_callback' in callbacks_cfg:
    #     del callbacks_cfg['ignore_keys_callback']
    
    return [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]