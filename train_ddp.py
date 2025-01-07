syncnorm?

import argparse, datetime, os
import logging
import random
import sys

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utils


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    # use cudnn benchmarking algorithm to select the best conv algorithm
    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa
        torch.backends.cudnn.deterministic = True
    
    # launched by either torch.distributed.elastic (single-node) or Slurm srun command (multi-node)
    # elastic launch with C10d rendezvous backend by default uses TCPStore
    # initialize with environment variables for maximum customizability
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=rank)
    local_world_size = (int(os.environ.get("LOCAL_WORLD_SIZE", "0")) or
                        int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) or
                        torch.cuda.device_count())
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) or rank % local_world_size
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(world_size))
    torch.cuda.set_device(local_rank)
    train_device = torch.device(f"cuda:{local_rank}")

    config = OmegaConf.load(args.config)
    
    args.logdir = os.path.join(args.logdir, config.exp_name)
    args.ckptdir = os.path.join(args.logdir, 'checkpoints')
    args.cfgdir = os.path.join(args.logdir, 'configs')
    args.now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if dist.get_rank() == 0:
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(args.ckptdir, exist_ok=True)
        os.makedirs(args.cfgdir, exist_ok=True)
        print('Project config')
        print(OmegaConf.to_yaml(config))
        OmegaConf.save(config, os.path.join(args.cfgdir, f'{args.now}.yaml'))
        
        logging.basicConfig(
            filename=os.path.join(args.logdir, 'log.txt'),
            filemode='w',
            level=logging.INFO
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info(f"Using distributed training with {world_size} GPU(s).")

    seed_all(config.model.seed)

    train_dataset = utils.instantiate_from_config(config.data.train)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size // world_size,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )
    val_datasets = [utils.instantiate_from_config(cfg) for cfg in config.data.val]
    val_dataloaders = [
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        ) for val_dataset in val_datasets
    ]

    # prepare model
    _model = utils.instantiate_from_config(config.model_config)
    _model.cuda()
    model = DistributedDataParallel(_model, device_ids=[local_rank])

    # prepare optimizer
    optimizer_config = config.pop("optimizer_config")
    optimizer = utils.get_obj_from_str(optimizer_config.optimizer)(
        model.trainable_params(), **optimizer_config.optimizer_params
    )
    scheduler = utils.get_obj_from_str(optimizer_config.lr_scheduler)(
        optimizer, **optimizer_config.lr_scheduler_params
    )

    train(args, model, train_dataloader, val_dataloaders, optimizer, scheduler)


def train(
    args,
    train_config,
    model_file,
    model,
    train_dataloader,
    val_dataloaders,
    optimizer,
    lr_scheduler
):
    global_step = 0
    start_epoch = 1
    for epoch in range(start_epoch, train_config.num_epochs + 1):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(train_dataloader):
            model_file.train_step(model, batch, optimizer, lr_scheduler)
            global_step += 1
        
        if epoch % train_config.eval_interval == 0:
            model.eval()
            eval()
        
        if epoch % train_config.ckpt_interval == 0 and dist.get_rank() == 0:
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.module.state_dict()
            }
            if train_config.save_optimizer_states:
                checkpoint["optimizer"] = optimizer.state_dict()
                checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
            torch.save(checkpoint, os.path.join(args.ckptdir, f"epoch{epoch}_step{global_step}.ckpt"))
        
        dist.barrier()
