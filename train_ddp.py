# syncnorm?

import argparse, datetime, os
import logging
import random
import sys

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
# import torch.profiler as profiler
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch_utils
import utils

os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    dist.init_process_group(
        "nccl", init_method="env://", world_size=world_size, rank=rank, timeout=datetime.timedelta(minutes=60)
    )
    local_world_size = (int(os.environ.get("LOCAL_WORLD_SIZE", "0")) or
                        int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) or
                        torch.cuda.device_count())
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) or rank % local_world_size
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(world_size))
    torch.cuda.set_device(local_rank)

    config = OmegaConf.load(args.config)
    
    args.logdir = os.path.join(args.logdir, config.exp_name)
    args.ckptdir = os.path.join(args.logdir, 'checkpoints')
    args.cfgdir = os.path.join(args.logdir, 'configs')
    args.now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if dist.get_rank() == 0:
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(args.ckptdir, exist_ok=True)
        os.makedirs(args.cfgdir, exist_ok=True)

        OmegaConf.save(config, os.path.join(args.cfgdir, f'{args.now}.yaml'))
        
        logging.basicConfig(
            filename=os.path.join(args.logdir, 'log.txt'),
            filemode='w',
            level=logging.INFO
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    writer = SummaryWriter(args.logdir) if rank == 0 else None
    
    logging.info("Project config" + "\n" + OmegaConf.to_yaml(config))
    logging.info(f"Using distributed training with {world_size} GPU(s).")

    seed_all(config.trainer.seed + rank)

    train_dataset = utils.instantiate_from_config(config.data.train)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size // world_size,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else default_collate
    )
    val_datasets = {cfg.name: utils.instantiate_from_config(cfg) for cfg in config.data.val}
    val_samplers = {
        name: torch_utils.InferenceSampler(len(val_dataset))
        for name, val_dataset in val_datasets.items()
    }
    val_dataloaders = {
        name: torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size // world_size,
            sampler=val_samplers[name],
            drop_last=False,
            pin_memory=True,
            num_workers=config.data.num_workers,
            collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else default_collate
        ) for name, val_dataset in val_datasets.items()
    }

    # prepare model
    num_training_steps = config.train_config.num_epochs * len(train_dataloader)
    config.trainer.optimizer_config.num_training_steps = num_training_steps
    trainer = utils.get_obj_from_str(config.trainer.target)(
        config.trainer.model_config,
        config.trainer.loss_config,
        config.trainer.optimizer_config,
        device=local_rank
    )
    
    train_config = config.pop("train_config")
    train(args, train_config, trainer, train_dataloader, val_dataloaders, writer)

    if rank == 0:
        writer.close()
    dist.destroy_process_group()


def train(
    args,
    train_config,
    trainer,
    train_dataloader,
    val_dataloaders,
    writer
):
    global_step = 1
    start_epoch = 1
    ### To use profiler, open chrome://tracing and load the trace.json file, or view in tensorboard
    # prof = profiler.profile(
    #     activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    #     schedule=profiler.schedule(skip_first=10, wait=1, warmup=1, active=1, repeat=1),
    #     on_trace_ready=profiler.tensorboard_trace_hander(args.logdir),
    #     record_shapes=False,
    #     profile_memory=False,
    #     with_stack=False
    # )
    for epoch in range(start_epoch, train_config.num_epochs + 1):
        train_dataloader.sampler.set_epoch(epoch)
        trainer.on_train_epoch_start()
        # prof.start()
        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}", total=len(train_dataloader)):
            # prof.step()
            output = trainer.train_step(batch, batch_idx, global_step)
            if global_step % train_config.log_interval == 0:
                logdict = trainer.log_step(batch, output, args.logdir, global_step, epoch, batch_idx)
                if dist.get_rank() == 0:
                    for key, val in logdict.items():
                        writer.add_scalar(f"{key}/train", val, global_step)
                dist.barrier()
            global_step += 1
        logdict = trainer.on_train_epoch_end(epoch)
        # prof.stop()
        # break
        
        if epoch % train_config.eval_interval == 0:
            datasets_results = eval(args, trainer, val_dataloaders, global_step, epoch)
            msg = f"Rank {dist.get_rank()}: Epoch {epoch}, validation metrics \n"
            for key, res in datasets_results.items():
                msg += f"Dataset {key}: {res}\n"
            logging.info(msg)
            if dist.get_rank() == 0:
                for dataset_name, dataset_res in datasets_results.items():
                    for key, val in dataset_res.items():
                        writer.add_scalar(f"val/{dataset_name}/{key}", val, global_step)
        
        dist.barrier()
        
        if epoch % train_config.ckpt_interval == 0 and dist.get_rank() == 0:
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model": trainer.get_model_state_dict()
            }
            if train_config.save_optimizer_states:
                checkpoint["optimizer"] = trainer.get_optimizer_state_dict()
                checkpoint["lr_scheduler"] = trainer.get_lr_scheduler_state_dict()
                checkpoint["scaler"] = trainer.get_scaler_state_dict()
            torch.save(checkpoint, os.path.join(args.ckptdir, f"epoch{epoch}_step{global_step}.ckpt"))
        
        dist.barrier()
    
    # prof.export_chrome_trace(os.path.join(args.logdir, "trace.json"))


def eval(args, trainer, val_dataloaders, global_step: int, epoch: int):
    datasets_results = dict()
    for name, dataloader in val_dataloaders.items():
        trainer.on_val_epoch_start()
        for batch_idx, batch in enumerate(dataloader):
            trainer.val_step(batch, global_step, epoch, batch_idx, args.logdir)
        metric_dict = trainer.on_val_epoch_end(name, dataloader.dataset, args.logdir)
        datasets_results[name] = metric_dict
        dist.barrier()
    return datasets_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, metavar='config.yaml', help='Path to experiment config.')
    parser.add_argument('--resume', type=str, const=True, nargs='?', help='Resume from logdir.')
    parser.add_argument('--logdir', type=str, default='logs', help='Directory for logs, checkpoints, samples, etc.')
    args, unknown = parser.parse_known_args()

    """
    As opposed to the case of rigid launch, distributed training now:
    (*: elastic launch only; **: Slurm srun only)
        *1. handles failures by restarting all the workers 
        *2.1 assigns RANK and WORLD_SIZE automatically
        **2.2 sets MASTER_ADDR & MASTER_PORT manually beforehand via environment variables
        *3. allows for number of nodes change
        4. uses TCP initialization by default
    **5. supports multi-node training
    """
    main(args)