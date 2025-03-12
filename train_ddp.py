# syncnorm?

import argparse, datetime, os
import logging
import random
import sys

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import utils


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_file):
    formatter = logging.Formatter(
        "%(asctime)s - Rank %(rank)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


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

    seed_all(config.trainer.seed)

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
        collate_fn=utils.get_obj_from_str(config.data.train.collate_fn)
    )
    val_datasets = {cfg.name: utils.instantiate_from_config(cfg) for cfg in config.data.val}
    val_dataloaders = {
        name: torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
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
    train(args, train_config, trainer, train_dataloader, val_dataloaders)

    dist.destroy_process_group()


def train(
    args,
    train_config,
    trainer,
    train_dataloader,
    val_dataloaders,
):
    global_step = 0
    start_epoch = 1
    for epoch in range(start_epoch, train_config.num_epochs + 1):
        train_dataloader.sampler.set_epoch(epoch)
        trainer.on_train_epoch_start()
        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}", total=len(train_dataloader)):
            trainer.train_step(batch, batch_idx, global_step)
            global_step += 1
        logdict = trainer.on_train_epoch_end(epoch)
        if dist.get_rank() == 0:
            logging.info(f"Rank {dist.get_rank()}: Epoch {epoch}, training losses {logdict}")
        
        if epoch % train_config.eval_interval == 0:
            datasets_results = eval(trainer, val_dataloaders)
            if dist.get_rank() == 0:
                msg = f"Rank {dist.get_rank()}: Epoch {epoch}, validation metrics \n"
                for key, res in datasets_results.items():
                    msg += f"Dataset {key}: {res}\n"
                logging.info(msg)
        
        if epoch % train_config.ckpt_interval == 0 and dist.get_rank() == 0:
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model": trainer.get_model_state_dict()
            }
            if train_config.save_optimizer_states:
                checkpoint["optimizer"] = trainer.get_optimizer_state_dict()
                checkpoint["lr_scheduler"] = trainer.get_lr_scheduler_state_dict()
            torch.save(checkpoint, os.path.join(args.ckptdir, f"epoch{epoch}_step{global_step}.ckpt"))
        
        dist.barrier()


def eval(trainer, val_dataloaders):
    datasets_results = dict()
    for name, dataloader in val_dataloaders.items():
        trainer.on_val_epoch_start()
        for batch_idx, batch in enumerate(dataloader):
            trainer.val_step(batch, batch_idx)
        epoch_metric_dict = trainer.on_val_epoch_end()
        datasets_results[name] = epoch_metric_dict
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