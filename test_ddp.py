import argparse
import datetime
import logging
import os

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, default_collate

import torch_utils
import utils


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
    args.logdir = '/'.join(args.checkpoint.split('/')[:-2])
    logger = setup_logging(os.path.join(args.logdir, "test_log.txt"), rank)

    test_datasets = {cfg.name: utils.instantiate_from_config(cfg) for cfg in config.data.test}
    test_samplers = {
        name: torch_utils.InferenceSampler(len(test_dataset))
        for name, test_dataset in test_datasets.items()
    }
    test_dataloaders = {
        name: DataLoader(
            test_dataset,
            batch_size=config.data.val_batch_size // world_size,
            sampler=test_samplers[name],
            drop_last=False,
            pin_memory=True,
            num_workers=config.data.num_workers,
            collate_fn=test_dataset.collate_fn if hasattr(test_dataset, "collate_fn") else default_collate,
        ) for name, test_dataset in test_datasets.items()
    }

    trainer = utils.get_obj_from_str(config.trainer.target)(
        config.trainer.model_config,
        config.trainer.loss_config,
        config.trainer.optimizer_config,
        device=local_rank
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    trainer.load_model_state_dict(checkpoint["model"])
    global_step = checkpoint["global_step"]
    epoch = checkpoint["epoch"]

    datasets_results = eval(args, trainer, test_dataloaders, global_step, epoch)
    msg = f"Checkpoint {args.checkpoint}, test metrics \n"
    for key, res in datasets_results.items():
        msg += f"Dataset {key}: {res}\n"
    logger.info(msg, extra={"rank": 0})

    dist.destroy_process_group()


def eval(args, trainer, test_dataloaders, global_step, epoch):
    datasets_results = dict()
    for name, dataloader in test_dataloaders.items():
        trainer.on_val_epoch_start()
        for batch_idx, batch in enumerate(dataloader):
            trainer.test_step(batch, global_step, epoch, batch_idx, args.logdir)
        metric_dict = trainer.on_val_epoch_end(name, dataloader.dataset, args.logdir)
        datasets_results[name] = metric_dict
        dist.barrier()
    return datasets_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, metavar='config.yaml', help='Path to experiment config.')
    parser.add_argument('--checkpoint', type=str, const=True, help='Checkpoint path.')
    args, unknown = parser.parse_known_args()

    main(args)