import argparse
import glob
import lightning as pl
from omegaconf import OmegaConf
import os
import torch

import pl_utils
import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, metavar='FILE', help='Path to checkpoint.')
    args = parser.parse_args()

    args.logdir = '/'.join(args.checkpoint.split('/')[:-2])
    config_paths = glob.glob(os.path.join(args.logdir, 'configs/*.yaml'))
    configs = [OmegaConf.load(p) for p in config_paths]
    config = OmegaConf.merge(*configs)
    pl_config = config.pop('lightning')

    test_dataset = utils.instantiate_from_config(config.data.test)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    pl_model = utils.instantiate_from_config(config.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    pl_model.load_state_dict(checkpoint['state_dict'])    # only model weights are loaded

    logger = pl_utils.get_logger(args, pl_config)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=logger,
    )

    trainer.test(pl_model, test_dataloader)