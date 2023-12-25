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
    parser.add_argument('--checkpoint', type=str, metavar='FILE', help='Path to the checkpoint.')
    parser.add_argument('--config', metavar='FILE', help='Config file for predict.')
    args = parser.parse_args()

    if args.checkpoint:
        args.logdir = '/'.join(args.checkpoint.split('/')[:-2])
        config_paths = glob.glob(os.path.join(args.logdir, 'configs/*.yaml'))
        configs = [OmegaConf.load(p) for p in config_paths]
        config = OmegaConf.merge(*configs)
    else:
        config = OmegaConf.load(args.config)
    pl_config = config.pop('lightning')
    
    test_dataset = utils.instantiate_from_config(config.data.predict)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data.predict_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    pl_model = utils.instantiate_from_config(config.model)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        pl_model.load_state_dict(checkpoint['state_dict'])
    
    logger = pl_utils.get_logger(args, pl_config)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        logger=logger,
    )

    trainer.predict(pl_model, test_dataloader)
