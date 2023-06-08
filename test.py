import argparse, glob, os
from omegaconf import OmegaConf
import pytorch_lightning as pl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for test.')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        raise ValueError(f'Cannot find checkpoint {args.checkpoint}.')
    if os.path.isfile(args.checkpoint):
        paths = args.checkpoint.split('/')
        args.logdir = '/'.join(paths[:-2])    # path before /checkpoints
    else:
        assert os.path.isdir(args.checkpoint)
        args.logdir = args.checkpoint
        args.checkpoint = os.path.join(args.logdir, 'checkpoints', 'last.ckpt')
    config_paths = glob.glob(os.path.join(args.logdir, 'configs/*.yaml'))
    configs = [OmegaConf.load(cfg) for cfg in config_paths]
    config = OmegaConf.merge(*configs)
    pl_config = config.pop('lightning', OmegaConf.create())

    logger = pl_utils.get_logger(pl_config)
    callbacks = pl_utils.get_callbacks(pl_config)
    trainer = pl.Trainer(
        pl_config.get('trainer', OmegaConf.create()), logger=logger, callbacks=callbacks
    )

    pl_data = instantiate_from_config(config.data)
    pl_data.prepare_data()
    pl_data.setup()
    print('#### Data ####')
    for k in pl_data.datasets:
        print(f'{k}, {pl_data.datasets[k].__class__.__name__}, {len(pl_data.datasets[k])}')
    
    pl_model = instantiate_from_config(config.model)

    trainer.test(pl_model, pl_data)