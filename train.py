import argparse, datetime, os
from omegaconf import OmegaConf
import pytorch_lightning as pl

import pl_utils
import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, metavar='config.yaml', help='Path to experiment config.')
    parser.add_argument('--resume', type=str, const=True, nargs='?', help='Resume from logdir.')
    parser.add_argument('--logdir', type=str, default='logs', help='Directory for logs, checkpoints, samples, etc.')
    args, unknown = parser.parse_known_args()

    if args.resume:
        ...
    else:
        ...
    
    config = OmegaConf.load(args.config)

    args.logdir = os.path.join(args.logdir, config.exp_name)
    args.ckptdir = os.path.join(args.logdir, 'checkpoints')
    args.cfgdir = os.path.join(args.logdir, 'configs')
    args.now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    pl.seed_everything(config.model.seed)
    model_config = config.model.params.model_config
    model_config.learning_rate = utils.scale_learning_rate(config)
    pl_config = config.pop('lightning', OmegaConf.create())

    # set callbacks
    logger = pl_utils.get_logger(args, pl_config)
    callbacks = pl_utils.get_callbacks(args, config, pl_config)
    trainer = pl.Trainer(
        **pl_config.get('trainer', OmegaConf.create()), logger=logger, callbacks=callbacks
    )

    # set datasets
    pl_data = utils.instantiate_from_config(config.data)
    pl_data.prepare_data()
    pl_data.setup()
    print('#### Data ####')
    for k in pl_data.datasets:
        print(f'{k}, {pl_data.datasets[k].__class__.__name__}, {len(pl_data.datasets[k])}')

    # set model
    pl_model = utils.instantiate_from_config(config.model)

    try:
        trainer.fit(pl_model, pl_data)
    except Exception:
        # melk()
        raise
    finally:
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
