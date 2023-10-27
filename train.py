import argparse, datetime, os
from omegaconf import OmegaConf
import torch
import lightning as pl

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
    # model_config = config.model.params.model_config
    # model_config.learning_rate = utils.scale_learning_rate(config)
    pl_config = config.pop('lightning', OmegaConf.create())

    # set callbacks
    logger = pl_utils.get_logger(args, pl_config)
    callbacks = pl_utils.get_callbacks(args, config, pl_config)
    trainer = pl.Trainer(
        **pl_config.get('trainer', OmegaConf.create()), logger=logger, callbacks=callbacks
    )

    # create folders and save config
    if trainer.is_global_zero:
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(args.ckptdir, exist_ok=True)
        os.makedirs(args.cfgdir, exist_ok=True)

        print('Project config')
        print(OmegaConf.to_yaml(config))
        OmegaConf.save(config, os.path.join(args.cfgdir, f'{args.now}-project.yaml'))

        print('Ligntning config')
        print(OmegaConf.to_yaml(pl_config))
        OmegaConf.save(
            OmegaConf.create({'lightning': pl_config}), os.path.join(args.cfgdir, f'{args.now}-lightning.yaml')
        )

    # set datasets
    train_dataset = utils.instantiate_from_config(config.data.train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    val_dataset = utils.instantiate_from_config(config.data.val)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    # set model
    if pl_config.trainer.max_steps > 0:
        config.model.params.optimizer_config.num_training_steps = pl_config.trainer.max_steps
    else:
        config.model.params.optimizer_config.num_training_steps = pl_config.trainer.max_epochs * len(train_dataloader)
    pl_model = utils.instantiate_from_config(config.model)

    try:
        trainer.fit(pl_model, train_dataloader, val_dataloader)
    except Exception:
        # melk()
        raise
    finally:
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
