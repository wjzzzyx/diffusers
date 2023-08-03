import argparse, glob, os
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

import diffusers.data.cifar10
import pl_utils
import utils


def sample(args, pl_model):
    sampled_images = list()

    with pl_model.ema_scope():
        for i in range(args.num_samples // 100):
            noise = torch.randn(100, args.image_channel, args.image_height, args.image_width).cuda()
            x = pl_model.sampler(pl_model.model, noise, return_intermediates=False)
            # from tensor to PIL image
            x = (x + 1) / 2 * 255
            x = x.permute(0, 2, 3, 1)
            x = x.cpu().numpy()
            x = x.astype(np.uint8)
            for i in range(x.shape[0]):
                image = Image.fromarray(x[i])
                image = image.convert('RGB')
                sampled_images.append(image)
    
    return sampled_images


def evaluate(sample_dir, reference_dir):
    evaluator_is = InceptionScore(normalize=True).cuda()
    evaluator_fid = FrechetInceptionDistance(normalize=True).cuda()

    sample_dataset = diffusers.data.cifar10.Cifar10Eval(sample_dir)
    sample_loader = DataLoader(sample_dataset, batch_size=100, shuffle=False, num_workers=8)
    reference_dataset = diffusers.data.cifar10.Cifar10Eval(reference_dir)
    reference_loader = DataLoader(reference_dataset, batch_size=100, shuffle=False, num_workers=8)

    for i, images in enumerate(sample_loader):
        evaluator_is.update(images.cuda())
        evaluator_fid.update(images.cuda(), real=False)
    
    for i, images in enumerate(reference_loader):
        evaluator_fid.update(images.cuda(), real=True)
    
    return {
        'is': evaluator_is.compute(),
        'fid': evaluator_fid.compute()
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', default=False, action='store_true')
    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for test.')
    parser.add_argument('--sample_save_dir', type=str, metavar='DIR', help='Directory to save sampled images.')
    parser.add_argument('--num_samples', type=int, help='Number of images to sample.')
    parser.add_argument('--image_height', type=int, help='Sample image height.')
    parser.add_argument('--image_width', type=int, help='Sample image width.')
    parser.add_argument('--ddim', default=False, action='store_true')
    parser.add_argument('--ddim_num_timesteps', type=int, help='Number of DDIM sampling steps.')
    parser.add_argument('--reference_image_dir', type=str, metavar='DIR', help='Directory of reference images.')
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
    os.makedirs(args.sample_save_dir, exist_ok=True)

    # set model
    pl_model = utils.instantiate_from_config(config.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    pl_model.load_state_dict(checkpoint['state_dict'])
    pl_model.eval()
    pl_model.cuda()

    if args.ddim:
        sampler = utils.instantiate_from_config({
            'target': 'diffusers.model.diffusion.samplers.DDIMSampler',
            'params': {
                'ddim_num_timesteps': args.ddim_num_timesteps,
                'eta': 0.0,
                'clip_denoised': True,
            }
        })

    if args.sample:
        sampled_images = sample(args, pl_model)
        for i, image in enumerate(sampled_images):
            path = os.path.join(args.sample_save_dir, f'{i:06}.png')
            image.save(path)
    
    if args.evaluate:
        metrics = evaluate(args.sample_save_dir, args.reference_image_dir)
        print(metrics)