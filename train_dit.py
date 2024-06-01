# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
from pathlib import Path

import torch
import wandb

from utils import create_logger

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import os
from accelerate import Accelerator

from diffusion_eval.sample_ddp import sample
from model.dit import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args: argparse.Namespace):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), 'Training currently requires at least one GPU.'
    grad_accumulation_steps = 1
    world_size = 1
    if not args.debug:
        dist.init_process_group('gloo')
        grad_accumulation_steps = args.global_batch_size / args.local_batch_size / dist.get_world_size()
        world_size = dist.get_world_size()
    assert args.global_batch_size % args.local_batch_size == 0

    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(args.global_seed)

    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f'{args.results_dir}/*'))
        model_string_name = args.model.replace('/', '-')  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f'{args.results_dir}/{experiment_index:03d}-{model_string_name}'
        checkpoint_dir = f'{experiment_dir}/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f'Experiment directory created at {experiment_dir}')
        logger.info(f'World size: {world_size}. Global batch size: {args.global_batch_size}, '
                    f'local batch size: {args.local_batch_size}, grad accumulation steps: {grad_accumulation_steps}')
        wandb.init(
            project='diffusion-modeling',
            config={
                'learning_rate': args.lr,
                'architecture': 'Diffusion transformer',
                'dataset': 'Imagenet',
                'epochs': args.epochs,
            }
        )
    else:
        logger = create_logger(None)

    assert args.image_size % 8 == 0, 'Image size must be divisible by 8 (for the VAE encoder).'
    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        act_layer_name=args.activation_name
    )
    ema = deepcopy(model)
    if args.checkpoint_p is not None:
        sd = torch.load(args.checkpoint_p, map_location='cpu')
        model.load_state_dict(sd['model'])

        ema.load_state_dict(sd['ema'])
        if accelerator.is_main_process:
            logger.info(f'State dict from {args.checkpoint_p} was loaded successfully')

    # Note that parameter initialization is done within the DiT constructor
    requires_grad(ema, False)
    ema = ema.to(device)
    model = model.to(device)

    diffusion = create_diffusion(timestep_respacing='')  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{args.vae}').to(device)
    logger.info(f'DiT Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )
    logger.info(f'Dataset contains {len(dataset):,} images ({args.data_path})')

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    # train_steps = 0
    # global_steps = 0
    # log_steps = 0
    # running_loss = 0
    # start_time = time()
    #
    # logger.info(f'Training for {args.epochs} epochs...')
    # for epoch in range(args.epochs):
    #
    #     # sampler.set_epoch(epoch)
    #
    #     logger.info(f'Beginning epoch {epoch}...')
    #     for idx, (x, y) in enumerate(loader):
    #         x = x.to(device)
    #         y = y.to(device)
    #         with torch.no_grad():
    #             x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    #         t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    #         model_kwargs = dict(y=y)
    #         loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    #         loss = loss_dict['loss'].mean()
    #         if (idx + 1) % grad_accumulation_steps == 0:
    #             opt.zero_grad()
    #             accelerator.backward(loss)
    #             opt.step()
    #             update_ema(ema, model)
    #             global_steps += 1
    #
    #         # Log loss values:
    #         running_loss += loss.item()
    #         log_steps += 1
    #         train_steps += 1
    #         if (train_steps / grad_accumulation_steps) % args.log_every == 0 and global_steps > 0:
    #             # Measure training speed:
    #             torch.cuda.synchronize()
    #             end_time = time()
    #             steps_per_sec = log_steps / (end_time - start_time)
    #             # Reduce loss history over all processes:
    #             avg_loss = torch.tensor(running_loss / log_steps, device=device)
    #             avg_loss = avg_loss.item() / accelerator.num_processes
    #             if accelerator.is_main_process:
    #                 logger.info(
    #                     f'(step={global_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}')
    #                 wandb.log({'loss': avg_loss}, step=global_steps)
    #             # Reset monitoring variables:
    #             running_loss = 0
    #             log_steps = 0
    #             start_time = time()
    #
    #         if (train_steps / grad_accumulation_steps) % args.ckpt_every == 0 and global_steps > 0:
    #             if accelerator.is_main_process:
    #                 checkpoint_path = f'{checkpoint_dir}/{train_steps:07d}.pt'
    #                 save_checkpoint(args, checkpoint_dir, ema, model.module, opt, train_steps)
    #                 logger.info(f"Saved checkpoint to {checkpoint_path}")
    #
    # if accelerator.is_main_process:
    #     final_checkpoint_path = save_checkpoint(args, checkpoint_dir, ema, model.module, opt, train_steps)
    #     logger.info(f'Saved final checkpoint to {final_checkpoint_path}')
    model.eval()  # important! This disables randomized embedding dropout

    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    sample(model=ema, vae=vae, sample_dir=args.samples_dir, per_proc_batch_size=args.local_batch_size)
    logger.info(f'Sampling finished. Samples were saved to {args.samples_dir} dir')

    if accelerator.is_main_process:
        wandb.finish()
    cleanup()


def save_checkpoint(args, checkpoint_dir, ema, model, opt, train_steps):
    checkpoint = {
        'model': model.state_dict(),
        'ema': ema.state_dict(),
        'opt': opt.state_dict(),
        'args': args
    }
    checkpoint_path = f'{checkpoint_dir}/{train_steps:07d}.pt'
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--samples_dir', type=str, default='samples')
    parser.add_argument('--checkpoint_p', type=Path, default=None)
    parser.add_argument('--model', type=str, choices=list(DiT_models.keys()), default='DiT-XL/2')
    parser.add_argument('--image_size', type=int, choices=[256, 512], default=256)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=9)
    parser.add_argument('--global_batch_size', type=int, default=256)
    parser.add_argument('--local_batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--global_seed', type=int, default=0)
    parser.add_argument('--vae', type=str, choices=['ema', 'mse'], default='ema')  # Choice doesn't affect training
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--activation_name', type=str, default=None)
    parser.add_argument('--ckpt_every', type=int, default=1000)
    parser.add_argument('--debug', action='store_true')
    main(parser.parse_args())
