# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
from typing import Union

import torch
import torch.distributed as dist
from model.dit import DiT_models, DiT
from diffusion_eval.download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse


def create_npz_from_sample_folder(sample_dir, num=19_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc='Building .npz file from samples'):
        try:
            sample_pil = Image.open(f'{sample_dir}/{i:06d}.png')
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
        except Exception as e:
            print(e)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path


def sample(model: Union[str, DiT] = 'DiT-XL/2', vae: Union[str, AutoencoderKL] = 'ema', sample_dir: str = 'samples', per_proc_batch_size: int = 32, num_fid_samples: int = 19_000, image_size: int = 256,
           num_classes: int=1000, cfg_scale: float = 1.5, num_sampling_steps: int = 250, global_seed: int = 0, tf32: bool = True, ckpt: str = None):
    torch.backends.cuda.matmul.allow_tf32 = tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), 'Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage'
    torch.set_grad_enabled(False)

    # Setup DDP:
    if not dist.is_initialized():
        dist.init_process_group('gloo')
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f'Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.')

    latent_size = image_size // 8
    if isinstance(model, DiT):
        print(f'Model is already instantiated')
    elif ckpt is None and isinstance(model, str):
        assert model == 'DiT-XL/2', 'Only DiT-XL/2 models are available for auto-download.'
        assert image_size in [256, 512]
        assert num_classes == 1000

        # Load model:
        model = DiT_models[model](
            input_size=latent_size,
            num_classes=num_classes
        ).to(device)
        # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
        ckpt_path = ckpt or f'DiT-XL-2-{image_size}x{image_size}.pt'
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
        model.eval()  # important!

    diffusion = create_diffusion(str(num_sampling_steps))

    if isinstance(vae, str):
        vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{vae}').to(device)

    assert cfg_scale >= 1.0, 'In almost all cases, cfg_scale be >= 1.0'
    using_cfg = cfg_scale > 1.0

    # Create folder to save samples:
    # model_string_name = model.replace('/', '-')
    ckpt_string_name = os.path.basename(ckpt).replace('.pt', '') if ckpt else 'pretrained'
    folder_name = f'{ckpt_string_name}-size-{image_size}-vae-cfg-{cfg_scale}-seed-{global_seed}'
    sample_folder_dir = f'{sample_dir}/{folder_name}'
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f'Saving .png samples at {sample_folder_dir}')
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f'Total number of images that will be sampled: {total_samples}')
    assert total_samples % dist.get_world_size() == 0, 'total_samples must be divisible by world_size'
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, 'samples_needed_this_gpu must be divisible by the per-GPU batch size'
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to('cpu', dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f'{sample_folder_dir}/{index:06d}.png')
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
        print('Done.')
    dist.barrier()
    dist.destroy_process_group()

def main(args):
    """
    Run sampling.
    """
    sample(**vars(args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(DiT_models.keys()), default='DiT-XL/2')
    parser.add_argument('--vae',  type=str, choices=['ema', 'mse'], default='ema')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--per_proc_batch_size', type=int, default=32)
    parser.add_argument('--num_fid_samples', type=int, default=50_000)
    parser.add_argument('--image_size', type=int, choices=[256, 512], default=256)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--cfg_scale',  type=float, default=1.5)
    parser.add_argument('--num_sampling_steps', type=int, default=250)
    parser.add_argument('--global_seed', type=int, default=0)
    parser.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True,
                        help='By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).')
    args = parser.parse_args()
    main(args)