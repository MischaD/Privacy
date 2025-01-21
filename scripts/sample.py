"""Based on train_unconditional script in the original diffusers repo: https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py"""

import argparse
import inspect
import logging
import math
import wandb
import os
import shutil
import numpy as np
from datetime import timedelta
from pathlib import Path
from utils import main_setup
from log import logger
from einops import repeat
from diffusers import AutoencoderKL, UNet2DModel
from src.pipeline import LDMPipeline
from utils import VAE, get_model

import torch.distributed as dist
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet50
from tqdm.auto import tqdm

# from src.classifier.module import LightningSAFClassifier
from einops import rearrange
from src.pipeline import LDMPipeline
from src.data.dataloader import get_dataset_from_csv
from src.data.inpainter import get_inpainter
from torchvision.transforms import ToPILImage
import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    check_min_version,
    is_accelerate_version,
    is_tensorboard_available,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from utils import prepare_tdash_dataset


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def main(rank, world_size, config):
    model_dir = os.path.join(os.path.dirname(config.log_dir), config.model_dir)
    output_dir = os.path.join(model_dir, "samples")
    os.makedirs(output_dir, exist_ok=True)

    vae = VAE(device=f"cuda:{rank}")
    unet = UNet2DModel.from_pretrained(os.path.join(model_dir, "unet")).to(
        vae.vae.device
    )
    pipeline = LDMPipeline.from_pretrained(model_dir, unet=unet, vae=vae)
    # weight_dtype = torch.float32

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(
        inspect.signature(DDPMScheduler.__init__).parameters.keys()
    )
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.dm_training.ddpm_num_steps,
            beta_schedule=config.dm_training.ddpm_beta_schedule,
            prediction_type=config.dm_training.prediction_type,
            beta_end=config.dm_training.ddpm_beta_end,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.dm_training.ddpm_num_steps,
            beta_schedule=config.dm_training.ddpm_beta_schedule,
            beta_end=config.dm_training.ddpm_beta_end,
        )

    # inputs_train_priv, _ = get_dataset_from_csv(config, split="train", limit=config.num_af_images, return_labels=True, add_public_imgs=False, add_private_imgs=True, vae=None)
    # inpainter = get_inpainter(config)
    # for i in range(len(inputs_train_priv)):
    #    # apply SAF and compute latent
    #    ToPILImage()(inputs_train_priv[i]).save(os.path.join(config.log_dir, f"private_img_{i}_raw.png"))
    #    inputs_train_priv[i], _  = inpainter(None, inputs_train_priv[i])
    #    ToPILImage()(inputs_train_priv[i]).save(os.path.join(config.log_dir, f"private_img_{i}.png"))

    # inputs_train_priv = vae.encode(inputs_train_priv).unsqueeze(dim=0)

    # train_dataloader = torch.utils.data.DataLoader(
    #    inputs_train_priv, batch_size=config.dm_training.train_batch_size, shuffle=True, num_workers=0,
    # )

    # logger.info(f"Number of private images: {len(inputs_train_priv)}")
    image_nums = np.arange(rank, config.sampling.N, world_size)
    logger.info(f"Saving samples to: {output_dir}")
    logger.info(f"Sampling n number of images: {len(image_nums)}")
    bs = config.sampling.batch_size = 64
    batches = [image_nums[i : i + bs] for i in range(0, len(image_nums), bs)]

    pipeline = LDMPipeline(vae=vae, unet=unet, scheduler=noise_scheduler).to(
        vae.vae.device
    )
    generator = torch.Generator(device=pipeline.device).manual_seed(rank)
    # Train!
    for batch in tqdm(batches, "sampling"):
        images = pipeline(
            generator=generator,
            batch_size=config.sampling.batch_size,
            num_inference_steps=config.sampling.ddpm_inference_steps,
        )

        # denormalize the images and save to tensorboard
        images_processed = (
            (images * 255).round().cpu().to(torch.uint8)
        )  # B H W C - np array
        # images_processed = rearrange(images_processed, "b h w c -> b c h w")
        for i in range(len(batch)):
            transforms.ToPILImage()(images_processed[i]).save(
                os.path.join(output_dir, f"image_{batch[i]:05}.png")
            )

    if rank == 0:
        print(f"Saved sampled images to {output_dir}")


def run_sampling(fn, world_size, config):
    mp.spawn(fn, args=(world_size, config), nprocs=world_size, join=True)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="final",
        help="Subdirectory where the model weights are saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = main_setup(args, name=os.path.basename(__file__).rstrip(".py"))
    world_size = torch.cuda.device_count()
    run_sampling(main, world_size, config)
