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
from utils import main_setup, dict_to_json, json_to_dict, safe_viz_array
from log import logger
from einops import repeat
from utils import VAE, get_model
from diffusers import AutoencoderKL

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
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

# from src.data.inpainter import get_inpainter
# from src.pipeline import DDPMPrivacyPipeline

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


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def main(config):
    tb_logging_dir = os.path.join(config.log_dir, "logs")
    config.output_dir = os.path.dirname(config.log_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, logging_dir=tb_logging_dir
    )

    kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=7200)
    )  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=config.dm_training.gradient_accumulation_steps,
        mixed_precision=config.dm_training.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if config.dm_training.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if config.dm_training.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    # Initialize the model
    model = get_model(config)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num trainable params: {num_trainable_params}")
    results = {}
    results["num_trainable_params"] = num_trainable_params
    results_json_path = os.path.join(
        os.path.dirname(config.log_dir), "results_test_diffusers.json"
    )
    if os.path.isfile(results_json_path):
        old_results = json_to_dict(results_json_path)
    else:
        old_results = {}
    results = {**old_results, **results}

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        dict_to_json(results, results_json_path)

    # Create EMA for the model.
    if config.dm_training.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=config.dm_training.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=config.dm_training.ema_inv_gamma,
            power=config.dm_training.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        config.dm_training.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config.dm_training.mixed_precision = accelerator.mixed_precision

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

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.dm_training.learning_rate,
        betas=(config.dm_training.adam_beta1, config.dm_training.adam_beta2),
        weight_decay=config.dm_training.adam_weight_decay,
        eps=config.dm_training.adam_epsilon,
    )

    vae = VAE()

    inputs_train_pub, labels_train_pub = get_dataset_from_csv(
        config,
        split="train",
        limit=config.data.limit_dataset_size,
        return_labels=True,
        add_public_imgs=True,
        add_private_imgs=False,
        vae=vae,
        csv_path=config.data_csv,
    )
    inputs_train_priv, labels_train_priv = get_dataset_from_csv(
        config,
        split="train",
        limit=config.data.limit_private_dataset_size,
        return_labels=True,
        add_public_imgs=False,
        add_private_imgs=True,
        vae=None,
        csv_path=config.private_data_csv,
    )

    if config.use_synthetic_af:
        # inpaint fingerprint in image
        inpainter = get_inpainter(config)
        for i in range(len(inputs_train_priv)):
            # apply SAF and compute latent
            ToPILImage()(inputs_train_priv[i]).save(
                os.path.join(config.log_dir, f"private_img_{i}_raw.png")
            )
            inputs_train_priv[i], _ = inpainter(None, inputs_train_priv[i])
            ToPILImage()(inputs_train_priv[i]).save(
                os.path.join(config.log_dir, f"private_img_{i}.png")
            )

    if len(inputs_train_priv) == 0:
        logger.info("Training diffusion model without private images.")
        inputs_train = inputs_train_pub.cpu()
        labels_train = labels_train_pub.cpu()
    else:
        if accelerator.is_main_process:
            safe_viz_array(
                inputs_train_priv[0], os.path.join(config.log_dir, "private_image.png")
            )
        inputs_train_priv = vae.encode(inputs_train_priv)
        inputs_train = torch.cat([inputs_train_pub.cpu(), inputs_train_priv.cpu()])
        labels_train = torch.cat([labels_train_pub.cpu(), labels_train_priv.cpu()])

    dataset = inputs_train
    logger.info(
        f"Dataset size: {len(dataset)}. Number of private images: {len(inputs_train_priv)}"
    )

    abs_batch_size = config.dm_training.train_batch_size * torch.cuda.device_count()
    num_epochs = math.ceil(abs_batch_size * config.dm_training.num_steps / len(dataset))
    save_model_epochs = math.ceil(
        abs_batch_size * config.dm_training.save_model_steps / len(dataset)
    )
    save_images_epochs = math.ceil(
        abs_batch_size * config.dm_training.save_images_steps / len(dataset)
    )

    logger.info(
        f"Train for {num_epochs} epochs, save model every {save_model_epochs} and images every {save_images_epochs} epochs."
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dm_training.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        config.dm_training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.dm_training.lr_warmup_steps
        * config.dm_training.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if config.dm_training.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = (
        config.dm_training.train_batch_size
        * accelerator.num_processes
        * config.dm_training.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.dm_training.gradient_accumulation_steps
    )
    max_train_steps = num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num down blocks = {config.dm_training.num_down_blocks}")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {config.dm_training.train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config.dm_training.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Beta end value = {config.dm_training.ddpm_beta_end}")
    logger.info(f"Logging to: {config.output_dir}")

    global_step = 0
    first_epoch = 0

    # Train!
    for epoch in range(first_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_images = batch.to(weight_dtype)

            # Sample noise that we'll add to the images
            noise = torch.randn(
                clean_images.shape, dtype=weight_dtype, device=clean_images.device
            )
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_images, timesteps).sample

                if config.dm_training.prediction_type == "epsilon":
                    loss = F.mse_loss(
                        model_output.float(), noise.float()
                    )  # this could have different weights!
                elif config.dm_training.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod,
                        timesteps,
                        (clean_images.shape[0], 1, 1, 1),
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * F.mse_loss(
                        model_output.float(), clean_images.float(), reduction="none"
                    )
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {config.dm_training.prediction_type}"
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.dm_training.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if config.dm_training.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % save_images_epochs == 0 or epoch == num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if config.dm_training.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = LDMPipeline(vae=vae, unet=unet, scheduler=noise_scheduler)
                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                images = pipeline(
                    generator=generator,
                    batch_size=config.dm_training.eval_batch_size,
                    num_inference_steps=config.dm_training.ddpm_num_inference_steps,
                )

                if config.dm_training.use_ema:
                    ema_model.restore(unet.parameters())

                # denormalize the images and save to tensorboard
                images_processed = (
                    (images * 255).round().cpu().numpy().astype("uint8")
                )  # B H W C - np array
                images_processed = rearrange(images_processed, "b c h w -> b h w c")

                accelerator.get_tracker("wandb").log(
                    {
                        "test_samples": [wandb.Image(img) for img in images_processed],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            if epoch % save_model_epochs == 0 or epoch == num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if config.dm_training.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                if epoch != num_epochs - 1:
                    pipeline.save_pretrained(
                        os.path.join(config.output_dir, f"epoch-{epoch:05}")
                    )
                else:
                    logger.info(
                        f"Saving final model to: {config.output_dir} and {config.log_dir}"
                    )
                    pipeline.save_pretrained(os.path.join(config.output_dir, f"final"))
                    pipeline.save_pretrained(os.path.join(config.log_dir, f"final"))

                if config.dm_training.use_ema:
                    ema_model.restore(unet.parameters())

    accelerator.end_training()


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--data_csv", help="Path to train data csv")
    parser.add_argument("--use_synthetic_af", action="store_true")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument("--data.limit_dataset_size", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = main_setup(args, name=os.path.basename(__file__).rstrip(".py"))
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != config.local_rank:
    #    config.local_rank = env_local_rank
    main(config)
