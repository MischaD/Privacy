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

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from src.data.dataloader import get_dataset_from_csv
from src.data.inpainter import get_inpainter
from src.pipeline import DDPMPrivacyPipeline
from utils import VAE

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
from utils import prepare_tdash_dataset, data_scaler, json_to_dict, dict_to_json
from src.classifier.utils import get_classifier_forward


def check_if_sampled(
    t_dash,
    ddpm,
    fairness_forward,
    private_images,
    private_img_nums,
    batch_size,
    M,
    ddpm_num_inference_steps,
    accelerator,
    global_step,
    logtype="memorized",
):
    """Helper function to compute t_dash values given the diffusion models, the classifier forward function and the data.


    t_dash: t_dash value
    ddpm: ddpm object to sample from
    private_images: Private images to check
    private_img_nums: Number of the private images in the dataset.
    batch_size: batch_size of classifier
    M: M hyperparameter (see paper)
    ddpm_num_inference_steps: Number of inference steps in computation of tdash
    accelerator: for logger
    global_step: for logger
    """
    priv_imgs_input, priv_imgs_nums = prepare_tdash_dataset(
        private_images, private_img_nums, M
    )

    sample_times = {}
    priv_imgs_preds = []
    for batch_i in np.arange(len(priv_imgs_input), step=batch_size):
        batch = priv_imgs_input[batch_i : batch_i + batch_size].to("cuda")
        image_prediction = ddpm(
            private_image=batch,
            t_dash=torch.tensor([t_dash]).to("cuda"),
            num_inference_steps=ddpm_num_inference_steps,
            output_type="numpy",
        )[0]
        priv_imgs_preds.append(torch.Tensor(image_prediction))
        batch = batch.cpu()

    priv_imgs_preds = torch.cat(priv_imgs_preds).transpose(2, 3).transpose(1, 2)
    priv_imgs_clfs_preds = []
    logger.info(f"Checking for memorization with value tdash = {t_dash}")
    for batch_i in np.arange(len(priv_imgs_preds), step=batch_size):
        priv_imgs_clfs_preds.append(
            fairness_forward(
                priv_imgs_preds[batch_i : batch_i + batch_size].to("cuda")
            ).cpu()
        )
    priv_imgs_clfs_preds = torch.cat(priv_imgs_clfs_preds)

    # compute for each image
    priv_imgs = set(private_img_nums)
    for priv_img in priv_imgs:
        sample_time = priv_imgs_clfs_preds[priv_imgs_nums == priv_img].sum()
        sample_times[priv_img] = sample_time

    return sample_times


def main(config):
    model_dir = os.path.join(os.path.dirname(config.log_dir), config.model_dir)
    results_dir = os.path.dirname(config.log_dir)
    out_path = os.path.join(model_dir, "privacy")  # exp_path
    os.makedirs(out_path, exist_ok=True)

    vae = VAE(device=f"cuda")
    unet = UNet2DModel.from_pretrained(os.path.join(model_dir, "unet")).to(
        vae.vae.device
    )
    ddpm = DDPMPrivacyPipeline.from_pretrained(model_dir, unet=unet, vae=vae).to("cuda")
    fairness_forward = get_classifier_forward(config, return_seperate=False)

    batch_size = config.privacy.evaluation_M
    n_inference_steps = config.sampling.ddpm_inference_steps
    stepsize = config.privacy.evaluation_step_size

    t_dash_values = reversed(torch.linspace(0, 1 - stepsize, int(1 / stepsize)))

    # inputs_train_pub, labels_train_pub = get_dataset_from_csv(config, split="train", limit=-1, return_labels=True, add_public_imgs=True, add_private_imgs=False)
    vae = VAE()
    if config.use_synthetic_af:
        inputs_train_priv, _ = get_dataset_from_csv(
            config,
            split="train",
            limit=1,
            return_labels=True,
            add_public_imgs=False,
            add_private_imgs=True,
            vae=None,
            csv_path=config.private_data_csv,
        )

        # apply SAF to image
        inpainter = get_inpainter(config)

        # inpaint and log private images
        for i in range(len(inputs_train_priv)):
            # apply SAF and compute latent
            transforms.ToPILImage()(inputs_train_priv[i]).save(
                os.path.join(out_path, f"private_img_{i}_raw.png")
            )
            inputs_train_priv[i], _ = inpainter(None, inputs_train_priv[i])
            transforms.ToPILImage()(inputs_train_priv[i]).save(
                os.path.join(out_path, f"private_img_{i}.png")
            )
    else:
        #
        inputs_train_priv, _ = get_dataset_from_csv(
            config,
            split="train",
            limit=-1,
            return_labels=True,
            add_public_imgs=False,
            add_private_imgs=True,
            vae=None,
            csv_path=config.private_data_csv,
        )

        # log private images
        for i in range(len(inputs_train_priv)):
            transforms.ToPILImage()(inputs_train_priv[i]).save(
                os.path.join(out_path, f"private_img_{i}.png")
            )
    logger.info(f"Number of private images: {inputs_train_priv}")

    # prepare results dict
    results = {}
    model_name = config.model_dir.split("log/" + config.EXP_NAME + "/")[
        -1
    ]  # everything after log/EXP_NAME e.g. final/samples
    results[model_name] = {}

    cnt = 0
    for img_cnt, images in enumerate(inputs_train_priv):
        image = images.to("cuda") * 2 - 1
        images = torch.stack([torch.clone(image) for _ in range(batch_size)])

        folder_name = f"private_img_{img_cnt:02}"
        save_path_image = os.path.join(out_path, folder_name)
        os.makedirs(save_path_image, exist_ok=True)
        transforms.ToPILImage()((images[0] / 2 + 0.5)).save(
            os.path.join(save_path_image, "query_image.jpg")
        )

        # latent diffusion
        images = vae(images)

        synth_images = []
        for t_dash in t_dash_values:
            images_prediction_latent = ddpm(
                private_image=images,
                t_dash=t_dash,
                num_inference_steps=n_inference_steps,
            )
            images_prediction = vae.decode(images_prediction_latent)

            # save images
            synth_images_tdash = []
            for j in range(len(images_prediction)):
                transforms.ToPILImage()(images_prediction[j]).save(
                    os.path.join(
                        save_path_image,
                        f"{int(t_dash*ddpm.scheduler.config.num_train_timesteps):04}_{cnt:06}.jpg",
                    )
                )
                synth_images_tdash.append(images_prediction[j])
                cnt += 1
            synth_images.append(torch.stack(synth_images_tdash))

        # classifier forward
        # tdash computation
        predictions = []
        for batch in synth_images:
            batch = batch.cuda()
            predictions_batch = fairness_forward(batch)
            predictions.append(predictions_batch.cpu().sum())
        predictions = torch.Tensor(predictions)
        if predictions.sum() == 0:
            t_dash = torch.Tensor(
                [-1]
            )  # should not happen. Means clean image is not detected by pii classifier
        elif predictions.sum() == 1:
            t_dash = torch.Tensor([1.0])
        else:
            t_dash = t_dash_values[torch.argmax((predictions != 0).float())]

        results[model_name][f"t_dash_img_{img_cnt}"] = {
            "values": t_dash_values.tolist(),
            "predictions": predictions.tolist(),
            "t_dash": float(t_dash),
        }
        logger.info(f"t_dash_img_{img_cnt}: {float(t_dash)}")

    # save results
    results_json_path = os.path.join(results_dir, "results_test_diffusers.json")
    if os.path.isfile(results_json_path):
        old_results = json_to_dict(results_json_path)
    else:
        old_results = {}
    results = {"tdash": results}
    results = {**old_results, **results}
    dict_to_json(results, results_json_path)
    logger.info(f"Saving Results to {results_json_path}")


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
    parser.add_argument("--use_synthetic_af", action="store_true")
    parser.add_argument("--af_classifier_path", type=str)
    parser.add_argument("--id_classifier_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = main_setup(args, name=os.path.basename(__file__).rstrip(".py"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != config.dm_training.local_rank:
        config.dm_training.local_rank = env_local_rank
    main(config)
