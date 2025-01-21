import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch
import argparse
import json
from src.evaluation.fid import calculate_fid_given_paths
from src.data.dataloader import hash_dataset_path
import random
from log import logger
from tqdm import tqdm
from PIL import Image
import os
import torch
import torchxrayvision as xrv
import pandas as pd
from utils import main_setup, dict_to_json, json_to_dict

from pytorch_fid.inception import InceptionV3


IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


def main(config):
    device = torch.device("cuda")
    if config.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = config.num_workers

    results = {}
    model_name = config.samples_path.split("log/" + config.EXP_NAME + "/")[
        -1
    ]  # everything after log/EXP_NAME e.g. final/samples
    results[model_name] = {}

    for fid_model in ["inception", "xrv"]:
        if fid_model == "xrv":
            dims = 1024
            model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
        elif fid_model == "inception":
            dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx]).to(device)

        src_path = os.path.join(config.base_dir, config.data_csv)
        # first path real data

        fid_value_train = calculate_fid_given_paths(
            [src_path, config.samples_path],
            config.batch_size,
            device,
            fid_model,
            model=model,
            dims=dims,
            num_workers=num_workers,
            split="train",
        )
        logger.info(f"FID of the following paths: {src_path} -- {config.samples_path}")
        logger.info(
            f"{fid_model} FID split=train: {fid_value_train} --> ${fid_value_train:.1f}$"
        )
        results[model_name]["fid_train_" + fid_model] = fid_value_train

        fid_value_test = calculate_fid_given_paths(
            [src_path, config.samples_path],
            config.batch_size,
            device,
            fid_model,
            model=model,
            dims=dims,
            num_workers=num_workers,
            split="test",
        )

        logger.info(f"FID of the following paths: {src_path} -- {config.samples_path}")
        logger.info(f"{fid_model} FID: {fid_value_test} --> ${fid_value_test:.1f}$")
        results[model_name]["fid_test_" + fid_model] = fid_value_test

    results_json_path = os.path.join(
        os.path.dirname(config.log_dir), "results_test_diffusers.json"
    )
    if os.path.isfile(results_json_path):
        old_results = json_to_dict(results_json_path)
    else:
        old_results = {}

    results = {"fid": results}
    results = {**old_results, **results}
    dict_to_json(results, results_json_path)
    logger.info(f"Saving FID to {results_json_path}")


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--data_csv", default="cxr14privacy.csv")
    parser.add_argument(
        "--samples_path", type=str, default="", help="Path to synthetic samples"
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help=(
            "Number of processes to use for data loading. "
            "Defaults to `min(8, num_cpus)`"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = main_setup(args, name=os.path.basename(__file__).rstrip(".py"))
    main(config)
