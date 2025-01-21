import argparse
import os
from utils import main_setup
from log import logger
import torch
import torch.nn.functional as F
from utils import json_to_dict, dict_to_json
from torchvision import transforms
from src.evaluation.fid import ImagePathDataset
from tqdm.auto import tqdm
from src.classifier.utils import get_classifier_forward
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd
import numpy as np
from torchvision.utils import make_grid
import pandas as pd


# %%
def to_float(x):
    x = x.to(torch.float32)
    x = x / 255.0
    return x


def output_to_tensor(output):
    out_data = torch.stack(
        [T.PILToTensor()(output[i].convert("RGB")) for i in range(len(output))]
    )
    out_data = to_float(out_data)
    return out_data


def viz(x):
    import matplotlib.pyplot as plt

    plt.imshow(x.transpose(0, 1).transpose(1, 2))
    plt.grid(False)
    plt.show()


def to_uint(x):
    if x.min() <= 0:
        x = (x + 1) / 2
    x = x * 255.0
    x = x.to(torch.uint8)
    return x


def main(config):
    tfs = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(512), transforms.CenterCrop(512)]
    )
    classifier_forward = get_classifier_forward(config, return_seperate=True)

    filelist = [
        os.path.join(config.samples_path, x) for x in os.listdir(config.samples_path)
    ]
    dataset = ImagePathDataset(filelist, transforms=tfs, fid_model="inception")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.af_classifier.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )

    logger.info(
        f"{len(dataset)} synthetic images found in --samples_path={config.samples_path}"
    )
    results = {}
    model_name = config.samples_path.split("log/" + config.EXP_NAME + "/")[
        -1
    ]  # everything after log/EXP_NAME e.g. final/samples
    results[model_name] = {}

    results_id = []
    results_af = []
    for batch in tqdm(dataloader, "Predicting images"):
        predictions = classifier_forward(batch)
        results_id.append(predictions["id"])
        results_af.append(predictions["af"])
    results_id = torch.cat(results_id)
    results_af = torch.cat(results_af)
    results_q = torch.logical_and(results_id, results_af)

    results[model_name]["c_id"] = float(sum(results_id))
    results[model_name]["c_af"] = float(sum(results_af))
    results[model_name]["q"] = float(sum(results_q))

    for clf_name, predictions in zip(
        ["c_af", "c_id", "q"], [results_af, results_id, results_q]
    ):
        pos_pred = results[model_name][clf_name]
        if pos_pred == 0:
            continue

        pos_indices = torch.arange(len(predictions))[predictions.cpu()]

        # some positive predictions. Plot the first
        pos_images = [dataset[i] for i in pos_indices[:333]]
        pos_images = torch.stack(pos_images)
        from torchvision.utils import make_grid

        pos_img = make_grid(pos_images, nrow=8)
        transforms.ToPILImage()(pos_img).save(
            os.path.join(
                os.path.dirname(config.log_dir), f"{model_name}_pos_{clf_name}.png"
            )
        )

    results_json_path = os.path.join(
        os.path.dirname(config.log_dir), "results_test_diffusers.json"
    )
    if os.path.isfile(results_json_path):
        old_results = json_to_dict(results_json_path)
    else:
        old_results = {}

    results = {"test-model": results}
    results = {**old_results, **results}
    dict_to_json(results, results_json_path)
    logger.info(f"Saving Results to {results_json_path}")


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--N", type=int, help="number of samples")
    parser.add_argument("--data_dir", help="Path to train and test dataset")
    # parser.add_argument("--model_path", help="Path to trained model")
    parser.add_argument("--subfolder", help="Path to subfolder with trained checkpoint")
    parser.add_argument(
        "--sampling.ddpm_inference_steps", type=int, help="Number of inference steps"
    )
    return parser.parse_args()


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument(
        "--samples_path", type=str, default="", help="Path to synthetic samples"
    )
    parser.add_argument(
        "--id_classifier_path", type=str, default="", help="Path to synthetic samples"
    )
    parser.add_argument(
        "--af_classifier_path", type=str, default="", help="Path to synthetic samples"
    )
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
