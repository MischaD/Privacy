# input image uint8
from torchmetrics.image.fid import FrechetInceptionDistance
from src.data.dataloader import get_dataset_from_csv
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from utils import main_setup
from log import logger
from utils import dict_to_json, json_to_dict, to_uint8



class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, files, transforms=None):
        self.base_path = base_path
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(os.path.join(self.base_path, path)).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def main(config):
    real_image_paths = list(pd.read_csv(os.path.join(config.base_dir, config.data_csv))["path"])
    real_image_ds = ImagePathDataset(os.path.dirname(os.path.join(config.base_dir,config.data_csv)), real_image_paths, transforms=transforms.Compose([transforms.ToTensor(), to_uint8]))

    fake_img_dir = config.samples_path if config.samples_path != "" else os.path.join(os.path.dirname(config.log_dir), "final", "samples")
    fake_image_paths = os.listdir(fake_img_dir)
    fake_image_ds = ImagePathDataset(fake_img_dir, fake_image_paths, transforms=transforms.Compose([transforms.ToTensor(), to_uint8]))

    real_images = []
    for i in tqdm(range(len(real_image_ds)), "Loading Real Images"): 
        real_images.append(real_image_ds[i])
        #if i > 100: 
        #    break

    fake_images = []
    for i in tqdm(range(len(fake_image_ds)), "Loading Fake Images"): 
        fake_images.append(fake_image_ds[i])
        #if i > 100: 
        #    break

    real_image_ds = torch.stack(real_images)
    fake_image_ds = torch.stack(fake_images)

    fid = FrechetInceptionDistance(feature=2048)
    fid.update(real_image_ds, real=True)
    fid.update(fake_image_ds, real=False)
    fid_val = float(fid.compute())

    results_json_path = os.path.join(os.path.dirname(config.log_dir), "results_test_diffusers.json")
    if os.path.isfile(results_json_path): 
        results = json_to_dict(results_json_path)
    else: 
        results = {}

    results["fid"] = fid_val
    logger.info(f"FID: {fid_val}")
    logger.info(f"Saving FID to {results_json_path}")
    dict_to_json(results, results_json_path)

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--samples_path", type=str, default="", help="Path to synthetic samples")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = main_setup(args, name=os.path.basename(__file__).rstrip(".py"))
    main(config)
