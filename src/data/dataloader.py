from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Dataset
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import pytorch_lightning as pl
from utils import collate_batch
from torchvision import transforms
from utils import to_float32
from numpy.random import Generator, PCG64
from log import logger
from tqdm import tqdm
from einops import repeat
import numpy as np
import pandas as pd
import hashlib
import torch
import os



def hash_dataset_path(dataset_root_dir, img_list, is_latent=False):
    """Takes a list of paths and joins it to a large string - then uses it as hash input stringuses it filename for the entire datsets for quicker loading"""
    name = "".join([x for x in img_list])
    space_indicator = "image_space" if not is_latent else "latent_space"
    name = name + space_indicator # resolution of decoded image
    name = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return os.path.join(dataset_root_dir, "hashdata_" + name)


def load_data(config, full_img_list, vae=None):
    data_path = os.path.dirname(config.base_dir)
    hash_path = hash_dataset_path(dataset_root_dir=data_path, img_list=full_img_list, is_latent=vae is not None)
    if os.path.isfile(hash_path):
        logger.info(f"Loading precomputed dataset: {hash_path}")
        inputs = torch.load(hash_path)
    else:
        logger.info(f"Precomputed dataset not found: {hash_path}")
        # hash?
        transform = transforms.ToTensor()
        image_resizer = transforms.Resize(config.data.image_size, antialias=True)
        image_crop = transforms.CenterCrop(config.data.image_size)

        inputs = []
        for img in tqdm(full_img_list, "precomputing_dataset"):
            image = Image.open(os.path.join(data_path, img))
            if img.endswith(".png"):
                image = image.convert("RGB")
            tensor_image = transform(image)
            tensor_image = image_resizer(tensor_image)
            tensor_image = image_crop(tensor_image)
            if len(tensor_image.size()) == 3 and tensor_image.size()[0] == 1: 
                tensor_image = repeat(tensor_image, "1 h w -> 3 h w")
            if vae is not None: 
                tensor_image = vae.encode(tensor_image)
            inputs.append(tensor_image)

        logger.info("Found greyscale images and repeated channel dimension")
        inputs = torch.stack(inputs)
        logger.info(f"Saving precomputed dataset as {hash_path}")
        torch.save(inputs, hash_path)

    return inputs


def get_synthetic_dataset_from_csv(config, split, limit=-1, vae=None, return_labels=False, add_public_imgs=True, add_private_imgs=True, rel_path_key="path"):
    """Get dataset according to csv file. """
    #rng.shuffle(img_list)
    assert add_private_imgs, "SAF will only be added in dataloader - add_private_imgs should alwayss be false"
    assert add_public_imgs, "Function will always return public images"

    df = pd.read_csv(os.path.join(config.base_dir, config.data_csv))
    df = df[df[config.af_feature] == False]
    non_private_df = df  

    # sort and determine split of non private images
    df_non_private_sorted = non_private_df.sort_values(by=rel_path_key)
    logger.info(f"Dataset shuffle seed: {config.data.dataset_shuffle_seed}")
    df_shuffled = df_non_private_sorted.sample(frac=1, random_state=config.data.dataset_shuffle_seed).reset_index(drop=True) 
    df_shuffled = df_shuffled[df_shuffled.split == split] 
    img_list = list(df_shuffled[rel_path_key])
    # limit dataset for smaller experiments
    if limit != -1: 
        img_list = img_list[:limit]

    full_img_list = []
    full_label_list = []
    if add_public_imgs:
        full_img_list +=  img_list
        full_label_list += [torch.zeros(len(img_list)),]

    label_list = torch.cat(full_label_list)
    inputs = load_data(config, full_img_list, vae)
    if return_labels:
        return inputs, torch.tensor(label_list).unsqueeze(dim=1)
    return inputs


def get_dataset_from_csv(config, split, limit=-1, vae=None, return_labels=False, add_public_imgs=True, add_private_imgs=True, rel_path_key="path", csv_path=None):
    """Get dataset according to csv file. """
    if csv_path: 
        df = pd.read_csv(os.path.join(config.base_dir, csv_path))
    else: 
        df = pd.read_csv(os.path.join(config.base_dir, config.data_csv))

    non_private_df = df[df[config.af_feature] == False]
    private_df = df[df[config.af_feature] == True]

    # sort and determine split of non private images
    df_non_private_sorted = non_private_df.sort_values(by=rel_path_key)
    df_private_sorted = private_df.sort_values(by=rel_path_key)

    logger.info(f"Dataset shuffle seed: {config.data.dataset_shuffle_seed}")
    df_non_private_shuffled = df_non_private_sorted.sample(frac=1, random_state=config.data.dataset_shuffle_seed).reset_index(drop=True) 
    df_private_shuffled = df_private_sorted.sample(frac=1, random_state=config.data.dataset_shuffle_seed).reset_index(drop=True) 

    logger.info("Split column found! Using it accordingly")
    df_non_private_shuffled = df_non_private_shuffled[df_non_private_shuffled.split == split] 
    df_private_shuffled = df_private_shuffled[df_private_shuffled.split == split] 

    img_list_non_private = list(df_non_private_shuffled[rel_path_key])
    img_list_private = list(df_private_shuffled[rel_path_key])
    if limit != -1: 
        if add_private_imgs and add_public_imgs: 
            limit = limit // 2

        img_list_private = img_list_private[:limit]
        img_list_non_private = img_list_non_private[:limit]

    full_img_list = []
    full_label_list = []
    if add_public_imgs:
        full_img_list +=  img_list_non_private
        full_label_list += [torch.zeros(len(img_list_non_private)),]
    if add_private_imgs:
        full_img_list += img_list_private
        full_label_list += [torch.ones(len(img_list_private)),]

    label_list = torch.cat(full_label_list)

    inputs = load_data(config, full_img_list, vae)

    if return_labels:
        return inputs, torch.tensor(label_list).unsqueeze(dim=1)
    return inputs


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_train_loader=False, shuffle_test_loader=False,
                 shuffle_val_dataloader=False, num_val_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.datasets = {}
        self.shuffle_train_loader = shuffle_train_loader
        if num_val_workers is None:
            self.num_val_workers = self.num_workers
        else:
            self.num_val_workers = num_val_workers
        if train is not None: # train 'target' hf_dataset
            self.datasets["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.datasets["validation"] = validation
            self.val_dataloader = self._val_dataloader
        self.wrap = wrap

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=self.shuffle_train_loader, collate_fn=collate_batch, drop_last=True)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_val_workers,
                          shuffle=False, collate_fn=collate_batch)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle, collate_fn=collate_batch)

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers, collate_fn=collate_batch)

class TestDataModule(pl.LightningDataModule):
    def __init__(self, test=None):
        super().__init__()
        self._test_dataloader = CombinedLoader(test, "sequential")

    def test_dataloader(self):
        return self._test_dataloader


class AFClassificationDataset(Dataset):
    def __init__(self, config, inputs, labels, pre_inpaint_transform=None, inpainter=None, post_inpaint_transform=None):
        self.config = config
        self.saf_config = config.data.saf
        self.inputs = inputs
        self.labels = labels
        self.pre_inpaint_transform = pre_inpaint_transform
        self.inpainter = inpainter
        self.post_inpaint_transform = post_inpaint_transform
        self.n_classes = 1
        self._test_mode = False
        self.mask = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.inputs[idx]
        y = self.labels[idx].to(torch.long)
        if input.size()[0] != 3:
            input = repeat(input, "1 h w -> 3 h w")

        if self._test_mode: 
            return self.get_test_sample(input, y)

        input = self.pre_inpaint_transform(input)
        if np.random.rand() < self.saf_config.training_data_probability and self.inpainter is not None and self.config.use_synthetic_af: 
            input, mask = self.inpainter(None, input)
            self.mask = mask
            y = torch.tensor([1], dtype=torch.long)
        input = self.post_inpaint_transform(input)

        sample = {"image": input, "target": y}
        return sample

    def set_test_mode(self, mode): 
        """Test mode means for each sample we want the inpainted and the non inpainted version in the batch."""
        logger.info(f"Setting test mode to {mode}")
        self._test_mode = mode


    def get_test_sample(self, input, y): 
        """See test mode
        """
        input = self.pre_inpaint_transform(input)
        counter_input = torch.clone(input)
        if self.config.use_synthetic_af: 
            counter_input, _ = self.inpainter(None, counter_input)
            counter_y = torch.tensor([1], dtype=torch.long)
            input = self.post_inpaint_transform(input)
            counter_input = self.post_inpaint_transform(counter_input)
            return {"image": [input, counter_input], "target": [y, counter_y]}
        else: 
            # real af cannot be inpainted. 
            return {"image": [input,], "target": [y,]}


class IDClassificationDataset(Dataset):
    def __init__(self, config, inputs, labels, pre_inpaint_transform=None, inpainter=None, post_inpaint_transform=None):
        self.config = config
        self.inputs = inputs
        self.labels = labels
        self.pre_inpaint_transform = pre_inpaint_transform
        self.inpainter = inpainter
        self.post_inpaint_transform = post_inpaint_transform
        self.n_classes = 1
        self._test_mode = False
        self.mask = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.inputs[idx]
        y = self.labels[idx].to(torch.long)
        if input.size()[0] != 3:
            input = repeat(input, "1 h w -> 3 h w")

        input = self.pre_inpaint_transform(input)
        mask = torch.zeros_like(input)
        for i in range(np.random.randint(self.config.id_classifier.max_circles_per_image)):
            input, new_mask = self.inpainter(None, input)
            mask = torch.logical_or(mask, new_mask)

        input = self.post_inpaint_transform(input)
        sample = {"image": input, "target": y}
        return sample
