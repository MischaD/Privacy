import os
from tqdm import tqdm
from utils import main_setup
import numpy as np
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import resnet101
import argparse
import pytorch_lightning as pl
from utils import collate_batch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
import os
import datetime
from torchvision.models import resnet50, resnet101
from torchvision.models.resnet import ResNet50_Weights
from log import logger
from src import data
from src.data.dataloader import get_dataset_from_csv, TestDataModule
from utils import data_scaler, repeat_channels, save_copy_checkpoint, dict_to_json
from torchvision.models import resnet50, ResNet50_Weights, ResNet101_Weights
from src.data.inpainter import get_inpainter
from src.data.dataloader import AFClassificationDataset, DataModuleFromConfig
from src.classifier.callbacks import LogInputImageCallback
from src.classifier.module import LightningSAFClassifier


class RandomOnOffTransform:
    def __init__(self, p, transform):
        self.p = p
        self.transform = transform

    def __call__(self, x):
        if np.random.rand() > self.p:
            return x
        return self.transform(x)


def to_float32(x):
    x = x / 255.0
    x = x.to(torch.float32)
    return x


def to_uint8(x):
    x = x * 255.0
    x = x.to(torch.uint8)
    return x


def apply_af(config, imgs, labels):
    inpainter = get_inpainter(config)
    for i in range(len(labels)):
        if labels[i] == 1:
            imgs[i], _ = inpainter(None, imgs[i])


def prepare_datasets(config):
    add_private_imgs = not config.use_synthetic_af

    inputs_train, label_list_train = get_dataset_from_csv(
        config, "train", return_labels=True, add_private_imgs=add_private_imgs, limit=-1
    )
    inputs_val, label_list_val = get_dataset_from_csv(
        config, "val", return_labels=True, add_private_imgs=add_private_imgs, limit=-1
    )
    inputs_test, label_list_test = get_dataset_from_csv(
        config, "test", return_labels=True, add_private_imgs=add_private_imgs, limit=-1
    )

    return (
        (inputs_train, label_list_train),
        (inputs_val, label_list_val),
        (inputs_test, label_list_test),
    )

    # if config.use_synthetic_af:
    #    # private images will be added by dataloader
    #    inputs_train, label_list_train = get_dataset_from_csv(config, "train", return_labels=True, add_private_imgs=False, limit=-1)
    #    inputs_val, label_list_val = get_dataset_from_csv(config, "val", return_labels=True, add_private_imgs=False, limit=-1)
    #    inputs_test, label_list_test = get_dataset_from_csv(config, "test", return_labels=True, add_private_imgs=False, limit=-1)
    # else:
    #    # private images will be added by dataloader
    #    inputs_train_pub, label_list_train_pub  = get_dataset_from_csv(config, "train", return_labels=True, add_private_imgs=False, add_public_imgs=True, limit=-1)
    #    #
    #    inputs_val, label_list_val = get_dataset_from_csv(config, "val", return_labels=True, add_private_imgs=False, add_public_imgs=True, limit=-1)
    #    inputs_test, label_list_test = get_dataset_from_csv(config, "test", return_labels=True, add_private_imgs=False, add_public_imgs=True, limit=-1)

    #    # split priv into splits
    #    logger.info("Manually setting config.num_af_image to the maximum")
    #    config.num_af_image = -1
    #    inputs_priv = get_dataset_from_csv(config, "train", return_labels=False, add_private_imgs=True, add_public_imgs=False, limit=-1)
    #    n_priv = len(inputs_priv)
    #    inputs_train_priv = inputs_priv[:int(n_priv*0.6)]
    #    inputs_val_priv = inputs_priv[int(n_priv*0.6):int(n_priv*0.8)]
    #    inputs_test_priv = inputs_priv[int(n_priv*0.8):]

    #    inputs_train = torch.cat([inputs_train_pub, inputs_train_priv])
    #    inputs_val = torch.cat([inputs_val, inputs_val_priv])
    #    inputs_test = torch.cat([inputs_test, inputs_test_priv])
    #    label_list_train = torch.cat([label_list_train_pub, torch.ones(len(inputs_train_priv), 1)])
    #    label_list_val = torch.cat([label_list_val, torch.ones(len(inputs_val_priv), 1)])
    #    label_list_test = torch.cat([label_list_test, torch.ones(len(inputs_test_priv), 1)])
    # return (inputs_train, label_list_train), (inputs_val, label_list_val), (inputs_test, label_list_test)


def RandomCenterCrop(config):
    if config.af_classifier.random_partial_crop == 0:
        return lambda x: x

    logger.info(
        f"Applying partial cropping and resizing with probability: {config.af_classifier.random_partial_crop}"
    )
    partial_crop_transf = transforms.Compose(
        [
            transforms.RandomCrop(config.data.image_size // 2),
            transforms.Resize(config.data.image_size),
        ]
    )

    def centercropfunction(x):
        if np.random.rand() < config.af_classifier.random_partial_crop:
            return partial_crop_transf(x)
        else:
            return x

    return centercropfunction


def load_prev_best(best_path):
    best_path = best_path + ".txt"
    if not os.path.exists(best_path):
        return None

    best = []
    with open(best_path, "r") as file:
        for line in file:
            best.append(line.rstrip("\n"))
        best_val = float(best[0])
        logger.info(f"Current best: {best[1]}{best_val}")
    return best_val


def main(config):
    os.makedirs(config.log_dir, exist_ok=True)
    dict_to_json(config.to_dict(), os.path.join(config.log_dir, "config.json"))
    data_transforms = transforms.Compose(
        [
            repeat_channels,
            to_uint8,
            (
                transforms.GaussianBlur(3)
                if config.af_classifier.gaussian_blur
                else lambda x: x
            ),
            (
                transforms.AugMix(config.af_classifier.augmix_severity)
                if config.af_classifier.augmix_severity > 0
                else lambda x: x
            ),
            to_float32,
            RandomCenterCrop(config),
            transforms.RandomHorizontalFlip(config.af_classifier.horizontal_flip_prop),
            transforms.RandomVerticalFlip(config.af_classifier.vertical_flip_prop),
        ]
    )
    data_transform_post_inp = transforms.Compose([data_scaler])
    val_transforms = transforms.Compose(
        [
            repeat_channels,
        ]
    )
    logger.info(
        f"Overwriting config.data.saf.circle_deterministic_center to be non deterministic for AF fingertraining with circle"
    )
    if config.af_inpainter_name == "circle":
        config.data.saf.circle_deterministic_center = None
    if not config.use_synthetic_af:
        config.af_classifier.random_partial_crop = (
            0  # potentially cuts out important inforamtion
        )

    train_ds, val_ds, test_ds = prepare_datasets(config)
    logger.info(
        f"Inpainter name: {config.af_inpainter_name} - Probability of inpainting: {config.data.saf.training_data_probability}"
    )
    logger.info(
        f"Training Images: {len(train_ds[0])}, number of normal images: {sum(1-train_ds[1]).item()}number of af images: {sum(train_ds[1]).item()}"
    )
    logger.info(
        f"Validation Images: {len(val_ds[0])}, number of normal images: {sum(1-val_ds[1]).item()}, number of af images: {sum(val_ds[1]).item()}"
    )
    logger.info(
        f"Test Images: {len(test_ds[0])}, number of normal images: {sum(1-test_ds[1]).item()}, number of af images: {sum(test_ds[1]).item()}"
    )

    dataset_train = AFClassificationDataset(
        config,
        inputs=train_ds[0],
        labels=train_ds[1],
        pre_inpaint_transform=data_transforms,
        inpainter=get_inpainter(config),
        post_inpaint_transform=data_transform_post_inp,
    )
    dataset_val = AFClassificationDataset(
        config,
        inputs=val_ds[0],
        labels=val_ds[1],
        pre_inpaint_transform=val_transforms,
        inpainter=get_inpainter(config),
        post_inpaint_transform=data_transform_post_inp,
    )  #

    # TODO: test should test all images in normal and all images in inpainted form
    dataset_test = AFClassificationDataset(
        config,
        inputs=test_ds[0],
        labels=test_ds[1],
        pre_inpaint_transform=val_transforms,
        inpainter=get_inpainter(config),
        post_inpaint_transform=data_transform_post_inp,
    )
    logger.info(f"Training Images: {len(dataset_train)}")
    logger.info(f"Validation Images: {len(dataset_val)}")

    if config.use_synthetic_af:
        model = resnet50(num_classes=1000, weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(
            model.fc.in_features, out_features=dataset_val.n_classes
        )

    else:
        if config.af_classifier.weights == "imagenet":
            model = resnet50(num_classes=1000, weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = torch.nn.Linear(
                model.fc.in_features, out_features=dataset_val.n_classes
            )
        else:
            from src.classifier.model import ResNet  # cxr model

            model = ResNet(weights="resnet50-res512-all")
            # only train classification layer
            if not config.af_classifier.finetune_full_model:
                model.af_classification_mode()
            else:
                print("Finetuning full model")

    # model = resnet50(num_classes=dataset_val.n_classes)

    wdb_logger = WandbLogger(
        project="privacy",
        save_dir=config.log_dir,
        config=config.to_dict(),
        tags=["af-clf", config.af_inpainter_name, os.path.basename(config.EXP_PATH)],
        name=os.path.basename(config.EXP_PATH)
        + "_"
        + config.EXP_NAME
        + "_"
        + os.path.basename(config.log_dir),
    )

    callbacks = []
    log_input_img_callback = LogInputImageCallback(
        os.path.join(config.log_dir, "image_callback")
    )
    callbacks.append(log_input_img_callback)

    if config.af_classifier.early_stopping == "val_loss":
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=20,
                verbose=False,
                mode="min",
            )
        )
        early_stopping_checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            filename="{epoch}-{val_loss_step:.7f}",
            dirpath=os.path.join(config.log_dir, "classifier_checkpoints"),
            monitor="val_loss",
            save_last=False,
        )
        callbacks.append(early_stopping_checkpoint_callback)
    else:
        raise ValueError("wrong config.af_classifier.early_stopping in config")

    runner = Trainer(
        logger=wdb_logger,
        callbacks=callbacks,
        max_epochs=config.af_classifier.max_epochs,
        strategy="ddp",
        accelerator="gpu",
        fast_dev_run=config.EXP_NAME.endswith("devrun"),
        check_val_every_n_epoch=config.af_classifier.check_val_every_n_epoch,
        devices=torch.cuda.device_count(),
    )

    pl_dataset = DataModuleFromConfig(
        batch_size=config.af_classifier.batch_size,
        train=dataset_train,
        validation=dataset_val,
        num_workers=config.af_classifier.num_workers,
    )

    pl_model = LightningSAFClassifier(model, config.af_classifier)
    logger.info(f"Logging to: {config.log_dir}")
    runner.fit(pl_model, pl_dataset)

    if pl_model.global_rank == 0:
        best_cur_model = early_stopping_checkpoint_callback.best_model_path
        logger.info(f"Best model path: {best_cur_model} -- Testing now")
        if pl_model.val_acc > 0.95:
            wdb_logger.experiment.tags = wdb_logger.experiment.tags + (">.95acc",)

        dataset_test.set_test_mode(
            True
        )  # return positive and negative for each sample if synthetic af
        test_dataloader = DataLoader(
            dataset_test,
            batch_size=config.af_classifier.batch_size,
            num_workers=4,
            shuffle=False,
            collate_fn=collate_batch,
        )

        dataset_train_noaug = AFClassificationDataset(
            config,
            inputs=train_ds[0],
            labels=train_ds[1],
            pre_inpaint_transform=val_transforms,
            inpainter=get_inpainter(config),
            post_inpaint_transform=data_transform_post_inp,
        )
        dataset_train_noaug.set_test_mode(True)
        train_dataloader_test_mode = DataLoader(
            dataset_train_noaug,
            batch_size=config.af_classifier.batch_size,
            num_workers=4,
            shuffle=False,
            collate_fn=collate_batch,
        )

        test_dl = TestDataModule(test=[test_dataloader, train_dataloader_test_mode])
        runner.test(pl_model, test_dl, best_cur_model)
        test_acc = pl_model.test_acc  # from 0th dataloader

        logger.info(f"Testing model on training data: {best_cur_model}")
        logger.info(f"Accuracy on test data: {pl_model.test_acc}")

        abs_best_path = os.path.join(
            os.path.dirname(config.log_dir), config.af_classifier.best_path
        )
        prev_best = load_prev_best(abs_best_path)

        if prev_best is None or prev_best < test_acc:
            # update best model
            logger.info(f"New best model: {best_cur_model} \nTest Acc: {test_acc}")
            save_copy_checkpoint(
                best_cur_model,
                abs_best_path,
                log_logdir=config.log_dir,
                log_wandb=wdb_logger.experiment.get_url(),
            )

            # update metadata file
            content = f"{test_acc}\n{best_cur_model}"
            with open(abs_best_path + ".txt", "w") as file:
                file.write(content)

        # if hasattr(wdb_logger.experiment, "name"):
        #    name = wdb_logger.experiment.name
        #    sweep_dir = os.path.join(config.log_dir, "sweep")
        #    os.makedirs(sweep_dir, exist_ok=True)
        #    best_path = os.path.join(sweep_dir, f"{name}_best.ckpt")
        #    save_copy_checkpoint(early_stopping_checkpoint_callback.best_model_path, best_path)
        #    dict_to_json(config.to_dict(), os.path.join(sweep_dir, f"{name}_config.json"))
        return test_acc


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--use_synthetic_af", action="store_true")
    parser.add_argument(
        "--af_classifier.best_path",
        default="best.ckpt",
        help="save in exp_name direktory with this name",
    )
    parser.add_argument("--data_csv", default="cxr14privacy.csv")
    parser.add_argument("--af_classifier.learning_rate_annealing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = main_setup(args, name=os.path.basename(__file__).rstrip(".py"))
    main(config)
