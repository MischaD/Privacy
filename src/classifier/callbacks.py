import os.path
import wandb
import torch
from torchvision.utils import make_grid
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class CustomEarlyStoppingAndCheckpointCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, dirpath, filename, min_value):
        super().__init__(dirpath=dirpath, filename=filename)
        self.min_value = min_value

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.val_acc >= self.min_value:
            path = self.format_checkpoint_name(metrics={"epoch": pl_module.current_epoch, "val_acc": pl_module.val_acc})
            trainer.save_checkpoint(path)
            trainer.should_stop = True
            trainer.strategy.teardown()


class LogInputImageCallback(pl.callbacks.Callback):
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.path = log_dir
        self.wrong_test_images = {"image":[], "gt_label":[], "pred":[]}

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0 and pl_module.current_epoch in [0,1]:
            x_grid, y_grid = self.log_images(batch["image"], batch["target"], extension="train_images.png")
            pl_module.logger.log_image("train_image", [wandb.Image(x_grid, caption="Train Image Input"), wandb.Image(y_grid, caption="Train Image Label. White contains SAF")])

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0:
            x_grid, y_grid = self.log_images(batch["image"], batch["target"], extension=f"val_epoch={pl_module.current_epoch}_images.png")
            # log prediction
            pred, _ = pl_module.predict(batch)
            label_grid = []
            for y_t in pred[:225]:
                label_grid.append(
                    torch.ones_like(batch["image"][0].detach().cpu()) if y_t else torch.zeros_like(batch["image"][0].detach().cpu()).detach().cpu())
            y_grid_pred = make_grid(torch.stack(label_grid).detach().cpu(), nrow=8, pad_value=0.5)
            pl_module.logger.log_image("val_image", [wandb.Image(x_grid, caption="Input"), wandb.Image(y_grid, caption="Groundtruth"), wandb.Image(y_grid_pred, caption="Prediction")])

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        pred, _ = pl_module.predict(batch)

        gt_label_grid = []
        for y_t in batch["target"]:
            gt_label_grid.append(torch.ones_like(batch["image"][0].detach().cpu()) if y_t else torch.zeros_like(batch["image"][0]).detach().cpu())

        label_grid = []
        for y_t in pred:
            label_grid.append(
                torch.ones_like(batch["image"][0].detach().cpu()) if y_t else torch.zeros_like(batch["image"][0].detach().cpu()).detach().cpu())

        for i in range(len(pred)): 
            if pred[i] != batch["target"][i]: 
                self.wrong_test_images["image"].append(batch["image"][i].detach().cpu())
                self.wrong_test_images["gt_label"].append(gt_label_grid[i])
                self.wrong_test_images["pred"].append(label_grid[i])

    def on_test_epoch_end(self, trainer, pl_module):
        # pl_module.logger.log_image("test_image", [wandb.Image(x_grid, caption="Input"), wandb.Image(y_grid, caption="Groundtruth"), wandb.Image(y_grid_pred, caption="Prediction")])
        # randomly plot 32 wrong predictions
        sample_idc = torch.randperm(len(self.wrong_test_images["image"]))[:32]
        images = []
        gt_labels = []
        preds = []
        for idx in sample_idc: 
            images.append(self.wrong_test_images["image"][idx])
            gt_labels.append(self.wrong_test_images["gt_label"][idx])
            preds.append(self.wrong_test_images["pred"][idx])

        if len(images) > 0:
            x_grid = make_grid(images, nrow=8, pad_value=0.5)
            y_grid = make_grid(gt_labels, nrow=8, pad_value=0.5)
            y_grid_pred = make_grid(preds, nrow=8, pad_value=0.5)
            pl_module.logger.log_image("Wrong Test Samples", [wandb.Image(x_grid, caption="Input"), wandb.Image(y_grid, caption="Groundtruth"), wandb.Image(y_grid_pred, caption="Prediction")])

        self.wrong_test_images = {"image":[], "gt_label":[], "pred":[]}

    def log_images(self, x, y, extension):
        x = ((x + 1) / 2).clip(0, 1)
        y = y[:225]

        # training images
        grid = make_grid(x[:225].detach().cpu(), nrow=8, pad_value=0.5)
        plt.imshow(grid.permute(1,2,0))
        plt.axis('off')
        plt.savefig(os.path.join(self.path, extension))

        label_grid = []
        for y_t in y:
            label_grid.append(torch.ones_like(x[0].detach().cpu()) if y_t else torch.zeros_like(x[0]).detach().cpu())
        y_grid = make_grid(torch.stack(label_grid).detach().cpu(), nrow=8, pad_value=0.5)
        plt.imshow(y_grid.permute(1, 2, 0))
        plt.savefig(os.path.join(self.path, "label_" + extension))
        return grid, y_grid


#class LogInputImageIDCallback(LogInputImageCallback):
#    def __init__(self, log_dir):
#        super().__init__(log_dir)
#
#    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
#        if batch_idx == 0 and pl_module.current_epoch == 0:
#            x_grid = self.log_images(batch["image"], extension="train_images.png")
#            pl_module.logger.log_image("train_image", [wandb.Image(x_grid, caption="Input")])
#
#    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
#        if batch_idx == 0:
#            x_grid = self.log_images(batch["image"], extension=f"val_epoch={pl_module.current_epoch}_images.png")
#            # log prediction
#            pred = pl_module.predict(batch)
#            label_grid = []
#            for y_l, y_t in zip(batch["target"], pred[:225]):
#                label_grid.append(
#                    torch.ones_like(batch["image"][0].detach().cpu()) if y_t == y_l else torch.zeros_like(batch["image"][0].detach().cpu()).detach().cpu())
#            y_grid_pred = make_grid(torch.stack(label_grid).detach().cpu(), nrow=8, pad_value=0.5, dpi=200)
#            pl_module.logger.log_image("val_image", [wandb.Image(x_grid, caption="Input"), wandb.Image(y_grid_pred, caption="Prediction (White is correct)")])
#            plt.imshow(y_grid_pred.permute(1, 2, 0))
#            plt.savefig(os.path.join(self.path, f"pred_val_epoch={pl_module.current_epoch}_images.png"))
#
#    def log_images(self, x, extension):
#        x = (x + 1) / 2
#        # training images
#        grid = make_grid(x[:225].detach().cpu(), nrow=8, dpi=200)
#        plt.imshow(grid.permute(1,2,0))
#        plt.axis('off')
#        plt.savefig(os.path.join(self.path, extension))
#        return grid
#
#
#
#
#
#
#
#






