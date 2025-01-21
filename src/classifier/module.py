from pytorch_lightning.utilities.types import STEP_OUTPUT
from log import logger
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torchvision.models as models


class LightningSAFClassifier(pl.LightningModule):
    def __init__(self, model, config):
        super(LightningSAFClassifier, self).__init__()
        self.model = model
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()
        self.val_acc = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = {
            "train": [],
            "test": [],
        }  # test on train and test data

    def on_train_epoch_start(self):
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["image"]
        y = batch["target"]

        y_hat = self.model(x)
        loss = self.criterion(y_hat.squeeze(dim=1), y.to(float))

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss_step", loss.item())
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        self.log(
            "train_epoch_loss", torch.stack(self.training_step_outputs).mean().float()
        )
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target"]
        y_hat = self.model(x)
        loss = self.criterion(y_hat.squeeze(dim=1), y.to(float))
        preds = (torch.sigmoid(y_hat).squeeze(dim=1) > 0.5) == y.squeeze()

        acc = sum(preds) / len(preds)
        self.log("val_acc_step", acc, sync_dist=True)
        self.log("val_loss_step", loss.item(), sync_dist=True)
        self.validation_step_outputs.append({"preds": preds, "loss": loss})
        return loss.item()

    def on_validation_epoch_end(self):
        acc = (
            torch.cat([output["preds"] for output in self.validation_step_outputs])
            .to(torch.float)
            .mean()
        )
        loss = (
            torch.stack([output["loss"] for output in self.validation_step_outputs])
            .to(torch.float)
            .mean()
        )
        self.log("val_acc", acc, sync_dist=True)
        self.log("val_loss", loss, sync_dist=True)
        self.val_acc = acc
        self.val_loss = loss
        self.validation_step_outputs.clear()

    def predict(self, batch):
        # (B x) C x H x W -> B: bool
        x = batch["image"]
        if x.ndim != 4:
            x = x.unsqueeze(dim=0)
        y_hat = self.model(x)
        pred = torch.sigmoid(y_hat).squeeze(dim=1) > 0.5
        return pred, y_hat

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pred, _ = self.predict(batch)
        y = batch["target"]
        pred_corr = pred == y.squeeze()
        key = "train" if dataloader_idx == 1 else "test"
        self.test_step_outputs[key].append(pred_corr)

    def on_test_epoch_end(self, dataloader_idx=0):
        for k in self.test_step_outputs.keys():
            outputs = torch.cat(self.test_step_outputs[k])
            acc = outputs.float().mean()
            if k == "test":
                self.test_acc = acc
            self.log(f"test_acc_{k}", acc, sync_dist=True)
        # Don't self.test_step_outputs.clear() - used by callbacks

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr)
        ret_dict = {"optimizer": optimizer, "monitor": "train_loss_step"}
        if self.config.learning_rate_annealing:
            patience = (
                10
                if not hasattr(self.config, "learning_rate_annealing_patience")
                else self.config.learning_rate_annealing_patience
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=patience, verbose=True
            )
            ret_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": self.config.check_val_every_n_epoch,
            }
        return ret_dict
