from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torchvision.models as models


class LightningSAFClassifier(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()
        self.val_acc = 0

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["image"]
        y = batch["target"]

        x_hat = self.model(x)
        loss = self.criterion(x_hat.squeeze(dim=1), y.to(float))

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss_step", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log("train_epoch_loss", torch.stack([x["loss"] for x in outputs]).mean().float())

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target"]
        y_hat = self.model(x)
        loss = self.criterion(y_hat.squeeze(dim=1), y.to(float))
        preds = (torch.sigmoid(y_hat).squeeze(dim=1) > 0.5) == y.squeeze()

        acc = sum(preds)/len(preds)
        self.log("val_acc_step", acc)
        self.log("val_loss_step", loss.item())
        return preds

    def validation_epoch_end(self, outputs):
        acc = torch.cat(outputs).to(torch.float).mean()
        self.log("val_acc", acc, sync_dist=True)
        self.val_acc = acc
        return {"val_acc": acc}

    def predict(self, batch):
        # (B x) C x H x W -> B: bool
        x = batch["image"]
        if x.ndim != 4:
            x = x.unsqueeze(dim=0)
        y_hat = self.model(x)
        pred = torch.sigmoid(y_hat).squeeze(dim=1) > 0.5
        return pred, y_hat

    def test_step(self, batch, batch_idx):
        pred, _ = self.predict(batch)
        y = batch["target"]
        pred_corr = pred == y.squeeze()
        return pred_corr

    def test_epoch_end(self, outputs):
        outputs = torch.cat(outputs)
        acc = outputs.float().mean()
        self.test_acc = acc
        self.log("test_acc", acc, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr)
        ret_dict = {"optimizer": optimizer, "monitor":"train_loss_step"}
        if self.config.learning_rate_annealing: 
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            ret_dict["scheduler"] = scheduler
        return ret_dict


# define the LightningModule
class LightningIDClassifier(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion =  nn.BCEWithLogitsLoss()
        self.save_hyperparameters()
        self.val_acc = 0

    def predict(self, batch):
        # (B x) C x H x W -> B: bool
        x = batch["image"]
        if x.ndim != 4:
            x = x.unsqueeze(dim=0)
        y_hat = self.model(x)
        pred = torch.sigmoid(y_hat).squeeze(dim=1) > 0.5
        return pred, y_hat

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["image"]
        y = batch["target"]
        x_hat = self.model(x)
        loss = self.criterion(x_hat.squeeze(dim=1), y.to(float))
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log("train_epoch_loss", torch.stack([x["loss"] for x in outputs]).mean().float(), sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target"]
        y_hat = self.model(x)
        loss = self.criterion(y_hat.squeeze(dim=1), y.to(float))
        preds = (torch.sigmoid(y_hat).squeeze(dim=1) > 0.5) == y.squeeze()

        acc = sum(preds)/len(preds)
        self.log("val_acc_step", acc)
        self.log("val_loss_step", loss.item())
        return preds

    def validation_epoch_end(self, outputs):
        acc = torch.cat(outputs).to(torch.float).mean()
        self.log("val_acc_epoch", acc, sync_dist=True)
        self.val_acc = acc
        return {"val_acc_epoch": acc}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr)
        ret_dict = {"optimizer": optimizer, "monitor":"train_epoch_loss"}
        if self.config.learning_rate_annealing: 
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            ret_dict["scheduler"] = scheduler
        return ret_dict

    def test_step(self, batch, batch_idx):
        pred, _ = self.predict(batch)
        y = batch["target"]
        pred_corr = pred == y.squeeze()
        return pred_corr

    def test_epoch_end(self, outputs):
        outputs = torch.cat(outputs)
        acc = outputs.float().mean()
        self.test_acc = acc
        self.log("test_acc", acc, sync_dist=True)