import torch
import torch.nn as nn
import numpy as np 
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..metric import accuracy

class Runner(LightningModule):
    def __init__(self, model, config, eval_type):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.eval_type = eval_type

    def forward(self, x):
        return self.model(x)

    def test_forward(self, x):
        x = x.squeeze(0)
        return self.model(x)

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay= self.config.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=opt,
            T_0= self.config.T_0
        )
        lr_scheduler = {
            'scheduler': scheduler, 
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': False, # For ReduceLROnPlateau scheduler
            'monitor': 'val_loss' # Metric to monitor
        }
        return [opt], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        audio, label, _ = batch
        prediction = self.forward(audio)
        loss = self.criterion(prediction, label.long())
        acc = accuracy(prediction, label.long())
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,    
            },
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    
    def validation_step(self, batch, batch_idx):
        audio, label, _ = batch
        prediction = self.forward(audio)
        loss = self.criterion(prediction, label.long())
        acc = accuracy(prediction, label.long())
        return {"val_loss": loss, "val_acc": acc}

    def validation_step_end(self, batch_parts):
        # sigle gpu case
        return batch_parts

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_acc = torch.mean(torch.stack([output["val_acc"] for output in outputs]))
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {
                "val_loss": val_loss,
                "val_acc": val_acc,
                }

    def test_step(self, batch, batch_idx):
        audio, label, fname = batch
        prediction = self.test_forward(audio)
        loss = self.criterion(prediction.mean(0,True), label.long())
        acc = accuracy(prediction, label.long())
        return {"val_loss": loss, "val_acc": acc, "prediction": prediction, "label":label, "fname":fname}

    def test_step_end(self, batch_parts):
        # sigle gpu case
        return batch_parts

    def test_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_acc = torch.mean(torch.stack([output["val_acc"] for output in outputs]))
        if self.eval_type == "last":
            self.log_dict(
                {
                    "last_loss": val_loss,
                    "last_acc": val_acc,
                },
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        elif self.eval_type == "best":
            self.log_dict(
                {
                    "best_loss": val_loss,
                    "best_acc": val_acc,
                },
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        result = {"test_loss": float(val_loss.detach().cpu()), "test_acc": float(val_acc.detach().cpu())}
        self.test_results = result
        return result
        