from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import FBetaScore
from torchmetrics.classification.auroc import AUROC
import os
from os.path import join, exists

class MedicalSegmentationModel(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        num_classes: int,
        init_lr: float,
        weight_decay: float,
        use_scheduler: bool,
        num_epochs: int,
        
    ):
        super().__init__()
 
        # Save the arguments as hyperparameters.
        self.save_hyperparameters()
 
        # Loading model using the function defined above.
        self.model = get_model(model_name=self.hparams.model_name, num_classes=self.hparams.num_classes)
 
        # Initializing the required metric objects.
        self.mean_train_loss = MeanMetric()
        self.mean_train_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average="macro")
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average="macro")
 
    def forward(self, data):
        outputs = self.model(pixel_values=data, return_dict=True)
        upsampled_logits = F.interpolate(outputs["logits"], size=data.shape[-2:], mode="bilinear", align_corners=False)
        return upsampled_logits
     
    def training_step(self, batch, *args, **kwargs):
        data, target = batch
        logits = self(data)
 
        # Calculate Combo loss (Segmentation specific loss (Dice) + cross entropy)
        loss = dice_coef_loss(logits, target, num_classes=self.hparams.num_classes)
         
        self.mean_train_loss(loss, weight=data.shape[0])
        self.mean_train_f1(logits.detach(), target)
 
        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True, logger=False)
        self.log("train/batch_f1", self.mean_train_f1, prog_bar=True, logger=False)
        return loss
 
    def on_train_epoch_end(self):
        # Computing and logging the training mean loss & mean f1.
        self.log("train/loss", self.mean_train_loss, prog_bar=True)
        self.log("train/f1", self.mean_train_f1, prog_bar=True)
        self.log("epoch", self.current_epoch)
 
    def validation_step(self, batch, *args, **kwargs):
        data, target = batch
        logits = self(data)
         
        # Calculate Combo loss (Segmentation specific loss (Dice) + cross entropy)
        loss = dice_coef_loss(logits, target, num_classes=self.hparams.num_classes)
 
        self.mean_valid_loss.update(loss, weight=data.shape[0])
        self.mean_valid_f1.update(logits, target)
 
    def on_validation_epoch_end(self):
         
        # Computing and logging the validation mean loss & mean f1.
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True)
        self.log("valid/f1", self.mean_valid_f1, prog_bar=True)
        self.log("epoch", self.current_epoch)
 
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.init_lr,
            weight_decay=self.hparams.weight_decay,
        )
 
        LR = self.hparams.init_lr
        WD = self.hparams.weight_decay
 
        if self.hparams.optimizer_name in ("AdamW", "Adam"):
            optimizer = getattr(torch.optim, self.hparams.optimizer_name)(model.parameters(), lr=LR, 
                                                                          weight_decay=WD, amsgrad=True)
        else:
            optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WD)
 
        if self.hparams.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.trainer.max_epochs // 2,], gamma=0.1)
 
            # The lr_scheduler_config is a dictionary that contains the scheduler
            # and its associated configuration.
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch", "name": "multi_step_lr"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
 
        else:
            return optimizer