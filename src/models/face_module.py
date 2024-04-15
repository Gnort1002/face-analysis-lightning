import os.path as osp
import logging
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy, BinaryAccuracy
from torchmetrics.classification.f_beta import FBetaScore, BinaryFBetaScore
from torchmetrics.classification.auroc import AUROC, BinaryAUROC
from .components.utils import load_pretrained

logger = logging.getLogger(__name__)


class FaceLitModule(LightningModule):
    """`LightningModule` for Face Color classification."""

    def __init__(
        self,
        net: torch.nn.Module,
        pretrained: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_heads: int,
        compile: bool,
        num_classes: Tuple[int, int, int, int, int, int],
        attributes: Tuple[str, str, str, str, str, str],
        binary_threshold: float = 0.5,
        binary_beta: float = 1.0,
    ) -> None:
        """Initialize a `FaceLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        if isinstance(pretrained, str) and osp.exists(pretrained):
            logger.info(f"Loading weights from {pretrained}.")
            pretrained_state_dict = torch.load(pretrained)
            state_dict = self.net.state_dict()
            load_pretrained(state_dict, pretrained_state_dict)
        
        self.attrs = attributes
        self.cur_val = 0

        # loss function
        # metric objects for calculating and averaging accuracy across batches
        for i in range(self.hparams.num_heads):
            if num_classes[i] != 1:
                setattr(
                    self,
                    f"criterion_{self.attrs[i]}",
                    torch.nn.CrossEntropyLoss(),
                )
                setattr(
                    self,
                    f"train_acc_{self.attrs[i]}",
                    Accuracy(task="multiclass", num_classes=num_classes[i]),
                )
                setattr(
                    self,
                    f"val_acc_{self.attrs[i]}",
                    Accuracy(task="multiclass", num_classes=num_classes[i]),
                )
                setattr(
                    self,
                    f"test_acc_{self.attrs[i]}",
                    Accuracy(task="multiclass", num_classes=num_classes[i]),
                )
                setattr(
                    self,
                    f"test_f1_{self.attrs[i]}",
                    FBetaScore(
                        task="multiclass",
                        num_classes=num_classes[i],
                    ),
                )
                setattr(
                    self,
                    f"test_auroc_{self.attrs[i]}",
                    AUROC(task="multiclass", num_classes=num_classes[i]),
                )
            else:
                setattr(self, f"criterion_{self.attrs[i]}", torch.nn.BCELoss())
                setattr(
                    self,
                    f"train_acc_{self.attrs[i]}",
                    BinaryAccuracy(threshold=binary_threshold),
                )
                setattr(
                    self,
                    f"val_acc_{self.attrs[i]}",
                    BinaryAccuracy(threshold=binary_threshold),
                )
                setattr(
                    self,
                    f"test_acc_{self.attrs[i]}",
                    BinaryAccuracy(threshold=binary_threshold),
                )
                setattr(
                    self,
                    f"test_f1_{self.attrs[i]}",
                    BinaryFBetaScore(beta=binary_beta),
                )
                setattr(self, f"test_auroc_{self.attrs[i]}", BinaryAUROC())

            # for averaging loss across batches
            setattr(self, f"train_loss_{self.attrs[i]}", MeanMetric())
            setattr(self, f"val_loss_{self.attrs[i]}", MeanMetric())
            setattr(self, f"test_loss_{self.attrs[i]}", MeanMetric())

        setattr(self, "val_acc", MeanMetric())
        setattr(self, "val_loss", MeanMetric())

        # for tracking best so far validation accuracy
        setattr(self, "val_acc_best", MaxMetric())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for i in range(self.hparams.num_heads):
            getattr(self, f"val_acc_{self.attrs[i]}").reset()
            getattr(self, f"val_loss_{self.attrs[i]}").reset()

        getattr(self, "val_acc").reset()
        getattr(self, "val_loss").reset()
        getattr(self, "val_acc_best").reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        xs, ys = batch
        # breakpoint()
        preds = self.forward(xs)
        losses = {}

        for i in range(self.hparams.num_heads):
            if i in [2, self.hparams.num_heads - 1]:
                preds[self.attrs[i]] = preds[self.attrs[i]].float().squeeze()
                ys[self.attrs[i]] = ys[self.attrs[i]].float()

            loss = getattr(self, f"criterion_{self.attrs[i]}")(preds[self.attrs[i]].squeeze(), ys[self.attrs[i]])
            losses[self.attrs[i]] = loss

        return losses, preds, ys

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses, preds, targets = self.model_step(batch)
        for i in range(self.hparams.num_heads):
            getattr(self, f"train_loss_{self.attrs[i]}")(losses[self.attrs[i]])
            getattr(self, f"train_acc_{self.attrs[i]}")(preds[self.attrs[i]], targets[self.attrs[i]])

            self.log(
                f"train/loss_{self.attrs[i]}",
                getattr(self, f"train_loss_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"train/acc_{self.attrs[i]}",
                getattr(self, f"train_acc_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # return loss or backpropagation will fail
        if self.hparams.num_heads > 1:
            loss_mean = torch.stack(list(losses.values())).mean()
        return loss_mean

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, preds, targets = self.model_step(batch)
        for i in range(self.hparams.num_heads):
            val_loss = getattr(self, f"val_loss_{self.attrs[i]}")(
                losses[self.attrs[i]]
            )
            val_acc = getattr(self, f"val_acc_{self.attrs[i]}")(preds[self.attrs[i]], targets[self.attrs[i]])
            getattr(self, "val_acc")(val_acc)
            getattr(self, "val_loss")(val_loss)
            self.log(
                f"val/loss_{self.attrs[i]}",
                getattr(self, f"val_loss_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"val/acc_{self.attrs[i]}",
                getattr(self, f"val_acc_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.cur_val = getattr(self, "val_acc").compute()
        self.log(
            "val/acc",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/loss",
            getattr(self, "val_loss").compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc_best = self.cur_val
        getattr(self, "val_acc_best")(acc_best)
        self.log(
            "val/acc_best",
            getattr(self, "val_acc_best").compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, preds, targets = self.model_step(batch)

        for i in range(self.hparams.num_heads):
            pred = preds[self.attrs[i]]
            gt = targets[self.attrs[i]]

            getattr(self, f"test_acc_{self.attrs[i]}")(pred, gt)
            getattr(self, f"test_f1_{self.attrs[i]}")(pred, gt)

            self.log(
                f"test/acc_{self.attrs[i]}",
                getattr(self, f"test_acc_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"test/f1_{self.attrs[i]}",
                getattr(self, f"test_f1_{self.attrs[i]}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            if i != 2:
                getattr(self, f"test_auroc_{self.attrs[i]}")(pred, gt.long())
                self.log(
                    f"test/auroc_{self.attrs[i]}",
                    getattr(self, f"test_auroc_{self.attrs[i]}"),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = FaceLitModule(None, None, None, None)
