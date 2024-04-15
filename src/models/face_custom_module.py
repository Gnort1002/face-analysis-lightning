from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import FBetaScore
from torchmetrics.classification.auroc import AUROC
import os
from os.path import join, exists


class FaceCustomLitModule(LightningModule):
    """`LightningModule` for Face Color classification."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        age_head: torch.nn.Module,
        gender_head: torch.nn.Module,
        race_head: torch.nn.Module,
        skintone_head: torch.nn.Module,
        emotion_head: torch.nn.Module,
        masked_head: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        attributes,
        checkpoint_dir: str
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
        self.checkpoint_dir = checkpoint_dir
        self.backbone = backbone
        self.age_head = age_head
        self.gender_head = gender_head
        self.race_head = race_head
        self.skintone_head = skintone_head
        self.emotion_head = emotion_head
        self.masked_head = masked_head
        self.attrs_arr = attributes
        self.criterion = criterion
        self.attrs = {}
        for attr in self.attrs_arr:
            self.attrs[attr[0]] = attr[1]
        for attr in self.attrs.keys():
            if not exists(join(self.checkpoint_dir, attr)):
                os.makedirs(join(self.checkpoint_dir, attr))
        if not exists(join(self.checkpoint_dir, "backbone")):
            os.makedirs(join(self.checkpoint_dir, "backbone"))
        self.cur_val = -1
        # self.val_acc_best = -1
        # loss function
        # metric objects for calculating and averaging accuracy across batches
        for attr in self.attrs.keys():
            if attr not in ["age", "gender", "masked"]:
                setattr(self,
                        f"criterion_{attr}",
                        self.criterion)
                setattr(self,
                        f"train_acc_{attr}",
                        Accuracy(task="multiclass",
                                 num_classes=int(self.attrs[attr])
                                 )
                        )
                setattr(self,
                        f"val_acc_{attr}",
                        Accuracy(task="multiclass",
                                 num_classes=int(self.attrs[attr])
                                 )
                        )
                setattr(self,
                        f"test_acc_{attr}",
                        Accuracy(task="multiclass",
                                 num_classes=int(self.attrs[attr])
                                 )
                        )
                setattr(self,
                        f"test_f1_{attr}",
                        FBetaScore(task="multiclass",
                                   num_classes=int(self.attrs[attr]),
                                   )
                        )
                setattr(self,
                        f"test_auroc_{attr}",
                        AUROC(task="multiclass",
                              num_classes=int(self.attrs[attr]))
                        )
            else:
                setattr(self,
                        f"criterion_{attr}",
                        torch.nn.BCELoss())
                if attr == "age":
                    setattr(self,
                            f"train_acc_{attr}",
                            Accuracy(task="multilabel",
                                     num_labels=int(self.attrs[attr])
                                     )
                            )
                    setattr(self,
                            f"val_acc_{attr}",
                            Accuracy(task="multilabel",
                                     num_labels=int(self.attrs[attr])
                                     )
                            )
                    setattr(self,
                            f"test_acc_{attr}",
                            Accuracy(task="multilabel",
                                     num_labels=int(self.attrs[attr])
                                     )
                            )
                    setattr(self,
                            f"test_f1_{attr}",
                            FBetaScore(task="multilabel",
                                       num_labels=int(self.attrs[attr]),
                                       )
                            )
                    setattr(self,
                            f"test_auroc_{attr}",
                            AUROC(task="multilabel",
                                  num_labels=int(self.attrs[attr])
                                  )
                            )
                else:
                    setattr(self,
                            f"train_acc_{attr}",
                            Accuracy(task="binary",
                                     num_classes=int(self.attrs[attr])
                                     )
                            )
                    setattr(self,
                            f"val_acc_{attr}",
                            Accuracy(task="binary",
                                     num_classes=int(self.attrs[attr])
                                     )
                            )
                    setattr(self,
                            f"test_acc_{attr}",
                            Accuracy(task="binary",
                                     num_classes=int(self.attrs[attr])
                                     )
                            )
                    setattr(self,
                            f"test_f1_{attr}",
                            FBetaScore(task="binary",
                                       num_classes=int(self.attrs[attr]),
                                       )
                            )
                    setattr(self,
                            f"test_auroc_{attr}",
                            AUROC(task="binary",
                                  num_classes=int(self.attrs[attr])
                                  )
                            )

            # for averaging loss across batches
            setattr(self, f"train_loss_{attr}", MeanMetric())
            setattr(self, f"val_loss_{attr}", MeanMetric())
            setattr(self, f"test_loss_{attr}", MeanMetric())
        self.val_acc = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        pred = {}
        with torch.no_grad():
            representation = self.backbone(x)
        for attr in self.attrs.keys():
            pred[attr] = getattr(self, f"{attr}_head")(representation)
        return pred

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for attr in self.attrs.keys():
            getattr(self, f"val_acc_{attr}").reset()
            getattr(self, f"val_loss_{attr}").reset()

        self.val_acc.reset()
        self.val_loss.reset()
        self.val_acc_best.reset()

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
        x, y = batch
        pred = self.forward(x)
        keys = list(pred.keys())
        if "age" in keys:
            pred['age'] = pred['age'].float()
            y['age'] = y['age'].float()
        if "gender" in keys:
            pred['gender'] = pred['gender'].float()
            y['gender'] = y['gender'].float()
        if "masked" in keys:
            pred['masked'] = pred['masked'].float()
            y['masked'] = y['masked'].float()
        losses = {}
        for attr in self.attrs.keys():
            loss = getattr(self,
                           f"criterion_{attr}")(pred[attr],
                                                y[attr])
            # getattr(self,
            #         f"train_loss_{attr}")(loss)
            # getattr(self,
            #         f"train_acc_{attr}")(pred[attr],
            #                              y[attr])
            losses[attr] = loss
        for attr in self.attrs.keys():
            if attr not in ["age", "gender", "masked"]:
                pred[attr] = torch.argmax(pred[attr], dim=1)
        return losses, pred, y

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
        for attr in self.attrs.keys():
            getattr(self, f"train_loss_{attr}")(losses[attr])
            getattr(self, f"train_acc_{attr}")(preds[attr],
                                               targets[attr])
            self.log(
                f"train/loss_{attr}",
                getattr(self, f"train_loss_{attr}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                )
            self.log(
                f"train/acc_{attr}",
                getattr(self, f"train_acc_{attr}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                )
        # return loss or backpropagation will fail
        if len(self.attrs.keys()) > 1:
            losses = torch.stack(list(losses.values())).mean()
        else:
            losses = losses[list(losses.keys())[0]]
        return losses

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
        val_acc_attr = 0
        val_loss_attr = 0
        for attr in self.attrs.keys():
            val_loss_attr = losses[attr]
            val_acc_attr = getattr(
                self,
                f"val_acc_{attr}")(preds[attr],
                                   targets[attr])
            self.val_acc.update(val_acc_attr)
            self.val_loss.update(val_loss_attr)
            # self.log(
            #     f"val/loss_{attr}",
            #     val_loss,
            #     on_step=False,
            #     on_epoch=True,
            #     prog_bar=True,
            # )
            self.log(
                f"val/acc_{attr}",
                val_acc_attr,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.log(
            "val/acc",
            self.val_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/loss",
            self.val_loss.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.cur_val = self.val_acc.compute()

        if self.val_acc >= self.val_acc_best.compute():
            for attr in self.attrs.keys():
                torch.save(
                    getattr(
                        self,
                        attr + "_head").state_dict(),
                    join(self.checkpoint_dir, attr, "best.ckpt")
                    )
            torch.save(
                self.backbone.state_dict(),
                join(self.checkpoint_dir, "backbone", "best.ckpt")
                )
        self.val_acc_best(self.cur_val)
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )
        for attr in self.attrs.keys():
            torch.save(
                getattr(
                    self,
                    attr + "_head").state_dict(),
                join(self.checkpoint_dir, attr, "last.ckpt")
                )
        torch.save(
            self.backbone.state_dict(),
            join(self.checkpoint_dir, "backbone", "last.ckpt")
            )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        for attr in self.attrs.keys():
            # breakpoint()
            getattr(self,
                    f"test_acc_{attr}")(preds[attr],
                                        targets[attr])
            getattr(self,
                    f"test_f1_{attr}")(preds[attr],
                                       targets[attr])
            # getattr(self,
            #         f"test_auroc_{attr}")(preds[attr],
            #                               targets[attr].long())
            self.log(
                f"test/acc_{attr}",
                getattr(self, f"test_acc_{attr}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"test/f1_{attr}",
                getattr(self, f"test_f1_{attr}"),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            # self.log(
            #     f"test/auroc_{attr}",
            #     getattr(self, f"test_auroc_{attr}"),
            #     on_step=False,
            #     on_epoch=True,
            #     prog_bar=True,
            # )

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
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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
    _ = FaceCustomLitModule(None, None, None, None)
