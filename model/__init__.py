from lightning_fabric.utilities.seed import seed_everything
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    roc_auc_score,
)
from typing import MutableSequence
from arrgh import arrgh # DEBUG

import augmentation
from .vit import SimpleViT
from .cnn import Cnn, ResNet
from .mlp import MLP
from .linear import LinearModel
from .ensemble import Ensemble
from loss import FocalLoss


def classname_to_class(classname):
    if classname == 'vit':
        return SimpleViT
    elif classname == 'cnn':
        return Cnn
    elif classname == 'resnet':
        return ResNet
    elif classname == 'mlp':
        return MLP
    elif classname == 'linear':
        return LinearModel
    elif classname == 'ensemble':
        return Ensemble
    else:
        raise ValueError(f'Unknown model: {classname}')


class SpectrumModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)

        if 'seed' in hparams.model and hparams.model.seed is not None:
            seed_everything(hparams.model.seed)

        Model = classname_to_class(hparams.model.classname)
        self.model = Model(hparams)

        if hparams.loss.name == 'cross_entropy':
            if hparams.loss.weight == 'none':
                weight = None
            elif isinstance(hparams.loss.weight, MutableSequence):
                weight = torch.tensor(hparams.loss.weight).float()
            else:
                raise ValueError(
                    f'Uknown weight for cross_entropy: {hparams.loss.weight}'
                )
            self.loss = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=hparams.loss.label_smoothing,
            )
        elif hparams.loss.name == 'focal':
            self.loss = FocalLoss(
                hparams.loss.gamma,
                hparams.loss.alpha,
            )
        else:
            raise ValueError(f'Uknown loss: {hparams.loss.name}')

        train_augmentations = hparams.data.train_batch_augmentations
        if train_augmentations:
            self.train_batch_augmentations = augmentation.Compose([
                augmentation.from_config(a) for a in train_augmentations
            ])
        else:
            self.train_batch_augmentations = augmentation.NoOp()

        val_test_augmentations = hparams.data.val_test_batch_augmentations
        if train_augmentations:
            self.val_test_batch_augmentations = augmentation.Compose([
                augmentation.from_config(a) for a in val_test_augmentations
            ])
        else:
            self.val_test_batch_augmentations = augmentation.NoOp()

        self.all_validation_preds = []
        self.all_validation_labels = []


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        spectra, labels, _ = self.train_batch_augmentations(*batch)
        preds = self.forward(spectra)
        loss = self.loss(preds, labels)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return { 'loss': loss, 'preds': preds }


    def validation_step(self, batch, batch_idx):
        spectra, labels, _ = self.val_test_batch_augmentations(*batch)
        preds = self.forward(spectra)

        self.all_validation_labels.append(labels)
        self.all_validation_preds.append(preds)


    def on_validation_epoch_end(self):
        labels = torch.cat(self.all_validation_labels, dim=0)
        preds = torch.cat(self.all_validation_preds, dim=0)
        preds_logits = preds
        preds = preds.argmax(dim=1)
        if labels.ndim > 1 and labels.shape[1] > 1:
            labels = labels.argmax(dim=1)

        # Compute GPU and/or PyTorch metrics
        loss = self.loss(preds_logits, labels)
        acc = (preds == labels).float().mean()

        # Convert to NumPy for remaning metrics
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        balanced_acc = balanced_accuracy_score(labels, preds)
        f1_w = f1_score(labels, preds, average='weighted')
        jaccard_w = jaccard_score(labels, preds, average='weighted')

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_bal_acc', balanced_acc, prog_bar=True, logger=True)
        self.log('val_f1_w', f1_w, prog_bar=True, logger=True)
        self.log('val_jaccard_w', jaccard_w, prog_bar=True, logger=True)

        self.all_validation_labels.clear()
        self.all_validation_preds.clear()


    def predict_step(self, batch, batch_idx):
        spectra, labels, probes = self.val_test_batch_augmentations(*batch)
        return self(spectra), labels, probes


    def configure_optimizers(self):
        if self.hparams.optimizer.name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.optimizer.lr,
                weight_decay=self.hparams.optimizer.weight_decay,
            )
        elif self.hparams.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.optimizer.lr,
                weight_decay=self.hparams.optimizer.weight_decay,
            )
        else:
            raise ValueError(
                f'Unknown optimizer: {self.hparams.optimizer.name}'
            )


        if self.hparams.scheduler.name == 'linear_warmup':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.hparams.scheduler.start_factor,
                total_iters=self.hparams.scheduler.warmup_iters
            )
        elif self.hparams.scheduler.name == 'linear_warmup_cosine_decay':
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=self.hparams.scheduler.start_factor,
                        total_iters=self.hparams.scheduler.warmup_iters
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        self.trainer.max_epochs - self.hparams.scheduler.warmup_iters
                    ),
                ],
                milestones=[self.hparams.scheduler.warmup_iters],
            )
        elif self.hparams.scheduler.name == 'none':
            scheduler = None
        else:
            raise ValueError(f'Uknown scheduler: {self.hparams.scheduler.name}')

        if scheduler is not None:
            return { 'optimizer': optimizer, 'lr_scheduler': scheduler }
        else:
            return optimizer