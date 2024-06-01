import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from collections.abc import MutableMapping
import datetime
import json
from os.path import join
from time import perf_counter as time

import hydra
import h5py
from lightning_fabric.utilities.seed import seed_everything
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

import augmentation
from data import SpectrumDataset
from eval import eval_and_print
from model import SpectrumModel


class ModelCheckpointWithWarmup(pl.callbacks.ModelCheckpoint):
    def __init__(self, warmup_epochs=0, **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs


    def _should_skip_saving_checkpoint(self, trainer):
        if trainer.current_epoch < self.warmup_epochs:
            return True
        else:
            return super()._should_skip_saving_checkpoint(trainer)


def prep_for_json(d):
    out = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = prep_for_json(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out


def remove_nonscalar_values(d):
    out = dict()
    for k, v in d.items():
        # https://stackoverflow.com/a/16807050
        if not hasattr(v, '__len__'):
            out[k] = v
    return out


def flatten_dict(dictionary, parent_key='', separator='_'):
    # https://stackoverflow.com/a/6027615
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    torch.set_float32_matmul_precision('medium')
    if cfg._print_cfg:
        print(OmegaConf.to_yaml(cfg, resolve=True))

    # Make result folder
    time_now = datetime.datetime.now()
    save_path = join(
        '/results/',
        cfg.data.short_name,
        cfg.model.name,
        datetime.date.today().isoformat(),
        time_now.strftime('%H_%M_%S'),
    )

    checkpoints_dir = join(save_path, 'model_checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Load data
    # TODO: Maybe refactor this to another function
    print('Load training data')
    t0 = time()
    with h5py.File(join(cfg.data.path, 'train.h5'), 'r') as f:
        train_dataset = SpectrumDataset(
            f,
            transforms=augmentation.from_config_list(
                cfg.data.train_sample_augmentations,
            ),
            one_hot=True,
            filter_spectra=cfg.data.filter_train
        )
    if cfg.data.sampler == 'class_even':
        labels = train_dataset.labels
        if labels.ndim > 1 and labels.shape[1] > 1:
            # Labels are one-hot - reverse to index
            labels = labels.argmax(dim=1)
        class_counts = labels.unique(return_counts=True)[-1]
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(train_dataset))
    elif cfg.data.sampler == 'random':
        sampler = None
    else:
        raise ValueError(f'Invalid sampler: {cfg.data.sampler}')
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        sampler=sampler,
    )
    print(f'  {time() - t0} sec.')

    print('Load validation data')
    t0 = time()
    with h5py.File(join(cfg.data.path, 'val.h5'), 'r') as f:
        val_dataset = SpectrumDataset(
            f,
            transforms=augmentation.from_config_list(
                cfg.data.val_test_sample_augmentations,
            ),
            one_hot=True,
            filter_spectra=cfg.data.filter_val
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )
    print(f'  {time() - t0} sec.')


    # Build model and setup trainer
    print('Create logger')
    logger = TensorBoardLogger(
        save_dir="/results/tensorboard",
        name=join(
            cfg.data.short_name,
            cfg.model.name,
            datetime.date.today().isoformat(),
            time_now.strftime('%H_%M_%S')
        ),
        version=0,
    )

    print('Create callbacks')
    if cfg.checkpoint_warmup_epochs >= cfg.epochs:
        import warnings
        warnings.warn(
            f'Epochs ({cfg.epochs}) less than checkpoint warmup epochs ({cfg.checkpoint_warmup_epochs}) - setting warmpup epochs to {cfg.epochs - 1}'
        )
        cfg.checkpoint_warmup_epochs = cfg.epochs - 1
    model_checkpoint = ModelCheckpointWithWarmup(
        warmup_epochs=cfg.checkpoint_warmup_epochs,
        dirpath=checkpoints_dir,
        monitor=cfg.checkpoint_moniter_metric,
        save_top_k=1,
        mode=cfg.checkpoint_moniter_mode
    )
    last_model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        save_on_train_epoch_end=True,
        filename='latest',
        every_n_epochs=1,
    )
    lr_logging = pl.callbacks.LearningRateMonitor(logging_interval='step')

    print('Create model')
    pl_model = SpectrumModel(cfg)
    trainer = pl.Trainer(
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        devices=cfg.devices,
        accelerator='gpu',
        max_epochs=cfg.epochs,
        callbacks=[
            lr_logging,
            model_checkpoint,
            last_model_checkpoint,
        ],
        logger=logger,
    )

    # Train model
    print('Train model')
    trainer.fit(pl_model, train_loader, val_loader)

    # Test model
    print('Re-Load training data')
    t0 = time()
    with h5py.File(join(cfg.data.path, 'train.h5'), 'r') as f:
        train_dataset = SpectrumDataset(
            f,
            transforms=augmentation.from_config_list(
                cfg.data.val_test_sample_augmentations,
            ),
            one_hot=True,
            filter_spectra=cfg.data.filter_train
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        pin_memory=True,
        num_workers=4
    )
    print(f'  {time() - t0} sec.')

    # New validation dataloader, since the old will crash if we interupted
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        pin_memory=True,
        num_workers=4
    )

    print('Load test data')
    t0 = time()
    with h5py.File(join(cfg.data.path, 'test.h5'), 'r') as f:
        test_dataset = SpectrumDataset(
            f,
            transforms=augmentation.from_config_list(
                cfg.data.val_test_sample_augmentations,
            ),
            one_hot=True,
            filter_spectra=cfg.data.filter_test
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        pin_memory=True,
        num_workers=4
    )
    print(f'  {time() - t0} sec.')

    results = eval_and_print(cfg, train_loader, val_loader, test_loader,
                             trainer, ckpt_path='best')
    with open(join(save_path, 'results.json'), 'w') as f:
        json.dump(prep_for_json(results), f, indent=2)

    tb_results = remove_nonscalar_values(flatten_dict(results))
    logger.log_hyperparams({}, metrics=tb_results)


if __name__ == '__main__':
    main()