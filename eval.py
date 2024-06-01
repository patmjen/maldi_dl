import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES',
                                                    default='2')

import argparse
from glob import glob
from os.path import join
from pathlib import Path
from time import time

import h5py
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader

import augmentation
from data import SpectrumDataset
from model import SpectrumModel


def compute_metrics(preds, labels):
    preds_idx = preds.argmax(dim=1)
    preds_sm = preds.softmax(dim=1)
    return {
        'accuracy': float((preds_idx == labels).float().mean()),
        'balanced_accuracy': balanced_accuracy_score(labels.numpy(), preds_idx.numpy()),
        'f1_weighted': f1_score(labels.numpy(), preds_idx.numpy(), average='weighted'),
        #'roc_auc_weighted_ovo': roc_auc_score(labels.numpy(), preds_sm.numpy(), average='weighted', multi_class='ovo'),
        'jaccard_weighted': jaccard_score(labels.numpy(), preds_idx.numpy(), average='weighted'),
        'confusion_matrix': confusion_matrix(labels.numpy(), preds_idx.numpy()),
    }


def compute_metrics_from_predictions(results, probe_stats=True):
    preds = []
    labels = []
    probes = []
    for pred, label, probe in results:
        preds.append(pred)
        if label.ndim > 1 and label.shape[1] > 1:
            label = label.argmax(dim=1)
        labels.append(label)
        probes.append(probe)
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    probes = torch.cat(probes)

    spectra_metrics = compute_metrics(preds, labels)

    if probe_stats:
        probe_preds = []
        probe_labels = []
        for c in probes.unique():
            index = probes == c
            lb = labels[index][0]
            assert (lb == labels[index]).all()
            probe_labels.append(lb)
            #probe_preds.append(preds[index].sum(dim=0, keepdim=True))
            probe_preds.append((preds[index].softmax(dim=1) - 0.5).sum(dim=0, keepdim=True))

        probe_preds = torch.cat(probe_preds)
        probe_labels = torch.stack(probe_labels)

        probe_metrics = compute_metrics(probe_preds, probe_labels)
    else:
        probe_metrics = {}

    return spectra_metrics, probe_metrics


def print_metrics(metrics, prefix='', suffix=''):
    if not metrics:
        return
    max_name_len = max(len(n) for n in metrics.keys())
    for name, value in metrics.items():
        pad = ' ' * (max_name_len - len(name))
        descr = f'{pad}{prefix}{name}{suffix}: '
        print(descr, end='')
        if isinstance(value, (np.ndarray, torch.Tensor)):
            value = np.asarray(value)
            if value.ndim >= 2:
                value = np.array2string(value, max_line_width=200,
                                        prefix=' ' * len(descr))
        print(value)


def eval_and_print(
    cfg,
    train_loader,
    val_loader,
    test_loader,
    trainer,
    model=None,
    ckpt_path=None,
):
    if model is None and ckpt_path is None:
        raise ValueError('Must supply either a model or checkpoint path')

    print('Train')
    train_metrics, train_metrics_probes = compute_metrics_from_predictions(
        trainer.predict(model=model, dataloaders=train_loader,
                        ckpt_path=ckpt_path),
        probe_stats=cfg.data.compute_probe_stats,
    )
    print('Validation')
    val_metrics, val_metrics_probes = compute_metrics_from_predictions(
        trainer.predict(model=model, dataloaders=val_loader,
                        ckpt_path=ckpt_path),
        probe_stats=cfg.data.compute_probe_stats,
    )
    print('Test')
    test_metrics, test_metrics_probes = compute_metrics_from_predictions(
        trainer.predict(model=model, dataloaders=test_loader,
                        ckpt_path=ckpt_path),
        probe_stats=cfg.data.compute_probe_stats,
    )

    probe_suffix = ' (probes)'
    pad = ' ' * len(probe_suffix) if cfg.data.compute_probe_stats else ''
    print_metrics(train_metrics, pad + 'Train ')
    print_metrics(val_metrics, pad + '  Val ')
    print_metrics(test_metrics, pad + ' Test ')
    if cfg.data.compute_probe_stats:
        print_metrics(train_metrics_probes, 'Train ', probe_suffix)
        print_metrics(val_metrics_probes, '  Val ', probe_suffix)
        print_metrics(test_metrics_probes, ' Test ', probe_suffix)

    results = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    if cfg.data.compute_probe_stats:
        results.update({
            "train_probes": train_metrics_probes,
            "val_probes": val_metrics_probes,
            "test_probes": test_metrics_probes,
        })

    return results


def main(args):
    torch.set_float32_matmul_precision('medium')

    # Load config from tensorboard directory
    tensorboard_path = join('/results/tensorboard', args.tensorboard_name)
    cfg = OmegaConf.load(join(tensorboard_path, 'hparams.yaml'))
    if args.print_cfg:
        print(OmegaConf.to_yaml(cfg, resolve=True))

    # Extract checkpoint directory from tensorboard path
    tensorboard_path = Path(tensorboard_path)
    checkpoint_dir = join('/results', *tensorboard_path.parts[3:7],
                          'model_checkpoints')
    checkpoint_path = glob(join(checkpoint_dir, '*.ckpt'))
    if len(checkpoint_path) > 1:
        if args.ckpt == 'best':
            checkpoint_path = checkpoint_path[0]
        else:
            checkpoint_path = checkpoint_path[-1]

    # Load data
    print('Load training data')
    t0 = time()
    with h5py.File(join(cfg.data.path, 'train.h5'), 'r') as f:
        train_dataset = SpectrumDataset(
            f,
            transforms=augmentation.from_config_list(
                cfg.data.val_test_sample_augmentations
            ),
            one_hot=True,
            filter_spectra=cfg.data.filter_train
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.val_batch_size,
        pin_memory=True,
        num_workers=0
    )
    print(f'  {time() - t0} sec.')

    print('Load validation data')
    t0 = time()
    with h5py.File(join(cfg.data.path, 'val.h5'), 'r') as f:
        val_dataset = SpectrumDataset(
            f,
            transforms=augmentation.from_config_list(
                cfg.data.val_test_sample_augmentations
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

    print('Load test data')
    t0 = time()
    with h5py.File(join(cfg.data.path, 'test.h5'), 'r') as f:
        augmentations = cfg.data.val_test_sample_augmentations
        if augmentations:
            augmentations = augmentation.Compose([
                augmentation.from_config(a) for a in augmentations
            ])
        else:
            augmentations = None
        test_dataset = SpectrumDataset(
            f,
            transforms=augmentations,
            one_hot=True,
            filter_spectra=cfg.data.filter_test
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        pin_memory=True,
        num_workers=0
    )
    print(f'  {time() - t0} sec.')

    # Load model and do inference
    model = SpectrumModel.load_from_checkpoint(checkpoint_path)

    trainer = pl.Trainer(
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        devices=cfg.devices,
        accelerator='gpu',
        max_epochs=cfg.epochs,
    )

    eval_and_print(cfg, train_loader, val_loader, test_loader, trainer,
                   model=model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tensorboard_name')
    parser.add_argument('--print_cfg', action='store_true')
    parser.add_argument('--ckpt', choices=['best', 'latest'], default='best')
    args = parser.parse_args()
    main(args)