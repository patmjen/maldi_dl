# MALDI Deep Learning

**A simple pipeline for MALDI spectrum classification with deep learning.**

The goal of this repository is to be a simple entry point for getting started with MALDI spectrum classification. The pipeline is configurable, so you can try different models, losses, data augmentations, etc. It is built on [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) so you can also add your own components.

Datasets are stored in a simple standard format, so you can easily add your own datasets too. See [Adding New Datasets](#adding-new-datasets). Once you've done this, all the remaning parts of the pipeline can be applied directly, without any other modifications.


## Installation
First, clone the repository:

```bash
git clone https://github.com/patmjen/maldi_dl.git
```
Then, install dependencies:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install einops hydra-core pyyaml scikit-learn tensorboard lightning h5py
```
That should be it! The pipeline is now ready to use.


### Exact Environment
To reproduce the exact environment this code was developed on, install the requirements in `frozenenv.txt`.


## Getting Started
The repository includes a small demo dataset. It is a *small* subset of the [DRIAMS dataset](https://doi.org/10.5061/dryad.bzkh1899q). The goal is to predict Penicilin resistance of a bacterial strain based on its MALDI-TOF mass spectrum. The data is stored in the `maldidata` folder in the standard format expected by the pipeline.

The default configuration is to use the demo data. To run, initialize the environment and simply call:

```bash
python train.py
```
When training is finished, summary results will be printed to the terminal like this:

```
                  Train accuracy: 0.8051947951316833
         Train balanced_accuracy: 0.7845767951933734
               Train f1_weighted: 0.8093783329416656
          Train jaccard_weighted: 0.6870342771982116
          Train confusion_matrix: [[160  57]
                                   [ 93 460]]
                    Val accuracy: 0.8571428656578064
           Val balanced_accuracy: 0.8379562043795621
                 Val f1_weighted: 0.8608255332208071
            Val jaccard_weighted: 0.7622624834290765
            Val confusion_matrix: [[ 36   9]
                                   [ 17 120]]
                   Test accuracy: 0.7827869057655334
          Test balanced_accuracy: 0.749031007751938
                Test f1_weighted: 0.7848295339661447
           Test jaccard_weighted: 0.6545394168055146
           Test confusion_matrix: [[ 48  24]
                                   [ 29 143]]
         Train accuracy (probes): 0.7994228005409241
Train balanced_accuracy (probes): 0.7792895279699066
      Train f1_weighted (probes): 0.8033405412243118
 Train jaccard_weighted (probes): 0.6782950094638406
 Train confusion_matrix (probes): [[147  54]
                                   [ 85 407]]
           Val accuracy (probes): 0.8571428656578064
  Val balanced_accuracy (probes): 0.8353488372093023
        Val f1_weighted (probes): 0.8600448019549942
   Val jaccard_weighted (probes): 0.7608264098228071
   Val confusion_matrix (probes): [[ 34   9]
                                   [ 15 110]]
          Test accuracy (probes): 0.784140944480896
 Test balanced_accuracy (probes): 0.7514458955223882
       Test f1_weighted (probes): 0.786310158921321
  Test jaccard_weighted (probes): 0.6563768502563141
  Test confusion_matrix (probes): [[ 45  22]
                                   [ 27 133]]
```
Solid results even if they are not ground breaking. To improve, you may try to adjust the hyperparameters in the configuration files.

## Configuration
The pipeline is configured with [hydra](https://hydra.cc/). The entry point is `config/config.yaml`:
```yaml
defaults:
  - data: driams_demo
  - model: vit
  - loss: cross_entropy
  - optimizer: adamw
  - scheduler: linear_warmup

  # Turn off output files for hydra
  - _self_
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

results: results/  # Folder for checkpoints, tensorboard logs, etc.
train_batch_size: 128
accumulate_grad_batches: 2
epochs: 100
checkpoint_warmup_epochs: 0
checkpoint_moniter_metric: val_bal_acc
checkpoint_moniter_mode: max
val_batch_size: ${train_batch_size}
test_batch_size: ${val_batch_size}
devices: 1
_print_cfg: true
_print_model: true
_print_data: true

# Turn off some hydra output
hydra:
  output_subdir: null
  run:
    dir: .
```
The most important options are the `defaults`, which control the dataset, model, loss function, optimizer and learning rate schedule. For each option, there is a corresponding folder in `config` with a yaml file for each implemented option. E.g., each model has a yaml file in `config/model` that also stores the specific hyperparameters for the model itself.

## Adding New Datasets
To add a new dataset to the pipeline, you must do two things:

1. Store the dataset in the expected format. You must create a folder with three HDF5 files: `train.h5`, `val.h5` and `test.h5`. These contain the training, validation and test data in the format specified [below](#hdf5-data-files).

2. Add a yaml configuration file to `config/data`. Here is configuration file for the demo data as an example:
```yaml
name: DRIAMS_demo_case_split          # Name of dataset
short_name: 'driams'                  # Short name - used for tensorboard logging
path: maldidata/                      # Path to folder with data
classes: ['Susceptible', 'Resistant'] # Class names
compute_probe_stats: true             # Compute probe-wise metrics for summary
spectrum_len: 1600                    # Length of each spectrum
filter_train: true                    # Remove training spectra with poor SNR
filter_val: true                      # Remove validation spectra with poor SNR
filter_test: ${.filter_val}           # Remove test spectra with poor SNR
sampler: class_even                   # Dataloader sampler (class_even or random)
train_sample_augmentations:           # List of training data augmentations
  - tic_normalize:                    #    (see augmentations.py for more)
  - pad_to_divisible:
      d: 256
      value: 0.0
  - scaled_gaussian_noise:
      std_factor: 0.05
  - random_scaling:
      scale: 0.5
train_batch_augmentations:           # List of training batch-wise data augmentations
  - mix_up:                          #    (see augmentations.py for more)
      alpha: 0.1
val_test_sample_augmentations:       # List of test data augmentations
  - tic_normalize:                   #    (see augmentations.py for more)
  - pad_to_divisible:
      d: 256
      value: 0.0
val_test_batch_augmentations: []     # List of test batch-wise data augmentations
```
Once this is done, select your new dataset by changing the `data:` entry in `config.yaml` and start training!

### HDF5 Data Files
Data must be stored in an HDF5 file with the following entries:
* `class_names`: Array of class names with format `['0:Name0', '1:Name1', ..., 'N:NameN']`.
* `class_info`: Array class wise info. May be empty of same as `class_names`.
* `cluster`: (N,) array with probe/cluster index for each of the N spectra. Probes/clusters refers to spectra groups, e.g., a patient index.
* `mz`: (M,) array with m/z values.
* `spot_coords`: (N, 2) array with XY-coordinate each of the N spectra were captured at.
* `spot_ids`: (N,) array with unique ID for each of the N spectra.
* `spot_labels_groundtruth`: (N,) array with gorund truth label for each of the N spectra.
* `spots`: (N,M) array with the N spectra as the rows.

Some of these entries are unused in the default pipeline but may be used by later methods or data augmentations. See the files in `maldidata` for example data files.

### Cross Validation Splits
If you have several cross validation splits for your datasets which share the same metadata, it can be beneficial to create a common "base" configuration file which is then included in the configuration files for each split. This ensures metadata and augmentations stays the same for each split.

**Example** `config/data/_amyl_probe_split_base.yaml` as the base of `config/data/amyl_probe_split_I.yaml`, `config/data/amyl_probe_split_II.yaml` and `config/data/amyl_probe_split_III.yaml`