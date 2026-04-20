# Simulation encoder repository

[![Build Status](https://github.com/bagherilab/simulation-encoder/workflows/build/badge.svg)](https://github.com/bagherilab/simulation-encoder/actions?query=workflow%3Abuild)
[![Codecov](https://codecov.io/gh/bagherilab/simulation-encoder/graph/badge.svg?token=nZSUf47ltR)](https://codecov.io/gh/bagherilab/simulation-encoder)
[![Lint Status](https://github.com/bagherilab/simulation-encoder/workflows/lint/badge.svg)](https://github.com/bagherilab/simulation-encoder/actions?query=workflow%3Alint)
[![Documentation](https://github.com/bagherilab/simulation-encoder/workflows/documentation/badge.svg)](https://bagherilab.github.io/simulation-encoder/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Note
This version of the code is under active development. For a stable version of the code associated with the manuscript "Inductive bias influences the spatial scale of biological features learned from images", please use the version on [Zenodo](https://doi.org/10.5281/zenodo.17073719).
The datasets used in the paper can also be found on [Zenodo](https://doi.org/10.5281/zenodo.17546242).

## Description

Spatiotemporal encoder is a machine learning tool for rapidly developing and testing neural network autoencoder architectures on timelapse imaging data. 
This project was developed as part of the research described in the manuscript "Inductive bias determines the spatial scale of biological features learned from images".

## Installation

Package and dependency management for this project is done with Poetry. 
To install dependencies, navigate to the project folder in the command line and run:

```bash
$ poetry install
```

If you do not have poetry installed, refer to the documentation they provide [here](https://python-poetry.org).

## Usage

Once dependencies are installed, place imaging data under the paths expected by your dataset YAML (see below). Training reads configuration from `src/conf/`.

### Main config

`src/conf/config.yaml` sets top-level flags:

- **`study_name`**: Base name of the study file in `src/conf/studies/` (without `.yaml`), e.g. `architecture-gastruloid` loads `src/conf/studies/architecture-gastruloid.yaml`.
- **`data_quantity_experiment`**: If `true`, runs the data-quantity sweep; if `false`, runs standard training from the study file.
- **`debug`**: Enables debug behavior in the training runner.

### Study configs

Files in `src/conf/studies/` define **experiments**. Each study YAML has an `experiments` mapping: keys are experiment IDs (e.g. `ae`, `vit`), values list datasets, model assets, and run options.

Required shape:

  ```yaml
  experiments:
    <experiment_id>:
      datasets:
        - <dataset_config_name>
      model:
        architecture: <model_yaml_stem>    # src/conf/models/<stem>.yaml
        num_timepoints: <int>
        params: <hyperparams_yaml_stem>    # src/conf/hyperparams/<stem>.yaml
      general_configs:
        pretrain: <true|false>
        verbose: <true|false>
  ```

The names under `datasets`, `architecture`, and `params` must match the corresponding YAML stems in `src/conf/datasets/`, `src/conf/models/`, and `src/conf/hyperparams/`. Packaged examples include `architecture-gastruloid.yaml` and `architecture-ARCADE.yaml` (extra architectures may appear commented—uncomment or add entries to enable them).

### Dataset configs

Files in `src/conf/datasets/` describe loaders and data layout. Required fields (see `DatasetConfig` in `src/simulation_encoder/dataclass/config_schemas.py`):

- **`loader`**: Which loader handles parsing (e.g. `gastruloid`, `ARCADE`).
- **`image_dir`**, **`label_dir`**: Paths to image stacks and label files.
- **`image_size`**: Edge length of square inputs (pixels).
- **`channels`**: Channel names passed to the model.
- **`batch_size`**, **`val_split`**, **`test_split`**: Batch size and validation/test fractions (`val_split` + `test_split` must be &lt; 1).
- **`keys`**: Which sample keys to include (dataset-specific).

Optional:

- **`augmentations`**: Augmentation list (e.g. `rotate: 90`); may be empty or commented.


### Model configs

Files in `src/conf/models/` define the network. The bundled manuscript-style configs use:

- **`type`**: e.g. `AE` for `ae_small`, `cae_small`, `neuralop_small`, and `vit_small`.
- **`architecture`**: Layer lists under **`encoder`**, **`decoder_image`**, and **`decoder_timepoint`**. Each layer has a **`type`** (PyTorch module name) and kwargs. Runtime placeholders include `num_channels`, `latent_dim`, `image_size`, and tokens such as `decoder_spatial_flat`.

Supported building blocks include standard conv/linear stacks, **`FNO`** (in `neuralop_small.yaml`), and **`VisionTransformer`** (`vit_small.yaml`). Other layouts (e.g. `flat_cnn.yaml`) may differ for one-off experiments.

Example skeleton:

  ```yaml
  type: AE
  architecture:
    encoder:
      - type: <layer>
        # layer kwargs...
    decoder_image:
      - type: <layer>
    decoder_timepoint:
      - type: <layer>
  ```

For the manuscript *Inductive bias influences the spatial scale of biological features learned from images*, the encoder YAMLs referenced in the paper are under `src/conf/models/`:

- MLP: `ae_small.yaml`
- CNN: `cae_small.yaml`
- FNO: `neuralop_small.yaml`
- ViT: `vit_small.yaml`

### Hyperparameter configs

Files in `src/conf/hyperparams/` describe training search spaces. `grid_optimizers.yaml` matches the current schema:

- **`num_epochs`**: Training epochs.
- **`continuous`**: Typically **`image_loss_weight`** with:
  - **`range`**: A single float **or** a two-element `[low, high]` interval (see `HyperparameterRangeConfig` in `config_schemas.py`).
  - **`search`**: e.g. `linear`.
  - **`num_samples`**: Number of samples along the continuous axis when a range is used.
- **`discrete`**: e.g. **`latent_dim`** with **`values`**: list of latent sizes; **`optimizer`** with **`values`**: a list of optimizer dicts. Fields such as `lr`, `betas`, `momentum`, and `nesterov` may be **lists**, which are expanded in a grid (see the Adam and SGD blocks in `grid_optimizers.yaml`).

Once configs are updated, start the Poetry virtual environment:

```bash
$ poetry shell
```

Finally, experiments can be run manually by running the `main.py` file

```bash
$ python src/simulation_encoder/main.py
```

Results and logs will be recorded, and the best performing model in each experiment will have its weights saved in a `.pth` file in the corresponding results folder.