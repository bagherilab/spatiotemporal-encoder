# Spatiotemporal encoder repository

[![Build Status](https://github.com/bagherilab/spatiotemporal-encoder/workflows/build/badge.svg)](https://github.com/bagherilab/spatiotemporal-encoder/actions?query=workflow%3Abuild)
[![Codecov](https://codecov.io/gh/bagherilab/simulation-encoder/graph/badge.svg?token=nZSUf47ltR)](https://codecov.io/gh/bagherilab/simulation-encoder)
[![Lint Status](https://github.com/bagherilab/spatiotemporal-encoder/workflows/lint/badge.svg)](https://github.com/bagherilab/spatiotemporal-encoder/actions?query=workflow%3Alint)
[![Documentation](https://github.com/bagherilab/spatiotemporal-encoder/workflows/documentation/badge.svg)](https://bagherilab.github.io/spatiotemporal-encoder/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[Description](#description) | [Installation](#installation) | [Usage](#usage)

## Note
This version of the code is under active development. For a stable version of the code associated with the manuscript "Fourier neural operator vs. convolutional neural network-based feature learning for biological image analysis", please use the version on [Zenodo](https://doi.org/10.5281/zenodo.17073719)

## Description

Spatiotemporal encoder is a machine learning tool for rapidly developing and testing neural network autoencoder architectures on timelapse imaging data. 
This project was developed as part of the research described in the manuscript "Architectural bias in feature learning: Fourier Neural Operators vs. Convolutional Neural Networks for biological images".

## Installation

Package and dependency management for this project is done with Poetry. 
To install dependencies, navigate to the project folder in the command line and run:

```bash
$ poetry install
```

If you do not have poetry installed, refer to the documantation they provide [here](https://python-poetry.org).

## Usage

Once dependencies are installed, add your data file (currently only `csv` files are supported) to the data folder. 
Next, there are several config files that inform the program on operating details.
All config files are located inside of the `src/conf` directory.

### Main config

The `config.yaml` file outlines high-level experimental details, incluing:
- The study name, which should be the same as the corresponding `study config`.
- Whether the experiment is a quantity experiment to test the effects of different amounts of training data (`quantity_experiment`)

### Study configs

Inside the `conf/studies` directory, config files can be specified for any study the user wants run. 
Examples can be found in the directory, but they must include:
- A list of studies (`experiments`) to run
- At least one experiment in the format

  ```yaml
  experiments:
    [experiment name]:
        datasets:
            - [dataset name]
        model:
            architecture: [model name]
            num_timepoints: [num timepoins per sample]
            params: [parameter set name]
        general_configs:
            verbose: [true/false]
  ```

- The datasets, architecture, and parameter set names must match their corresponding config files

### Dataset configs

Inside the `conf/datasets` directory, dataset configs can be used to specify the dataset location and hyperparameters. 

Examples can be found in the directory, but they must include:
- The loader responsible to managing the data. The primary difference between the loaders is parsing any image metadata from the file name.
- The image directory where the jpg/png files can be found.
- The hyperparameters associated with the dataset in the following format:

  ```yaml
    loader: [loader name]
    image_dir: [image path]
    image_size: [num pixels in image side length]
    channels:
        - [image channels model should consider]
    batch_size: [batch size]
    val_split: [data fraction to use for validation]
    test_split: [data fraction to use for testing]
    keys:
        - [key in image names to include in study]
    augmentations:
        - rotate: [degree of rotation]
  ```

### Model configs

Inside the `conf/models` directory, model configs can be used to specify the model architecture.
This allows quick model development without modifying any python code. 

Examples can be found in the directory, but they must include:
- The type of model to construct. Currently only vanilla autoencoders (`AE`) are compatible.
- The architecture and layers of the encoder, spatial decoder (`decoder_image`), and temporal decoder (`decoder_timepoint`) in the following format:

  ```yaml
    type: AE
    architecture:
        encoder:
            - type: [layer type]
            [layer parameters]
        decoder_image:
            - type: [layer type]
            [layer parameters]
        decoder_timepoint:
            - type: [layer type]
            [layer parameters]
  ```
- Most existing pytorch layers should be compatible with this yaml format.

### Hyperparameters configs

Inside the `conf/hyperparams` directory, configs can be used to specify the model hyperparameters.

Examples can be found in the directory, but they must include:
- Either a single value or range of image decoder loss wieghts (`image_loss_weight`). This will determing how much the spatial information is prioritized over temporal information in the encodings.
- A list of dimensionality sizes to test, determining the size of the learned latent dimension.
- Other optimzer parameters in the following format:

  ```yaml
    num_epochs: [num epochs]
    continuous:
    image_loss_weight:
        range: [lower bound, upper bound]
        search: linear
        num_samples: [number of samples]
    discrete:
    latent_dim:
        values: [dimensionality values]
    optimizer:
        values:
        - type: [Adam/SGD]
            lr: [learning rate]
  ```

Once config files have been updated, start the Poetry virtual environment:

```bash
$ poetry shell
```

Finally, experiments can be run manually by running the `main.py` file

```bash
$ python src/simulation_encoder/main.py
```

Results and logs will be recorded, and the best performing model in each experiment will have it's weights saved in a `.pth` file in the corresponding results folder.
