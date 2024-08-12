import cProfile
import pstats
import time

import os
from copy import deepcopy

import yaml
import traceback
from pydantic import BaseModel, ValidationError


import sys
from pathlib import Path

# For local imports in the module
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))


from simulation_encoder.runner import Runner
from simulation_encoder.writer import Writer
from simulation_encoder.plotter import Plotter
from simulation_encoder.logger import Logger
from simulation_encoder.loader import Loader, ARCADELoader, AlphaNumericLoader

from simulation_encoder.models.abstract_cnn import BaseCNN
from simulation_encoder.models.cae import CAE
from simulation_encoder.models.vae import VAE

from simulation_encoder.dataclass.param_sets import DatasetParams, ModelParams
from simulation_encoder.dataclass.config_schemas import (
    MainConfig,
    DatasetConfig,
    HyperparameterConfig,
    ExperimentConfig,
    ModelParamsConfig,
    ModelArchitectureConfig,
)
from conf.utils.generate_hyperparams import generate_hyperparameters

CONFIG_YAML = "src/conf/config.yaml"


def main() -> None:
    """Entry point for the simulation encoder. This function reads the config.yaml file and
    creates the necessary objects to run the simulation encoder. The main components are the
    dataset parameters, model parameters, and the runner object. The runner object is responsible
    for running the simulation encoder with the given dataset and model parameters. The dataset
    parameters are created from the config.yaml file, and the model parameters are created from the
    hyperparameter yaml files. The runner object is created with the dataset and model parameters,
    and the run method is called to start the simulation encoder."""

    try:
        main_config = _load_yaml(CONFIG_YAML, MainConfig)
    except ValidationError as e:
        traceback.print_exc()
        raise ValidationError(f"Configuration validation error: {e}") from e

    config_name = main_config.experiment_name  # type: ignore
    for experiment_name, experiment_config in main_config.experiments.items():  # type: ignore
        pretrain = experiment_config.general_configs.pretrain
        verbose = experiment_config.general_configs.verbose

        logger = Logger(log_name=f"{config_name}", verbose=verbose)

        runner = Runner(pretrain, logger, verbose)
        writer = Writer(results_dir=f"results/{config_name}", experiment_name=experiment_name)
        plotter = Plotter(results_dir=f"results/{config_name}", experiment_name=experiment_name)

        # Datasets
        datasets = create_datasets(experiment_config, logger)
        runner.add_datasets(datasets)

        for dataset_name, dataset in datasets.items():
            writer.write_train_test_indices(dataset_name, dataset.get_indices())

        # Models
        models = create_models(experiment_config, logger)
        runner.add_models(models)

        # Run
        encoder_results = runner.run_encoder(experiment_name)
        handle_encoder_results(encoder_results, runner, writer, plotter)

    # Emulation (if needed)
    emulation_results = runner.run_emulator(config_name)
    if emulation_results:
        writer.write_emulation_results(emulation_results)


def create_datasets(experiment_config: ExperimentConfig, logger: Logger) -> dict[str, Loader]:
    """Create a list of datasets from the experiment config file."""
    datasets = {}
    dataset_configs = {}
    dataset_names = experiment_config.datasets
    
    for dataset_name in dataset_names:
        dataset_configs[dataset_name] = _load_dataset_yaml(dataset_name)

    for dataset_name, dataset_config in dataset_configs.items():
        dataset_params = create_dataset_params(dataset_name, dataset_config)
        dataset = create_dataset(dataset_name, dataset_params, logger)
        datasets[dataset_name] = dataset

    return datasets

def create_dataset(dataset_name: str, dataset_params: DatasetParams, logger: Logger) -> Loader:
    """Create the dataset object from the dataset parameters"""
    params_dict = dataset_params.__dict__
    loader = params_dict.pop("loader")
    logger.log(f"Creating dataset {dataset_name} with loader - {loader}")
    if loader.lower() == "arcade":
        dataset = ARCADELoader(
            **params_dict,
            logger=logger,
        )
    elif loader.lower() == "alphanumeric":
        del params_dict["label_dir"]
        del params_dict["labels"]
        dataset = AlphaNumericLoader(
            **params_dict,
            logger=logger,
        )

    return dataset

def create_dataset_params(
    dataset_name: str, dataset_config: DatasetConfig
) -> DatasetParams:
    """Create the dataset parameters from the experiment config file"""
    dataset_params = DatasetParams(
        loader=dataset_config.loader,
        image_dir=dataset_config.image_dir,
        label_dir=dataset_config.label_dir,
        channels=dataset_config.channels,
        batch_size=dataset_config.batch_size,
        val_split=dataset_config.val_split,
        test_split=dataset_config.test_split,
        keys=dataset_config.keys,
        augmentations=dataset_config.augmentations,
        labels=dataset_config.labels,
        name = dataset_name
    )

    return dataset_params

def create_models(experiment_config: ExperimentConfig, logger: Logger) -> list[BaseCNN]:
    """Creates a list of models based on the experiment configuration."""
    model_param_sets = create_model_param_sets(experiment_config.model)
    models = []

    for i, model_param_set in enumerate(model_param_sets):
        params_dict = model_param_set.__dict__
        model_type = params_dict.pop("model_type")
        logger.log(f"Creating model with architecture - {model_type} params - {experiment_config.model.params}")

        if model_type == "CAE":
            model = CAE(**deepcopy(model_param_set.__dict__), logger=logger)
        elif model_type == "VAE":
            model = VAE(**deepcopy(model_param_set.__dict__), logger=logger)
        else:
            raise ValueError(f"Model type {model_type} not recognized")

        models.append(model)

    return models

def create_model_param_sets(model_config: ModelParamsConfig) -> list[ModelParams]:
    """Create the model parameters from model config files and hyperparameter yaml files."""

    model_name = model_config.architecture
    num_channels = model_config.num_channels
    model_yaml = _load_model_yaml(model_name)
    
    model_params = _load_hyperparam_yaml(model_config.params)
    num_epochs = model_params.num_epochs
    continuous_params = model_params.continuous
    discrete_params = model_params.discrete
    param_sets = generate_hyperparameters(continuous_params, discrete_params)

    model_param_sets = []
    for param_set in param_sets:
        model_params = ModelParams(
            name=model_name,
            model_type=model_yaml.type,  # type: ignore
            architecture=model_yaml.architecture.model_dump(exclude_none=True),  # type: ignore
            num_channels=num_channels,
            num_epochs=num_epochs,
            params=param_set,
        )
        model_param_sets.append(model_params)

    return model_param_sets

def handle_encoder_results(
    encoder_results: dict, runner: Runner, writer: Writer, plotter: Plotter
) -> None:
    """Handles writing and plotting of encoder results"""
    for dataset_name, dataset_results in encoder_results.items():
        for model_id, data in dataset_results.items():
            writer.write_encoded_data(model_id, dataset_name, data["encoded_data"])
            writer.write_model_state(model_id, dataset_name, data["model_state"])
            writer.write_encoder_results(model_id, runner.get_dataset(dataset_name), runner.get_model(model_id), data["losses"])

            # Plot results
            plotter.line_plot(model_id, data["grad_norms"], "grad_norms", "Epoch", "Gradient Norm")

            # Access LossData object for losses
            losses = data["losses"]
            plotter.loss_plot(
                model_id, 
                losses.losses_train.get("combined", []), 
                losses.losses_val.get("combined", [])
            )

def _load_yaml(yaml_file: str, config_class: BaseModel) -> BaseModel:
    try:
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
        return config_class(**config)  # type: ignore
    except FileNotFoundError as e:
        traceback.print_exc()
        raise FileNotFoundError(f"File {yaml_file} not found") from e
    except ValidationError as e:
        traceback.print_exc()
        raise ValidationError(f"Configuration validation error: {e}") from e

def _load_hyperparam_yaml(yaml_name: str) -> HyperparameterConfig:
    yaml_file = yaml_name if yaml_name.endswith(".yaml") else yaml_name + ".yaml"
    yaml_path = f"src/conf/hyperparams/{yaml_file}"
    return _load_yaml(yaml_path, HyperparameterConfig)

def _load_dataset_yaml(yaml_name: str) -> DatasetParams:
    yaml_file = yaml_name if yaml_name.endswith(".yaml") else yaml_name + ".yaml"
    yaml_path = f"src/conf/datasets/{yaml_file}"
    return _load_yaml(yaml_path, DatasetParams)

def _load_model_yaml(
    architecture_name: str, yaml_path: str = f"src/conf/models"
) -> ModelArchitectureConfig:
    yaml_file = (
        architecture_name if architecture_name.endswith(".yaml") else architecture_name + ".yaml"
    )
    yaml_path = os.path.join(yaml_path, yaml_file)
    return _load_yaml(yaml_path, ModelArchitectureConfig)

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        start_time = time.time()
        main()
        end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("profile_output.prof")
