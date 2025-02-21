import cProfile
import pstats
import time

import os

import yaml
import traceback
from pydantic import BaseModel, ValidationError

from simulation_encoder.runner import Runner
from simulation_encoder.writer import Writer
from simulation_encoder.plotter import Plotter
from simulation_encoder.logger import Logger
from simulation_encoder.loaders.loader import Loader
from simulation_encoder.loaders.arcade_loader import ARCADELoader
from simulation_encoder.loaders.alphanumeric_loader import AlphanumericLoader
from simulation_encoder.loaders.gastruloid_loader import GastruloidLoader
from simulation_encoder.loaders.glims_loader import GlimsLoader

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

        # Loaders
        loaders = create_loaders(experiment_config, logger)
        runner.add_loaders(loaders)

        for loader_name, loader in loaders.items():
            writer.write_train_test_indices(loader_name, loader.get_indices())

        # # Models
        model_param_sets = create_model_param_sets(experiment_config.model)
        runner.add_models(model_param_sets)

        # # Run
        encoder_results = runner.run_encoder(experiment_name)
        handle_encoder_results(encoder_results, runner, writer, plotter)

    # Emulation (optional)
    # emulation_results = runner.run_emulator(config_name)
    # if emulation_results:
    #     writer.write_emulation_results(emulation_results)


def create_loaders(experiment_config: ExperimentConfig, logger: Logger) -> dict[str, Loader]:
    """Create a list of loaders from the experiment config file."""
    loaders = {}
    loader_configs = {}
    dataset_names = experiment_config.datasets

    for dataset_name in dataset_names:
        loader_configs[dataset_name] = _load_dataset_yaml(dataset_name)

    for dataset_name, loader_config in loader_configs.items():
        dataset_params = create_dataset_params(dataset_name, loader_config)
        loader = create_loader(dataset_name, dataset_params, logger)
        loaders[dataset_name] = loader

    return loaders


def create_loader(dataset_name: str, dataset_params: DatasetParams, logger: Logger) -> Loader:
    """Create the loader object from the dataset parameters"""
    params_dict = dataset_params.__dict__
    loader_type = params_dict.pop("loader")
    logger.log(f"Creating dataset {dataset_name} with loader - {loader_type}")
    if loader_type.lower() == "arcade":
        loader = ARCADELoader(
            **params_dict,
            logger=logger,
        )
    elif loader_type.lower() == "alphanumeric":
        del params_dict["label_dir"]
        del params_dict["labels"]
        loader = AlphanumericLoader(
            **params_dict,
            logger=logger,
        )
    elif loader_type.lower() == "gastruloid":
        del params_dict["label_dir"]
        del params_dict["labels"]
        loader = GastruloidLoader(
            **params_dict,
            logger=logger,
        )

    elif loader_type.lower() == "glims":
        del params_dict["label_dir"]
        del params_dict["labels"]
        loader = GlimsLoader(
            **params_dict,
            logger=logger,
        )

    return loader


def create_dataset_params(dataset_name: str, dataset_config: DatasetConfig) -> DatasetParams:
    """Create the loader parameters from the experiment config file"""
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
        name=dataset_name,
    )

    return dataset_params


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
    best_model = None
    best_val_loss = float("inf")

    for dataset_name, dataset_results in encoder_results.items():
        for model_id, data in dataset_results.items():
            writer.write_model_state(model_id, dataset_name, data["model_state"])
            writer.write_encoder_results(
                model_id,
                runner.get_loader(dataset_name),
                runner.get_model(model_id),
                data["losses"],
            )

            plotter.line_plot(
                model_id, dataset_name, data["grad_norms"], "grad_norms", "Epoch", "Gradient Norm"
            )

            # Access LossData object for losses
            losses = data["losses"]
            plotter.loss_plot(
                model_id,
                dataset_name,
                losses.losses_train.get("weighted_loss", []),
                losses.losses_val.get("weighted_loss", []),
            )

            if losses.losses_val.get("weighted_loss"):
                final_val_loss = losses.losses_val["weighted_loss"][-1]
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_model_info = {
                        "model_id": model_id,
                        "dataset_name": dataset_name,
                        "data": data,
                    }

    if best_model_info is not None:
        best_model_id = best_model_info["model_id"]
        best_model = runner.get_model(best_model_id)

        best_model_dataset_name = best_model_info["dataset_name"]
        best_model_dataset = runner.get_loader(best_model_dataset_name)

        best_model_data = best_model_info["data"]

        writer.write_encoder_results(
            "_best_model", best_model_dataset, best_model, best_model_data["losses"]
        )
        writer.write_encoded_data(
            "_best_model", best_model_dataset_name, best_model_data["encoded_data"]
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
        raise ValidationError(
            f"Error in loading yaml file: {e}, issue validation with {type(config_class).__name__} param set"
        ) from e


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
