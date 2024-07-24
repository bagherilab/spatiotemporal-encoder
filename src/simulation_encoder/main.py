import cProfile
import pstats
import time

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

# For local imports in the module
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

from simulation_encoder.runner import Runner
from simulation_encoder.writer import Writer
from simulation_encoder.plotter import Plotter
from simulation_encoder.dataclass.param_sets import DatasetParams, ModelParams
from simulation_encoder.dataclass.config_schemas import MainConfig, HyperparameterConfig, ExperimentConfig, ModelConfig
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
        print(f"Configuration validation error: {e}")
        return

    config_name = main_config.experiment_name
    for experiment_name, experiment_config in main_config.experiments.items():
        try:
            pretrain = experiment_config.general_configs.pretrain
            verbose = experiment_config.general_configs.verbose

            runner = Runner(pretrain, verbose)
            writer = Writer(results_dir=f"results/{config_name}", experiment_name=experiment_name)
            plotter = Plotter(results_dir=f"results/{config_name}", experiment_name=experiment_name)
            dataset_params = create_dataset_params(experiment_name, experiment_config)
            indices = runner.add_dataset(dataset_params)
            writer.write_train_test_indices(indices)

            n_channels = len(experiment_config.general_configs.channels)
            model = experiment_config.model
            model_param_sets = create_model_param_sets(model, n_channels)
            runner.add_models(model_param_sets)

            encoder_results = runner.run_encoder()
            handle_encoder_results(encoder_results, runner, writer, plotter)

            emulation_results = runner.run_emulator(config_name)
            if emulation_results:
                writer.write_emulation_results(emulation_results)
        except Exception as e:
            print(f"Error processing experiment '{experiment_name}': {e}")
            continue


def create_dataset_params(experiment_name: str, experiment_config: ExperimentConfig) -> DatasetParams:
    """Create the dataset parameters from the experiment config file"""

    if "alphanumeric" in experiment_name.lower():
        dataset_params = DatasetParams(
            loader="alphanumeric",
            channels=experiment_config.general_configs.channels,
            image_dir=experiment_config.data.image_dir,
            batch_size=experiment_config.general_configs.batch_size,
            val_split=experiment_config.general_configs.val_split,
            test_split=experiment_config.general_configs.test_split,
            keys=experiment_config.keys,
            augmentations=experiment_config.data.augmentations,
        )
    elif "arcade" in experiment_name.lower():
        dataset_params = DatasetParams(
            loader="ARCADE",
            channels=experiment_config.general_configs.channels,
            image_dir=experiment_config.data.image_dir,
            label_dir=experiment_config.data.label_dir,
            batch_size=experiment_config.general_configs.batch_size,
            val_split=experiment_config.general_configs.val_split,
            test_split=experiment_config.general_configs.test_split,
            keys=experiment_config.keys,
            labels=experiment_config.emulator_targets,
            augmentations=experiment_config.data.augmentations,
        )

    return dataset_params


def create_model_param_sets(model: ModelConfig, n_channels: int) -> list[ModelParams]:
    """Create the model parameters from model config files and hyperparameter yaml files"""

    model_name = model.architecture
    model_yaml = _load_model_yaml(model_name)

    model_params = _load_hyperparams(model.params)
    num_epochs = model_params.num_epochs
    continuous_params = model_params.continuous
    discrete_params = model_params.discrete
    param_sets = generate_hyperparameters(continuous_params, discrete_params)

    model_param_sets = []
    for param_set in param_sets:
        model_params = ModelParams(
            name=model_name,
            model_type=model_yaml["type"],
            architecture=model_yaml["architecture"],
            num_channels=n_channels,
            num_epochs=num_epochs,
            params=param_set,
        )
        model_param_sets.append(model_params)

    return model_param_sets


def handle_encoder_results(
    encoder_results: dict, runner: Runner, writer: Writer, plotter: Plotter
) -> None:
    """Handles writing and plotting of encoder results"""
    best_model_id = encoder_results["best_model"]
    best_losses = None
    dataset = runner.get_dataset()

    for model_id, data in encoder_results.items():
        if model_id == "best_model":
            continue

        writer.write_encoded_data(model_id, data["encoded_data"])
        writer.write_model_state(model_id, data["model_state"])

        plot_data = data["plot_data"]
        plotter.line_plot(model_id, plot_data["grad_norms"], "grad_norms", "Epoch", "Gradient Norm")
        plotter.loss_plot(model_id, plot_data["losses"]["combined"], plot_data["val_losses"]["combined"])

        losses = data["losses"][model_id]
        writer.write_encoder_results(model_id, runner.get_model(model_id), dataset, losses)

        if model_id == best_model_id:
            best_losses = losses

    writer.write_encoder_results(
        "_best_model",
        runner.get_model(best_model_id),
        dataset,
        best_losses,
    )

def _load_yaml(yaml_file: str, config_class: BaseModel) -> BaseModel:
    try:
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
        return config_class(**config)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File {yaml_file} not found") from e
    except ValidationError as e:
        raise ValidationError(f"Configuration validation error: {e}") from e

def _load_hyperparams(yaml_name: str) -> HyperparameterConfig:
    yaml_file = yaml_name if yaml_name.endswith(".yaml") else yaml_name + ".yaml"
    yaml_path = f"src/conf/hyperparams/{yaml_file}"
    return _load_yaml(yaml_path, HyperparameterConfig)


def _load_model_yaml(architecture_name: str, yaml_path: str = f"src/conf/models") -> dict[str, Any]:
    yaml_file = (
        architecture_name if architecture_name.endswith(".yaml") else architecture_name + ".yaml"
    )
    yaml_path = os.path.join(yaml_path, yaml_file)
    return _load_yaml(yaml_path, dict)  # Assuming model yaml is a dict


if __name__ == "__main__":
    # with cProfile.Profile() as pr:
    #     main()
    
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats("profile_output.prof")


    start_time = time.time()
    main()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
