import os
import sys
from pathlib import Path
from typing import Any

import yaml

# For local imports in the module
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

from simulation_encoder.runner import Runner
from simulation_encoder.dataclass.param_sets import DatasetParams, ModelParams
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

    main_config = load_yaml(CONFIG_YAML)
    verbose = main_config["general_configs"]["verbose"]

    runner = Runner(verbose)
    dataset_params = create_dataset_params(main_config)
    runner.add_dataset(dataset_params)

    for model in main_config["models"]:
        model_param_sets = create_model_param_sets(model)
        runner.add_models(model_param_sets)

    runner.run()


def create_dataset_params(main_config: dict[str, Any]) -> DatasetParams:
    dataset_params = DatasetParams(
        image_dir=main_config["data"]["image_dir"],
        label_dir=main_config["data"]["label_dir"],
        batch_size=main_config["general_configs"]["batch_size"],
        val_split=main_config["general_configs"]["val_split"],
        test_split=main_config["general_configs"]["test_split"],
        augmentations=main_config["augmentations"],
        keys=main_config["keys"],
    )
    return dataset_params


def create_model_param_sets(model: dict[str, Any]) -> list[ModelParams]:
    model_name = model["architecture"]
    architecture = load_model_architecture(model_name)

    model_params = load_hyperparams(model["params"])
    num_epochs = model_params["num_epochs"]

    continuous_params = model_params["continuous"]
    discrete_params = model_params["discrete"]
    param_sets = generate_hyperparameters(continuous_params, discrete_params)

    model_param_sets = []
    for param_set in param_sets:
        model_params = ModelParams(
            name=model_name,
            architecture=architecture["architecture"],
            num_epochs=num_epochs,
            params=param_set,
        )
        model_param_sets.append(model_params)

    return model_param_sets


def load_yaml(yaml_file: str) -> dict[str, Any]:
    try:
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File {yaml_file} not found") from e


def load_hyperparams(yaml_name: str) -> dict[str, Any]:
    yaml_file = yaml_name if yaml_name.endswith(".yaml") else yaml_name + ".yaml"
    yaml_path = f"src/conf/hyperparams/{yaml_file}"
    return load_yaml(yaml_path)


def load_model_architecture(
    architecture_name: str, yaml_path: str = f"src/conf/models"
) -> dict[str, Any]:
    yaml_file = (
        architecture_name if architecture_name.endswith(".yaml") else architecture_name + ".yaml"
    )
    yaml_path = os.path.join(yaml_path, yaml_file)
    return load_yaml(yaml_path)


if __name__ == "__main__":
    main()
