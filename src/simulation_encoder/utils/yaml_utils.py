import os
import yaml
import traceback
from pydantic import BaseModel, ValidationError

from simulation_encoder.dataclass.param_sets import DatasetParams
from simulation_encoder.dataclass.config_schemas import (
    StudyConfig,
    HyperparameterConfig,
    ModelArchitectureConfig,
)


def load_yaml(yaml_file: str, config_class: BaseModel) -> BaseModel:
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


def load_hyperparam_yaml(
    yaml_name: str, yaml_path: str = "src/conf/hyperparams"
) -> HyperparameterConfig:
    yaml_file = yaml_name if yaml_name.endswith(".yaml") else yaml_name + ".yaml"
    yaml_path = os.path.join(yaml_path, yaml_file)
    return load_yaml(yaml_path, HyperparameterConfig)


def load_dataset_yaml(yaml_name: str, yaml_path: str = "src/conf/datasets") -> DatasetParams:
    yaml_file = yaml_name if yaml_name.endswith(".yaml") else yaml_name + ".yaml"
    yaml_path = os.path.join(yaml_path, yaml_file)
    return load_yaml(yaml_path, DatasetParams)


def load_study_yaml(yaml_name: str, yaml_path: str = "src/conf/studies") -> StudyConfig:
    yaml_file = yaml_name if yaml_name.endswith(".yaml") else yaml_name + ".yaml"
    yaml_path = os.path.join(yaml_path, yaml_file)
    return load_yaml(yaml_path, StudyConfig)


def load_model_yaml(
    architecture_name: str, yaml_path: str = f"src/conf/models"
) -> ModelArchitectureConfig:
    yaml_file = (
        architecture_name if architecture_name.endswith(".yaml") else architecture_name + ".yaml"
    )
    yaml_path = os.path.join(yaml_path, yaml_file)
    return load_yaml(yaml_path, ModelArchitectureConfig)
