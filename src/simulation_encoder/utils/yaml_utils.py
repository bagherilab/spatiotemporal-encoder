import os
import yaml
import traceback
from pathlib import Path

from pydantic import BaseModel, ValidationError

from simulation_encoder.dataclass.param_sets import DatasetParams
from simulation_encoder.dataclass.config_schemas import (
    StudyConfig,
    HyperparameterConfig,
    ModelArchitectureConfig,
)

# Default conf paths relative to project root (src/simulation_encoder/utils -> project root)
_CONF_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src" / "conf"


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
    yaml_name: str,
    yaml_path: str | None = None,
) -> HyperparameterConfig:
    base = yaml_path or str(_CONF_DIR / "hyperparams")
    yaml_file = yaml_name if yaml_name.endswith(".yaml") else yaml_name + ".yaml"
    path = os.path.join(base, yaml_file)
    return load_yaml(path, HyperparameterConfig)


def load_dataset_yaml(
    yaml_name: str,
    yaml_path: str | None = None,
) -> DatasetParams:
    base = yaml_path or str(_CONF_DIR / "datasets")
    yaml_file = yaml_name if yaml_name.endswith(".yaml") else yaml_name + ".yaml"
    path = os.path.join(base, yaml_file)
    return load_yaml(path, DatasetParams)


def load_study_yaml(
    yaml_name: str,
    yaml_path: str | None = None,
) -> StudyConfig:
    base = yaml_path or str(_CONF_DIR / "studies")
    yaml_file = yaml_name if yaml_name.endswith(".yaml") else yaml_name + ".yaml"
    path = os.path.join(base, yaml_file)
    return load_yaml(path, StudyConfig)


def load_model_yaml(
    architecture_name: str,
    yaml_path: str | None = None,
) -> ModelArchitectureConfig:
    base = yaml_path or str(_CONF_DIR / "models")
    yaml_file = (
        architecture_name if architecture_name.endswith(".yaml") else architecture_name + ".yaml"
    )
    path = os.path.join(base, yaml_file)
    return load_yaml(path, ModelArchitectureConfig)
