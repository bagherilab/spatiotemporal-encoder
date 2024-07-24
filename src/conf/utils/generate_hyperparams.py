from itertools import product
from typing import Any

import numpy as np

from simulation_encoder.dataclass.config_schemas import HyperparameterDiscreteConfig, HyperparameterRangeConfig


def generate_hyperparameters(
    continuous_params: HyperparameterRangeConfig, discrete_params: HyperparameterDiscreteConfig
) -> list[dict[str, Any]]:
    """Generates sets of hyperparameters using a grid search over continuous and discrete parameters."""
    continuous_values = _get_continuous_values(continuous_params)
    discrete_values = _get_discrete_values(discrete_params)

    param_sets = []
    for continuous_value in product(*continuous_values.values()):
        for discrete_value in product(*discrete_values.values()):
            param_set = {}
            for i, param in enumerate(continuous_values.keys()):
                param_set[param] = continuous_value[i]
                if param == "image_loss_weight":
                    param_set["timepoint_loss_weight"] = 1 - continuous_value[i]
                    param_set[param] = continuous_value[i]
            for i, param in enumerate(discrete_values.keys()):
                param_set[param] = discrete_value[i]
            param_sets.append(param_set)

    return param_sets

def _get_continuous_values(continuous_params: dict[str, HyperparameterRangeConfig]) -> dict[str, list[float]]:
    continuous_values = {}
    if continuous_params:
        for param, param_info in continuous_params.items():
            num_samples = param_info.num_samples
            if param_info.search == "linear":
                values = np.linspace(
                    param_info.range[0], param_info.range[1], num_samples
                ).tolist()
            elif param_info.search == "log":
                values = np.logspace(
                    np.log10(param_info.range[0]),
                    np.log10(param_info.range[1]),
                    num_samples,
                ).tolist()
            else:
                raise ValueError(f"Unsupported search method: {param_info.search}")
            continuous_values[param] = values
    return continuous_values

def _get_discrete_values(discrete_params: dict[str, HyperparameterDiscreteConfig]) -> dict[str, list[float]]:
    discrete_values = {}
    if discrete_params:
        for param, param_info in discrete_params.items():
            values = param_info.values
            discrete_values[param] = values
    return discrete_values
