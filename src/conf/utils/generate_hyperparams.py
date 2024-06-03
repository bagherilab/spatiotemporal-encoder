from itertools import product
from typing import Any

import numpy as np


def generate_hyperparameters(
    continuous_params: dict[str, Any], discrete_params: dict[str, Any]
) -> list[dict[str, Any]]:
    continuous_values = get_continuous_values(continuous_params)
    discrete_values = get_discrete_values(discrete_params)

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


def get_continuous_values(continuous_params: dict[str, Any]) -> dict[str, list[float]]:
    continuous_values = {}
    if continuous_params:
        for param, param_info in continuous_params.items():
            num_samples = param_info["num_samples"]
            if param_info["search"] == "linear":
                values = np.linspace(
                    param_info["range"][0], param_info["range"][1], num_samples
                ).tolist()
            elif param_info["search"] == "log":
                values = np.logspace(
                    np.log10(param_info["range"][0]),
                    np.log10(param_info["range"][1]),
                    num_samples,
                ).tolist()
            continuous_values[param] = values
    return continuous_values


def get_discrete_values(discrete_params: dict[str, Any]) -> dict[str, list[float]]:
    discrete_values = {}
    if discrete_params:
        for param, param_info in discrete_params.items():
            values = param_info["values"]
            discrete_values[param] = values
    return discrete_values
