from itertools import product
from typing import Any

import numpy as np

from simulation_encoder.dataclass.config_schemas import (
    HyperparameterDiscreteConfig,
    HyperparameterRangeConfig,
)


def generate_hyperparameters(
    continuous_params: HyperparameterRangeConfig,
    discrete_params: HyperparameterDiscreteConfig,
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


def _get_continuous_values(
    continuous_params: dict[str, HyperparameterRangeConfig],
) -> dict[str, list[float]]:
    continuous_values = {}
    if continuous_params:
        for param, param_info in continuous_params.items():
            num_samples = param_info.num_samples
            if num_samples == 1:
                values = [param_info.range]
            else:
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


def _get_discrete_values(
    discrete_params: dict[str, HyperparameterDiscreteConfig],
) -> dict[str, list[Any]]:
    discrete_values = {}
    if discrete_params:
        for param, param_info in discrete_params.items():
            if param == "optimizer":
                discrete_values[param] = _get_optimizer_values(param_info.values)
            else:
                discrete_values[param] = param_info.values
    return discrete_values


def _get_optimizer_values(
    optimizer_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    optimizer_values = []

    for opt in optimizer_configs:
        try:
            opt_type = (
                globals()[opt["type"]]
                if isinstance(opt["type"], str) and opt["type"] in globals()
                else opt["type"]
            )
        except KeyError:
            raise ValueError(f"Invalid optimizer type: {opt['type']}")

        opt_params = {k: v for k, v in opt.items() if k != "type"}
        opt_combinations = list(product(*opt_params.values()))

        for combo in opt_combinations:
            param_set = {k: v for k, v in zip(opt_params.keys(), combo, strict=False)}
            param_set["type"] = opt_type

            # Skip this combination if nesterov is True and momentum is 0
            if param_set.get("nesterov", False) and param_set.get("momentum", 0) == 0:
                continue

            optimizer_values.append(param_set)

    return optimizer_values
