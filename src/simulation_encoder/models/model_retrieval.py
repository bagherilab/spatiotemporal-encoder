import os
import json

import neuralop
import torch
from torch.optim import Adam, SGD

from simulation_encoder.models.ae import AE
from simulation_encoder.models.vae import VAE
from simulation_encoder.models.base_nn import BaseNN
from simulation_encoder.utils.yaml_utils import load_model_yaml

torch.serialization.add_safe_globals([torch._C._nn.gelu])
torch.serialization.add_safe_globals([neuralop.layers.spectral_convolution.SpectralConv])


def create_model(model_params, model_base_name, num_channels, num_timepoints, params, dataset_dir):
    # for layer in model_params.architecture.encoder:
    #     if layer.type == 'AdaptiveAvgPool2d':
    #         layer.output_size = 1
    if model_params.type == "AE":
        model = AE(
            name=model_base_name,
            num_channels=num_channels,
            num_timepoints=num_timepoints,
            architecture=model_params.architecture.model_dump(exclude_none=True),
            params=params,
        )
    elif model_params.type == "VAE":
        model = VAE(
            name=model_base_name,
            num_channels=num_channels,
            num_timepoints=num_timepoints,
            architecture=model_params.architecture.model_dump(exclude_none=True),
            params=params,
        )
    else:
        raise ValueError("Model type not supported")

    state_dict = torch.load(f"{dataset_dir}/model_state.pth", weights_only=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loading model {model_base_name} ({trainable:.2e} parameters)")
    state_dict.pop("_metadata", None)
    model.load_state_dict(state_dict)

    return model


def load_models(
    results_path: str,
    num_timepoints: int = 1,
    best_models_flag: bool = False,
) -> dict[str, dict[str, dict | BaseNN]]:
    """
    Unified function to load models from files.

    Parameters:
    -----------
    results_path : str
        Path to the directory containing experiment results.
    best_models_flag : bool, default=False
        If True, only load the best models. If False, load all models.

    Returns:
    --------
    Dict[str, Dict[str, Union[Dict, object]]]
        Nested dictionary containing the loaded models.
    """
    if best_models_flag:
        return _load_best_models(results_path, num_timepoints)
    else:
        return _load_all_models(results_path, num_timepoints)


def _load_best_models(
    results_path: str, num_timepoints: int, best_model_name: str = "_best_model"
) -> dict[str, dict[str, object]]:
    """
    Load only the best models from each experiment.

    Parameters:
    -----------
    results_path : str
        Path to the directory containing experiment results.

    Returns:
    --------
    Dict[str, Dict[str, object]]
        Nested dictionary containing the loaded best models.
    """
    best_models = {}

    for model_type in os.listdir(results_path):
        best_models[model_type] = {}

        model_base_name = None
        model_params = None

        for model_name in os.listdir(f"{results_path}/{model_type}"):
            if model_name in [best_model_name] or model_name.endswith("json"):
                continue

            model_base_name = "_".join(model_name.split("_")[0:-1])
            model_params = load_model_yaml(model_base_name)
            break

        if model_base_name is None:
            continue

        best_model_path = f"{results_path}/{model_type}/{best_model_name}"
        if not os.path.exists(best_model_path):
            continue

        for dataset_name in os.listdir(best_model_path):
            dataset_dir = f"{best_model_path}/{dataset_name}"

            with open(f"{dataset_dir}/results.json") as file:
                results = json.load(file)

            params = results["model_params"]
            params["optimizer"]["type"] = Adam if params["optimizer"]["type"] == "Adam" else SGD
            num_channels = len(results["channels"])

            model = create_model(
                model_params, model_base_name, num_channels, num_timepoints, params, dataset_dir
            )

            best_models[model_type][dataset_name] = model

    return best_models


def _load_all_models(
    results_path: str, num_timepoints: int
) -> dict[str, dict[str, dict[str, object]]]:
    """
    Load all models from the results directory structure.

    Parameters:
    -----------
    results_path : str
        Path to the directory containing experiment results.

    Returns:
    --------
    Dict[str, Dict[str, Dict[str, object]]]
        Nested dictionary: model_type -> model_name -> dataset_name -> model object
    """
    all_models = {}
    best_model_names = ["best", "best_model", "_best", "_best_model"]

    for model_type in os.listdir(results_path):
        print(model_type)

        model_type_path = os.path.join(results_path, model_type)

        for model_name in os.listdir(model_type_path):
            if model_name.lower() in best_model_names:
                continue

            print(model_name)

            model_name_path = os.path.join(model_type_path, model_name)

            if not os.path.isdir(model_name_path):
                continue

            model_base_name = "_".join(model_name.split("_")[:-1])

            try:
                model_params = load_model_yaml(model_base_name)
            except FileNotFoundError:
                continue

            all_models[model_name] = {}

            for dataset_name in os.listdir(model_name_path):
                dataset_dir = os.path.join(model_name_path, dataset_name)
                if not os.path.isdir(dataset_dir):
                    continue

                results_path = os.path.join(dataset_dir, "results.json")
                if not os.path.exists(results_path):
                    continue

                with open(results_path) as file:
                    results = json.load(file)

                params = results["model_params"]
                opt_type = params["optimizer"]["type"]
                params["optimizer"]["type"] = Adam if opt_type == "Adam" else SGD
                num_channels = len(results["channels"])

                model = create_model(
                    model_params,
                    model_base_name,
                    num_channels,
                    num_timepoints,
                    params,
                    dataset_dir
                )
                all_models[model_name][dataset_name] = model

        return all_models