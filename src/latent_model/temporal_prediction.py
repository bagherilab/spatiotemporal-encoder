import itertools
import json
import os

import neuralop
import torch
from torch.optim import SGD, Adam

torch.serialization.add_safe_globals([torch._C._nn.gelu])
torch.serialization.add_safe_globals(
    [neuralop.layers.spectral_convolution.SpectralConv]
)

from latent_model.sequence_loader import SequenceLoader
from latent_model.temporal_models import LSTMModel, RNNModel, TemporalModel
from simulation_encoder.main import create_encoder_model
from simulation_encoder.models.base_nn import BaseNN
from simulation_encoder.utils.yaml_utils import load_model_yaml

RESULTS_DIR = "results"
EXPERIMENT_NAME = "arch_simple_cancer_v3"


def main() -> None:
    results_path = f"{RESULTS_DIR}/{EXPERIMENT_NAME}"
    _ = create_loaders(results_path)
    # sequential_loaders = create_loaders(results_path)
    # models = create_models(results_path, sequential_loaders)

    # runner = Runner()
    # runner.add_loaders(sequential_loaders)
    # runner.add_models(models)

    # results = runner.run_temporal_model()
    # models = get_encoder_model(results_path)

    # print(results)

    # for model_name, dataset in models.items():
    #     for dataset_name, encoder_model in dataset.items():
    #         temporal_model = results[model_name][dataset_name]["best_model"]
    #         loader = sequential_loaders[model_name][dataset_name]
    #         test_dataloader = loader.get_dataloader("test")

    #         for batch, labels in test_dataloader:
    #             x_batch = batch[:, :-1, :]
    #             y_batch = batch[:, -1, :]
    #             y_pred = temporal_model(x_batch)
    #             image_decoded = encoder_model.decode_image(y_batch)
    #             pred_image_decoded = encoder_model.decode_image(y_pred)
    #             Plotter.show_images([image_decoded, pred_image_decoded], ["decoded", "predicted"])
    #             break


def create_loaders(results_path: str) -> dict[str, dict[str, SequenceLoader]]:
    """Creates data loaders for all of the best performing models in a given encoder result"""
    loaders = {}
    for model_name in os.listdir(results_path):
        loaders[model_name] = {}
        model_path = f"{results_path}/{model_name}/_best_model"
        for dataset_name in os.listdir(model_path):
            loaders[model_name][dataset_name] = create_loader(model_path, dataset_name)
    return loaders


def create_loader(model_path: str, dataset_name: str) -> SequenceLoader:
    """Creates a loader for encoded data for a specific model and dataset"""
    return SequenceLoader(f"{model_path}/{dataset_name}/encoded_data.csv")


def create_models(
    results_path: str, loaders: dict[str, dict[str, SequenceLoader]]
) -> dict[str, dict[str, list[TemporalModel]]]:
    temporal_models = {}
    for model_name in os.listdir(results_path):
        temporal_models[model_name] = {}
        model_path = f"{results_path}/{model_name}/_best_model"
        for dataset_name in os.listdir(model_path):
            loader = loaders[model_name][dataset_name]
            num_dims = loader.num_dims
            # seq_len = loader.sequence_len
            temporal_models[model_name][dataset_name] = create_models_list(
                input_size=num_dims, output_size=num_dims
            )
    return temporal_models


def create_models_list(input_size: int, output_size: int) -> list[TemporalModel]:
    model_classes = [RNNModel, LSTMModel]
    hidden_sizes = [32, 64, 128]
    num_layers = [1, 2, 3]
    models = []
    for model_class, hidden_size, num_layer in itertools.product(
        model_classes, hidden_sizes, num_layers
    ):
        model = model_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            output_size=output_size,
        )
        models.append(model)

    return models


def get_encoder_model(results_path: str) -> dict[str, BaseNN]:
    best_models = {}
    for model_type in os.listdir(results_path):
        for model_name in os.listdir(f"{results_path}/{model_type}"):
            best_models[model_type] = {}
            if model_name in ["_best_model"] or model_name.endswith("json"):
                continue
            model_base_name = "_".join(model_name.split("_")[0:-1])
            model_params = load_model_yaml(model_base_name, yaml_path="src/conf/models")
            break

        for dataset_name in os.listdir(f"{results_path}/{model_type}/_best_model"):
            dataset_dir = f"{results_path}/{model_type}/_best_model/{dataset_name}"
            with open(f"{dataset_dir}/results.json") as file:
                results = json.load(file)

            params = results["params"]
            params["optimizer"]["type"] = (
                Adam if params["optimizer"]["type"] == "Adam" else SGD
            )
            num_channels = len(results["channels"])

            model = create_encoder_model(
                model_params, model_base_name, num_channels, params, dataset_dir
            )
            best_models[model_type][dataset_name] = model
    return best_models


if __name__ == "__main__":
    main()
