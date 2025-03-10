import os
import itertools

from simulation_encoder.plotter import Plotter
from simulation_encoder.models.model_retrieval import load_models
from simulation_encoder.loaders.loader_retrieval import load_loaders

from latent_model.models.temporal_models import TemporalModel, RNNModel, LSTMModel
from latent_model.loaders.sequence_loader import SequenceLoader
from latent_model.runner import Runner

RESULTS_DIR = "results"
DATA_DIR = "data"
EXPERIMENT_NAME = "arch_simple_cancer_v3"


def main() -> None:
    results_path = f"{RESULTS_DIR}/{EXPERIMENT_NAME}"
    sequential_loaders = create_loaders(results_path)
    temporal_models = create_models(results_path, sequential_loaders)

    runner = Runner()
    runner.add_loaders(sequential_loaders)
    runner.add_models(temporal_models)

    results = runner.run_temporal_model()

    encoder_models = load_models(results_path, best_models_flag=True)
    encoder_loaders = load_loaders(results_path, DATA_DIR)

    for model_name, dataset in encoder_models.items():
        for dataset_name, encoder_model in dataset.items():
            temporal_model = results[model_name][dataset_name]["best_model"]
            loader = sequential_loaders[model_name][dataset_name]
            test_dataloader = loader.get_dataloader("test")
            original_test_dataloader = encoder_loaders[dataset_name].get_dataloader("test")

            for batch, labels in test_dataloader:
                x_batch = batch[:, :-1, :]
                y_batch = batch[:, -1, :]
                y_pred = temporal_model(x_batch)
                image_decoded = encoder_model.decode_image(y_batch)
                pred_image_decoded = encoder_model.decode_image(y_pred)
                Plotter.show_images([image_decoded, pred_image_decoded], ["decoded", "predicted"])
                break


def create_loaders(results_path: str, seq_len: int | None = None) -> dict[str, dict[str, SequenceLoader]]:
    """Creates data loaders for all of the best performing models in a given encoder result"""
    loaders = {}
    for model_name in os.listdir(results_path):
        loaders[model_name] = {}
        model_path = f"{results_path}/{model_name}/_best_model"
        for dataset_name in os.listdir(model_path):
            loaders[model_name][dataset_name] = SequenceLoader(
                f"{model_path}/{dataset_name}/encoded_data.csv", max_seq_len=seq_len
            )
    return loaders


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
            temporal_models[model_name][dataset_name] = create_models_list(
                input_size=num_dims, output_size=num_dims
            )
    return temporal_models


def create_models_list(input_size: int, output_size: int, classification: bool=False) -> list[TemporalModel]:
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
            classification=classification
        )
        models.append(model)

    return models


if __name__ == "__main__":
    main()
