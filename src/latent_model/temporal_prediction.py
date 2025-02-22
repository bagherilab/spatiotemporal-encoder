import os
import itertools

from latent_model.temporal_models import TemporalModel, RNNModel, LSTMModel
from latent_model.sequence_loader import SequenceLoader
from latent_model.runner import Runner

RESULTS_DIR = "results"
EXPERIMENT_NAME = "test"

def main() -> None:
    results_path = f"{RESULTS_DIR}/{EXPERIMENT_NAME}"
    loaders = create_loaders(results_path)
    models = create_models(results_path, loaders)

    runner = Runner()
    runner.add_loaders(loaders)
    runner.add_models(models)

    results = runner.run_temporal_model()
    print(results)


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

def create_models(results_path:str, loaders: dict[str, dict[str, SequenceLoader]]) -> dict[str, dict[str, list[TemporalModel]]]:
    temporal_models = {}
    for model_name in os.listdir(results_path):
        temporal_models[model_name] = {}
        model_path = f"{results_path}/{model_name}/_best_model"
        for dataset_name in os.listdir(model_path):
            loader = loaders[model_name][dataset_name]
            num_dims = loader.num_dims
            seq_len = loader.sequence_len
            temporal_models[model_name][dataset_name] = create_models_list(input_size=num_dims, output_size=num_dims)
    return temporal_models


def create_models_list(input_size: int, output_size: int) -> list[TemporalModel]:
    model_classes = [RNNModel, LSTMModel]
    hidden_sizes = [32, 64, 128]
    num_layers = [1, 2, 3]
    models = []
    for model_class, hidden_size, num_layer in itertools.product(model_classes, hidden_sizes, num_layers):
        models.append(model_class(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, output_size=output_size))

    return models

    
if __name__ == "__main__":
    main()