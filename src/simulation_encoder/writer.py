import os
import json
import pandas as pd
import torch

from simulation_encoder.runner import Runner
from simulation_encoder.loaders.loader import Loader
from simulation_encoder.models.base_nn import BaseNN
from simulation_encoder.dataclass.loss_data import LossData
from simulation_encoder.dataclass.supervised_results import SupervisedResults


class Writer:
    """Class for writing information to disk"""

    def __init__(self, results_dir: str = "results/", experiment_key: str = ""):
        self.experiment_key = experiment_key
        self.results_dir = results_dir
        self.results_path = os.path.join(results_dir, str(self.experiment_key))

        self._setup()

    def write_encoder_results(
        self, model_name: str, dataset: Loader, model: BaseNN, losses: LossData
    ) -> None:
        """Writes the results of running the models to disk under the dataset_name prefix"""
        dataset_name = dataset.name
        model_path = os.path.join(self.results_path, model_name)
        dataset_path = os.path.join(model_path, dataset_name)
        augmentations = dataset.augmentation_manager.augmentations or []
        self._create_dir(dataset_path)

        results = {
            "model": model_name,
            "architecture": model.name,
            "loader": dataset.__class__.__name__,
            "dataset": dataset_name,
            "channels": dataset.channels,
            "sequence": dataset.sequence,
            "max_sequence_length": dataset.max_sequence_length,
            "model_params": model.params,
            "data_augmentations": augmentations,
            "data_keys": dataset.keys,
            "losses": {
                "train": losses.losses_train,
                "val": losses.losses_val,
                "test": losses.losses_test,
            },
        }

        with open(os.path.join(dataset_path, "results.json"), "w", encoding="utf-8") as r_file:
            json.dump(results, r_file, indent=4)

    def write_train_test_indices(
        self, dataset_name: str, indices: tuple[list[int], list[int], list[int]]
    ) -> None:
        """Writes the train and test indices to disk"""
        indices_path = os.path.join(self.results_path, f"{dataset_name}_indices.json")

        indices_dict = {"train": indices[0], "val": indices[1], "test": indices[2]}
        with open(indices_path, "w", encoding="utf-8") as i_file:
            json.dump(indices_dict, i_file, indent=4)

    def write_encoded_data(
        self,
        model_name: str,
        loader: Loader,
        dataset_name: str,
        model: BaseNN,
    ) -> None:
        """Saves encoded dataset and labels with dataset_name prefix"""
        model_path = os.path.join(self.results_path, model_name)
        dataset_path = os.path.join(model_path, dataset_name)
        self._create_dir(model_path)
        self._create_dir(dataset_path)

        embeddings = Runner._encode_dataset(model, loader)
        combined_embeddings = pd.concat(embeddings.values(), ignore_index=True)
        combined_embeddings.to_csv(os.path.join(dataset_path, "encoded_data.csv"), index=False)

    def write_model_state(self, model_name: str, dataset_name: str, model: BaseNN) -> None:
        """Writes the model state to disk under the model_name directory"""
        model_path = os.path.join(self.results_path, model_name)
        dataset_path = os.path.join(model_path, dataset_name)

        self._create_dir(model_path)
        self._create_dir(dataset_path)

        torch.save(model.state_dict(), os.path.join(dataset_path, "model_state.pth"))

    def write_supervised_results(self, results: SupervisedResults) -> None:
        """Writes the results of all emulators to disk"""
        results_path = os.path.join(self.results_path, "supervised_results.json")

        results = {}
        for encoder_model, encoder_result in results.get_results().items():
            results[encoder_model] = {
                label: {
                    model_type: {
                        "model_params": model_result.model_params,
                        "r2_score": model_result.r2_score,
                    }
                    for model_type, model_result in label_result.model_results.items()
                }
                for label, label_result in encoder_result.label_results.items()
            }

        with open(results_path, "w", encoding="utf-8") as r_file:
            json.dump(results, r_file, indent=4)

    def _setup(self) -> None:
        self._create_dir(self.results_dir)
        self._create_dir(self.results_path)

    def _create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
