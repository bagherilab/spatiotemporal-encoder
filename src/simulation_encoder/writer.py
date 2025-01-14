import os
import json

import pandas as pd
import torch

from simulation_encoder.loaders.loader import Loader
from simulation_encoder.models.base_nn import BaseNN

from simulation_encoder.dataclass.loss_data import LossData
from simulation_encoder.dataclass.emulator_results import EmulationResults


class Writer:
    """Class for writing information to disk"""

    def __init__(self, results_dir: str = "results/", experiment_name: str = ""):
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.results_path = os.path.join(results_dir, str(self.experiment_name))

        self._setup()

    def write_encoder_results(
        self, model_name: str, dataset: Loader, model: BaseNN, losses: LossData
    ) -> None:
        """Writes the results of running the models to disk under the dataset_name prefix"""
        dataset_name = dataset.name
        model_path = os.path.join(self.results_path, model_name)
        dataset_path = os.path.join(model_path, dataset_name)
        self._create_dir(dataset_path)

        results = {
            "model": model_name,
            "dataset": dataset_name,
            "architecture": model.name,
            "channels": dataset.channels,
            "params": model.params,
            "data_augmentations": dataset.augmentations,
            "keys": dataset.keys,
            "losses": {
                "train": losses.losses_train,
                "val": losses.losses_val,
                "test": losses.losses_test,
            },
        }

        with open(os.path.join(dataset_path, "results.json"), "w", encoding="utf-8") as r_file:
            json.dump(results, r_file, indent=4)

    def write_emulation_results(self, emulation_results: EmulationResults) -> None:
        """Writes the results of all emulators to disk"""
        emulation_results_path = os.path.join(self.results_path, "emulation_results.json")

        results = {}
        for encoder_model, encoder_result in emulation_results.get_results().items():
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

        with open(emulation_results_path, "w", encoding="utf-8") as r_file:
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
        self, model_name: str, dataset_name: str, encoded_data: dict[str, pd.DataFrame]
    ) -> None:
        """Saves encoded dataset and labels with dataset_name prefix"""
        model_path = os.path.join(self.results_path, model_name)
        dataset_path = os.path.join(model_path, dataset_name)
        self._create_dir(model_path)
        self._create_dir(dataset_path)

        full_data = pd.DataFrame()
        for key, data in encoded_data.items():
            full_data = pd.concat([full_data, data], axis=0)

        full_data.to_csv(os.path.join(dataset_path, "encoded_data.csv"), index=False)

    def write_model_state(self, model_name: str, dataset_name: str, model_state: dict) -> None:
        """Writes the model state to disk under the model_name directory"""
        model_path = os.path.join(self.results_path, model_name)
        dataset_path = os.path.join(model_path, dataset_name)
        self._create_dir(model_path)
        self._create_dir(dataset_path)

        torch.save(model_state, os.path.join(dataset_path, "model_state.pth"))

    def _setup(self) -> None:
        self._create_dir(self.results_dir)
        self._create_dir(self.results_path)

    def _create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
