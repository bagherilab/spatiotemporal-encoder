import os
import json

import pandas as pd

from simulation_encoder.loader import ARCADELoader, AlphaNumericLoader, Loader
from simulation_encoder.models.cae import CAE
from simulation_encoder.dataclass.loss_data import LossData


class Writer:
    """Class for writing information to disk"""

    def __init__(self, results_dir: str = "results/", uuid: str = "none"):
        self.uuid = uuid
        self.results_dir = results_dir
        self.results_path = os.path.join(results_dir, str(self.uuid))

    def write_results(self, model_name: str, model: CAE, dataset: Loader, losses: LossData) -> None:
        """Writes the results of running the models to disk"""
        self._setup()

        model_path = os.path.join(self.results_path, model_name)
        loader = "ARCADE" if isinstance(dataset, ARCADELoader) else "alphanumeric"
        self._create_dir(model_path)
        results = {
            "model": model_name,
            "loader": loader,
            "architecture": model.name,
            "params": model.params,
            "data_augmentations": [aug for aug in dataset.augmentations],
            "keys": dataset.keys,
            "combined_loss": {
                "train": losses.combined_loss_train,
                "val": losses.combined_loss_val,
                "test": losses.combined_loss_test,
            },
            "reconstruction_loss": {
                "train": losses.reconstruction_loss_train,
                "val": losses.reconstruction_loss_val,
                "test": losses.reconstruction_loss_test,
            },
            "timepoint_loss": {
                "train": losses.timepoint_loss_train,
                "val": losses.timepoint_loss_val,
                "test": losses.timepoint_loss_test,
            },
        }
        with open(os.path.join(model_path, "results.json"), "w", encoding="utf-8") as r_file:
            json.dump(results, r_file, indent=4)

    def write_emulation_results(
        self, model_type: str, target: str, model_params: dict[str, float], r2: float
    ) -> None:
        """Writes the results of all emulators"""
        self._setup()
        results_path = os.path.join(self.results_path, "emulator_results.json")

        if not os.path.exists(results_path):
            results = {}
        else:
            with open(results_path, "r") as r_file:
                results = json.load(r_file)

        if model_type not in results:
            results[model_type] = {}

        results[model_type][target] = model_params
        results[model_type][target]["r2"] = r2

        with open(results_path, "w", encoding="utf-8") as r_file:
            json.dump(results, r_file, indent=4)

    def write_indices(self, indices: tuple[list[int], list[int], list[int]]) -> None:
        """Writes the train and test indices to disk"""
        self._setup()
        indices_path = os.path.join(self.results_path, "indices.json")

        indices_dict = {"train": indices[0], "val": indices[1], "test": indices[2]}
        with open(indices_path, "w", encoding="utf-8") as i_file:
            json.dump(indices_dict, i_file, indent=4)

    def write_encoded_data(self, model_name: str, encoded_data: dict[str, pd.DataFrame]) -> None:
        """Saves encoded dataset and labels for downstream tasks"""
        self._setup()
        model_path = os.path.join(self.results_path, model_name)
        self._create_dir(model_path)
        encoded_data_path = os.path.join(model_path, "encoded_data")
        self._create_dir(encoded_data_path)

        for key, data in encoded_data.items():
            data.to_csv(os.path.join(encoded_data_path, f"{key}.csv"), index=False)

    def _setup(self) -> None:
        self._create_dir(self.results_dir)
        self._create_dir(self.results_path)

    def _create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
