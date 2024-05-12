import os
import json

from simulation_encoder.loader import PNGLoader
from simulation_encoder.models.cnn import CAE
from simulation_encoder.dataclass.loss_data import LossData


class Writer:
    """Class for writing information to disk"""

    def __init__(self, results_dir: str = "results/", uuid: str = "none"):
        self.uuid = uuid
        self.results_dir = results_dir
        self.results_path = os.path.join(results_dir, str(self.uuid))

    def write_results(
        self, model_name: str, model: CAE, dataset: PNGLoader, losses: LossData
    ) -> None:
        """Writes the results of running the models to disk"""
        self._setup()

        model_path = os.path.join(self.results_path, model_name)
        self._create_dir(model_path)
        results = {
            "model": model_name,
            "params": model.params,
            "data_augmentations": dataset.augmentations,
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

    def write_indices(self, indices: tuple[list[int], list[int], list[int]]) -> None:
        """Writes the train and test indices to disk"""
        self._setup()
        indices_path = os.path.join(self.results_path, "indices.json")

        indices_dict = {"train": indices[0], "val": indices[1], "test": indices[2]}
        with open(indices_path, "w", encoding="utf-8") as i_file:
            json.dump(indices_dict, i_file, indent=4)

    def _setup(self) -> None:
        self._create_dir(self.results_dir)
        self._create_dir(self.results_path)

    def _create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
