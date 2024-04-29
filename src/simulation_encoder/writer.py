import os
import json

from simulation_encoder.dataclass.loss_data import LossData


class Writer:
    """Class for writing information to disk"""

    def __init__(self, results_dir: str = "results/", uuid: str = "none"):
        self.uuid = uuid
        self.results_path = os.path.join(results_dir, str(self.uuid))
        self.figures_dir = os.path.join(self.results_path, "figures")

        self._create_dir(results_dir)
        self._create_dir(self.results_path)
        self._create_dir(self.figures_dir)

    def write_results(self, model_name: str, losses: LossData) -> None:
        """Writes the results of running the models to disk"""
        model_path = os.path.join(self.results_path, model_name)
        self._create_dir(model_path)
        results = {
            "model": model_name,
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

    def write_indices(
        self, train_idices: list[int], val_indices: list[int], test_indices: list[int]
    ) -> None:
        """Writes the train and test indices to disk"""
        indices = {"train": train_idices, "val": val_indices, "test": test_indices}
        indices_path = os.path.join(self.results_path, "indices.json")
        with open(indices_path, "w", encoding="utf-8") as i_file:
            json.dump(indices, i_file, indent=4)

    def _create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
