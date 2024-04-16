import os
import json
import numpy as np
import matplotlib.pyplot as plt


class Writer:
    """Class for writing information to disk"""

    def __init__(self, results_dir: str = "results/", uuid: str = "none"):
        self.uuid = uuid
        self.results_path = os.path.join(results_dir, str(self.uuid))
        self.figures_dir = os.path.join(self.results_path, "figures")

        self._create_dir(results_dir)
        self._create_dir(self.results_path)
        self._create_dir(self.figures_dir)

    def write_results(self, model_name: str, losses: dict[str, dict[str, list[float]]]) -> None:
        """Writes the results of running the models to disk"""
        model_path = os.path.join(self.results_path, model_name)
        self._create_dir(model_path)
        results = {
            "model": model_name,
            "reconstruction_loss": {
                "train": losses[model_name]["train_loss"],
                "val": losses[model_name]["val_loss"],
                "test": losses[model_name]["test_loss"],
            },
        }
        with open(os.path.join(model_path, "results.json"), "w", encoding="utf-8") as r_file:
            json.dump(results, r_file, indent=4)

    def write_indices(self, train_idices: list[int], test_indices: list[int]) -> None:
        """Writes the train and test indices to disk"""
        indices = {"train": train_idices, "test": test_indices}
        indices_path = os.path.join(self.results_path, "indices.json")
        with open(indices_path, "w", encoding="utf-8") as i_file:
            json.dump(indices, i_file, indent=4)

    def write_loss_plots(self, model_name: str, losses: dict[str, dict[str, list[float]]]) -> None:
        """Saves plots of the training and validation loss for each model"""
        plt.plot(np.arange(len(losses[model_name]["train_loss"]), losses[model_name]["train_loss"]))
        plt.plot(np.arange(len(losses[model_name]["val_loss"]), losses[model_name]["val_loss"]))
        plt.legend(["Train loss", "Validation loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{self.figures_dir}/loss_{model_name}.png")

    def _create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
