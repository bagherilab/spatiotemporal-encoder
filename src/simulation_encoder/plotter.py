import os
import matplotlib.pyplot as plt


class Plotter:
    """Class for creating and saving plots"""

    def __init__(self, results_dir: str = "results/", uuid: str = ""):
        self.uuid = uuid
        self.results_dir = results_dir
        self.results_path = os.path.join(results_dir, str(self.uuid))
        self.figure_path = os.path.join(self.results_path, "figures")

        self._setup()

    def line_plot(
        self,
        data: dict[str, list[float]],
        title: str,
        x_label: str = "",
        y_label: str = "",
    ) -> None:
        """Creates a line plot of the data"""
        plt.figure()
        for _, values in data.items():
            plt.plot(range(len(values)), values)
        plt.legend(data.keys())
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.savefig(os.path.join(self.figure_path, f"{title}.png"))

    def loss_plot(self, train_loss: list[float], val_loss: list[float]) -> None:
        """Creates a plot of the training and validation loss"""
        plt.figure()
        plt.plot(range(len(train_loss)), train_loss, label="Train Loss")
        plt.plot(range(len(val_loss)), val_loss, label="Validation Loss")
        plt.legend()
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.savefig(os.path.join(self.figure_path, "loss.png"))

    def _setup(self) -> None:
        self._create_dir(self.results_dir)
        self._create_dir(self.results_path)
        self._create_dir(self.figure_path)

    def _create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
