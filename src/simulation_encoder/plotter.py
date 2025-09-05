import os

import torch
import matplotlib.pyplot as plt


class Plotter:
    """Class for creating and saving plots"""

    def __init__(self, results_dir: str = "results/", experiment_key: str = ""):
        self.experiment_key = experiment_key
        self.results_dir = results_dir
        self.results_path = os.path.join(results_dir, str(self.experiment_key))

        self._setup()

    def line_plot(
        self,
        model_name: str,
        dataset_name: str,
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

        model_path = os.path.join(self.results_path, model_name)
        dataset_path = os.path.join(model_path, dataset_name)
        figure_path = os.path.join(dataset_path, "figures")
        self._create_dir(figure_path)

        plt.savefig(os.path.join(figure_path, f"{title}.png"))
        plt.close()

    def loss_plot(
        self,
        model_name: str,
        dataset_name: str,
        train_loss: list[float],
        val_loss: list[float],
    ) -> None:
        """Creates a plot of the training and validation loss"""
        plt.figure()
        plt.plot(range(len(train_loss)), train_loss, label="Train Loss")
        plt.plot(range(len(val_loss)), val_loss, label="Validation Loss")
        plt.legend()
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        model_path = os.path.join(self.results_path, model_name)
        dataset_path = os.path.join(model_path, dataset_name)
        figure_path = os.path.join(dataset_path, "figures")
        self._create_dir(figure_path)

        plt.savefig(os.path.join(figure_path, "loss.png"))
        plt.close()

    @staticmethod
    def show_images(image_lists: list[list[torch.Tensor]], row_labels: list[str] = []) -> None:
        """TODO"""
        num_rows = len(image_lists)
        if num_rows == 0:
            raise ValueError("At least one list of images must be provided.")

        num_images = len(image_lists[0])
        if not all(len(lst) == num_images for lst in image_lists):
            raise ValueError("All image lists must have the same length.")

        fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images, num_rows * 2))

        if num_rows == 1:
            axes = [axes]
        if num_images == 1:
            axes = [[ax] for ax in axes]

        for i in range(num_rows):
            for j in range(num_images):
                ax = axes[i][j]
                img = image_lists[i][j].squeeze().detach().numpy()
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

            # Add row labels
            if row_labels:
                axes[i][0].set_ylabel(
                    row_labels[i], fontsize=16, rotation=90, labelpad=10, va="center"
                )

        plt.tick_params(left=False, right=False, bottom=False, labelleft=False, labelbottom=False)
        plt.show()

    def _setup(self) -> None:
        self._create_dir(self.results_dir)
        self._create_dir(self.results_path)

    def _create_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
