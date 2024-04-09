import os
import json

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch

from simulation_encoder.models.cnn import CAE
from simulation_encoder.loader import PNGLoader
from simulation_encoder.logger import ExperimentLogger


class Runner:
    """
    Class for managing the training and saving of models

    Parameters
    ----------
    exp_name : str
        Name of the experiment
    verbose : bool
        Controls if model training is output to console
    """
    def __init__(self, exp_name: str, verbose: bool) -> None:
        self.models = {}
        self.dataset = None
        self.logger = ExperimentLogger(exp_name)
        self.losses = {}
        self.val_losses = {}

        self.exp_name = exp_name
        self.verbose = verbose

    def add_model(self, model_yaml) -> None:
        """
        Add model to be trained by the runner
        """
        with open(model_yaml, "r", encoding='utf-8') as file:
            params = yaml.safe_load(file)
        model_name = os.path.splitext(os.path.basename(model_yaml))[0]
        model = CAE(params)
        self.models[model_name] = model

    def add_dataset(self, data_dir: str, test_split: float, batch_size:int, healthy_flag: bool=False) -> None:
        """
        Add dataset on which models should be trained
        """
        self.dataset = PNGLoader(data_dir, test_split=test_split, batch_size=batch_size, logger=self.logger, healthy_flag=healthy_flag)

    def run_models(self) -> None:
        """
        Runs all models currently in runner
        """
        if not self.dataset:
            raise ValueError("No dataset has been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")
    
        print(self.dataset)
        for model_name, model in self.models.items():
            print(model_name)
            print(model)
            self._run_model(model_name, model)

    def save_model(self, model_name: str, model: CAE) -> None:
        """
        Saves trained model parameters
        """
        self.logger.log(f"Trained model saved at saved_models/{self.exp_name}-{model_name}.pth")
        torch.save(model.state_dict(), f"saved_models/{self.exp_name}-{model_name}.pth")

    def save_results(self) -> None:
        """
        Writes the results of running the models to disk
        """
        with open(f"results/{self.exp_name}", "w", encoding='utf-8') as r_file:
            for model_name, _ in self.models:
                json.dump({"train_loss": self.losses[model_name], "val_loss": self.val_losses[model_name]}, r_file)

    @staticmethod
    def plot_loss(loss: list[float], vloss: list[float], name: str) -> None:
        """
        Plot the loss and validation loss against epochs.

        Parameters
        ----------
        loss : list[float]
            Training loss values.
        vloss : list[float]
            Validation loss values.

        """
        plt.plot(np.arange(len(loss)), loss)
        plt.plot(np.arange(len(vloss)), vloss)
        plt.legend(["Train loss", "Validation loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"figures/{name}.png")

    def _run_model(self, model_name: str, model: CAE) -> None:
        """
        Run a convolutional autoencoder on the ARCADE dataset.
        """
        train_loader = self.dataset.get_train_data()
        test_loader = self.dataset.get_test_data()

        self.logger.log(
            f"Training on {len(train_loader)} examples. Testing on {len(test_loader)} examples."
        )
        if self.verbose:
            print(f"Training on {len(train_loader)} examples. Testing on {len(test_loader)} examples.")

        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.MSELoss()

        losses, val_losses = model.fit(
            train_loader, epochs=10, optimizer=optimizer, loss_fn=loss_fn, val_loader=test_loader
        )

        self.losses[model_name] = losses
        self.val_losses[model_name] = val_losses
