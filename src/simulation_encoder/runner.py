import os
import json
import uuid
from typing import Any

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

    def __init__(self, verbose: bool = False) -> None:
        self.models: dict[str, CAE] = {}
        self.dataset: PNGLoader = None
        self.verbose = verbose
        self._UUID = uuid.uuid4()
        self.logger = ExperimentLogger(uuid=self._UUID)

        self.losses: dict[str, list[float]] = {}
        self.val_losses: dict[str, list[float]] = {}
        self.test_losses: dict[str, float] = {}

    def add_models(self, model_files: list[str]) -> None:
        """Add models to be trained by the runner"""
        for model_yaml in model_files:
            self.add_model(model_yaml)

    def add_model(self, model_yaml: str) -> None:
        """Add model to be trained by the runner"""
        with open(model_yaml, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        model_name = os.path.splitext(os.path.basename(model_yaml))[0]
        model = CAE(params)
        self.models[model_name] = model

    def add_dataset(
        self, data_dir: str, test_split: float, batch_size: int, healthy_flag: bool = False
    ) -> None:
        """Add dataset on which models should be trained"""
        self.dataset = PNGLoader(
            data_dir,
            test_split=test_split,
            batch_size=batch_size,
            logger=self.logger,
            healthy_flag=healthy_flag,
            uuid=self._UUID,
        )

    def train_models(self, num_epochs: int) -> None:
        """Trains all models currently in runner"""
        if not self.dataset:
            raise ValueError("No dataset has been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")

        self.logger.log("---- TRAINING ----")
        msg = f"Training points: {self.dataset.n_train}"
        self.logger.log(msg)
        if self.verbose:
            print(msg)

        for model_name, model in self.models.items():
            print(f"Training {model_name}")
            self.logger.log("----------------------------")
            self.logger.log(f"Training model {model_name}")
            self.logger.log(f"Model: {model}")
            self._train_model(model_name, model, num_epochs)

    def eval_models(self) -> None:
        """Evaluates all models currently in runner"""
        if not self.dataset:
            raise ValueError("No dataset has been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")

        self.logger.log("---- Testing ----")
        msg = f"Testing points: {self.dataset.n_test}"
        self.logger.log(msg)
        if self.verbose:
            print(msg)

        for model_name, model in self.models.items():
            self.logger.log("----------------------------")
            self.logger.log(f"Evaluating model {model_name}")
            self._eval_model(model_name, model)

    def save_models(self) -> None:
        """Saves trained model parameters"""
        for model_name, model in self.models.items():
            torch.save(model.state_dict(), f"saved_models/{model_name}_{self._UUID}.pth")
            self.logger.log(f"Trained model saved at saved_models/{model_name}_{self._UUID}.pth")

    def save_results(self) -> None:
        """Writes the results of running the models to disk"""
        results: dict[str, Any] = {"models": {}}
        for model_name, _ in self.models.items():
            results["models"][model_name] = {
                "train_loss": self.losses[model_name],
                "val_loss": self.val_losses[model_name],
                "test_loss": self.test_losses[model_name],
            }

        with open(f"results/{self._UUID}.json", "w", encoding="utf-8") as r_file:
            json.dump(results, r_file, indent=4)

    def plot_loss(self) -> None:
        """Plot the loss and validation loss against epochs."""
        for model_name, _ in self.models.items():
            plt.plot(np.arange(len(self.losses[model_name][0])), self.losses[model_name][0])
            plt.plot(np.arange(len(self.val_losses[model_name][0])), self.val_losses[model_name][0])
            plt.legend(["Train loss", "Validation loss"])
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(f"figures/loss_{model_name}_{self._UUID}.png")

    def _train_model(self, model_name: str, model: CAE, num_epochs: int) -> None:
        for train_loader, val_loader in self.dataset.get_cv_splits(k_folds=2):
            self.losses[model_name] = []
            self.val_losses[model_name] = []
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = torch.nn.MSELoss()

            losses, val_losses = model.fit(
                train_loader,
                epochs=num_epochs,
                optimizer=optimizer,
                loss_fn=loss_fn,
                val_loader=val_loader,
            )

            self.losses[model_name].append(losses)
            self.val_losses[model_name].append(val_losses)

    def _eval_model(self, model_name: str, model: CAE) -> None:
        test_loader = self.dataset.get_test_dataloader()
        loss_fn = torch.nn.MSELoss()
        test_loss = model.eval_one_epoch(test_loader, loss_fn)
        self.test_losses[model_name] = test_loss
        self.logger.log(f"Test loss for {model_name}: {test_loss}")
        if self.verbose:
            print(f"Test loss for {model_name}: {test_loss}")
