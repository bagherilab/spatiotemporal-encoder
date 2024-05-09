import os
import uuid
from typing import Optional

import yaml
import torch

from simulation_encoder.loader import PNGLoader
from simulation_encoder.logger import ExperimentLogger
from simulation_encoder.writer import Writer
from simulation_encoder.models.cnn import CAE
from simulation_encoder.dataclass.loss_data import LossData
from simulation_encoder.plotter import line_plot, loss_plot


class Runner:
    """
    Class for managing the training and saving of models

    Parameters
    ----------
    augmentations : list[str]
        List of augmentations to be applied to the dataset
    verbose : bool
        Controls if model training is output to console

    Attributes
    ----------
    UUID : uuid.UUID
        Unique identifier for the run
    models : dict[str, CAE]
        Dictionary of model names and their corresponding CAE models
    dataset : PNGLoader
        Dataset to be used for training and evaluation
    writer : Writer
        Writer object for saving things to disk
    logger : ExperimentLogger
        Logger object for logging
    losses : dict[str, LossData]
        Dictionary of model names and their corresponding loss data
    """

    def __init__(self, augmentations: Optional[list[str]] = None, verbose: bool = False) -> None:
        self._UUID = uuid.uuid4()
        self.augmentations = augmentations or []
        self.models: dict[str, CAE] = {}
        self.dataset: PNGLoader = None
        self.writer = Writer(uuid=self._UUID)
        self.logger = ExperimentLogger(uuid=self._UUID, verbose=verbose)
        self.losses: dict[str, LossData] = {}
        self.verbose = verbose

    def add_models(self, model_files: list[str]) -> None:
        """Add models to be trained by the runner"""
        for model_yaml in model_files:
            self.add_model(model_yaml)

    def add_model(self, model_yaml: str) -> None:
        """Add model to be trained by the runner"""
        with open(model_yaml, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        model_name = os.path.splitext(os.path.basename(model_yaml))[0]
        model = CAE(params=params, logger=self.logger)
        self.models[model_name] = model
        self.losses[model_name] = LossData()

    def add_dataset(
        self,
        image_dir: str,
        label_dir: str,
        keys: list[str],
        val_split: float,
        test_split: float,
        batch_size: int,
    ) -> None:
        """
        Add dataset on which models should be trained

        Parameters
        ----------
        image_dir : str
            Path to the directory containing the images
        label_dir : str
            Path to the directory containing the labels
        keys : list[str]
            List of prefixes for the images that will be loaded
        val_split : float
            Fraction of the dataset to be used for validation
        test_split : float
            Fraction of the dataset to be used for testing
        batch_size : int
            Batch size for the DataLoader
        """
        self.dataset = PNGLoader(
            image_dir=image_dir,
            label_dir=label_dir,
            keys=keys,
            val_split=val_split,
            test_split=test_split,
            batch_size=batch_size,
            logger=self.logger,
            writer=self.writer,
            augmentations=self.augmentations,
        )
        self.keys = keys

    def run(self, num_epochs: int) -> None:
        """Runs the training and evaluation of models"""
        if not self.dataset:
            raise ValueError("No dataset has been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")

        self.logger.log(f"Run ID: {self._UUID}")
        self.logger.log(f"Training points: {self.dataset.n_train} (including any augmented images)")
        self.logger.log(f"Testing points: {self.dataset.n_test}")

        for model_name, model in self.models.items():
            self.logger.log(f"------------------- {model_name} -------------------")

            self._train_model(model_name, model, num_epochs)
            self._eval_model(model_name, model)
            self._save_model(model_name, model)
            self.writer.write_results(
                model_name, self.keys, self.augmentations, self.losses[model_name]
            )

    def _train_model(self, model_name: str, model: CAE, num_epochs: int) -> None:
        """Trains a model on the dataset"""
        train_loader = self.dataset.get_train_dataloader()
        val_loader = self.dataset.get_val_dataloader()
        losses, val_losses, grad_norms = model.fit(
            train_loader,
            epochs=num_epochs,
            val_loader=val_loader,
        )

        self.losses[model_name].add_train_loss(losses)
        self.losses[model_name].add_val_loss(val_losses)

        line_plot(grad_norms, "grad_norms", self._UUID, model_name, "Epoch", "Gradient Norm")
        loss_plot(losses["combined"], val_losses["combined"], self._UUID, model_name)

    def _eval_model(self, model_name: str, model: CAE) -> None:
        """Evaluates all models currently in runner"""
        test_loader = self.dataset.get_test_dataloader()
        test_loss = model.eval_one_epoch(test_loader)
        self.losses[model_name].add_test_loss(test_loss)
        self.logger.log(f"Test loss: {self.losses[model_name].combined_loss_test}")

    def _save_model(self, model_name: str, model: CAE) -> None:
        """Saves trained model parameters"""
        torch.save(model.state_dict(), f"results/{self._UUID}/{model_name}/trained_model.pth")
        self.logger.log(
            f"Trained model saved at results/{self._UUID}/{model_name}/trained_model.pth"
        )
