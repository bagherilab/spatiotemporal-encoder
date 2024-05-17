import uuid
from typing import Any

import torch

from simulation_encoder.loader import PNGLoader
from simulation_encoder.logger import ExperimentLogger
from simulation_encoder.writer import Writer
from simulation_encoder.models.cae import CAE
from simulation_encoder.dataclass.param_sets import DatasetParams, ModelParams
from simulation_encoder.dataclass.loss_data import LossData
from simulation_encoder.plotter import line_plot, loss_plot


class Runner:
    """
    Class for managing the training and saving of models

    Parameters
    ----------
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

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

        self._UUID = uuid.uuid4()
        self.models: dict[str, CAE] = {}
        self.dataset: PNGLoader = None
        self.writer = Writer(uuid=self._UUID)
        self.logger = ExperimentLogger(uuid=self._UUID, verbose=verbose)
        self.losses: dict[str, LossData] = {}

    def add_dataset(self, dataset_params: DatasetParams) -> None:
        """
        Set the dataset on which models should be trained

        Parameters
        ----------
        dataset_params : DatasetParams
            Object containing dataset parameters
        """
        self.dataset = PNGLoader(
            **dataset_params.__dict__,
            logger=self.logger,
        )

        self.writer.write_indices(self.dataset.get_indices())

    def add_models(self, model_param_sets: list[ModelParams]) -> None:
        """
        Add models to be trained by the runner

        Parameters
        ----------
        model_param_sets : list[ModelParams]
            List of dataclasses containing model hyperparameters
        """
        model_num = 0
        for model_param_set in model_param_sets:
            model = CAE(**model_param_set.__dict__.copy(), logger=self.logger)
            model_id = f"{model_param_set.name}_{model_num}"
            while model_id in self.models:
                model_num += 1
                model_id = f"{model_param_set.name}_{model_num}"
            self.models[model_id] = model
            self.losses[model_id] = LossData()

    def run(self) -> None:
        """Runs the training and evaluation of models"""
        if not self.dataset:
            raise ValueError("No dataset has been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")

        self.logger.log(f"Run ID: {self._UUID}")
        self.logger.log(f"Training points: {self.dataset.n_train} (including any augmented images)")
        self.logger.log(f"Testing points: {self.dataset.n_test}")

        for model_id, model in self.models.items():
            self.logger.log(f"------------------- {model_id} -------------------")
            self._train_model(model_id, model)
            # self._eval_model(model_id, model)
            self._save_model(model_id, model)
            self.writer.write_results(model_id, model, self.dataset, self.losses[model_id])

        best_model = min(self.losses, key=lambda x: self.losses[x].combined_loss_val)
        self.writer.write_results(
            "_best_model", self.models[best_model], self.dataset, self.losses[best_model]
        )

        

    def _train_model(self, model_name: str, model: CAE) -> None:
        """Trains a model on the dataset"""
        train_loader = self.dataset.get_train_dataloader()
        val_loader = self.dataset.get_val_dataloader()
        losses, val_losses, grad_norms = model.fit(
            train_loader,
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
