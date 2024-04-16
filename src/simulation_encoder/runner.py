import os
import uuid

import yaml
import torch

from simulation_encoder.models.cnn import CAE
from simulation_encoder.loader import PNGLoader
from simulation_encoder.logger import ExperimentLogger
from simulation_encoder.writer import Writer


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
        self._UUID = uuid.uuid4()
        self.models: dict[str, CAE] = {}
        self.dataset: PNGLoader = None
        self.writer = Writer(uuid=self._UUID)
        self.logger = ExperimentLogger(uuid=self._UUID)
        self.losses: dict[str, dict[str, list[float]]] = {}
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
        model = CAE(params)
        self.models[model_name] = model
        self.losses[model_name] = {}

    def add_dataset(
        self, data_dir: str, test_split: float, batch_size: int, healthy_flag: bool = False
    ) -> None:
        """Add dataset on which models should be trained"""
        self.dataset = PNGLoader(
            data_dir,
            test_split=test_split,
            batch_size=batch_size,
            logger=self.logger,
            writer=self.writer,
            healthy_flag=healthy_flag,
        )

    def run(self, num_epochs: int) -> None:
        """Runs the training and evaluation of models"""
        if not self.dataset:
            raise ValueError("No dataset has been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")
        uuid_msg = f"Run ID: {self._UUID}"
        train_msg = f"Training points: {self.dataset.n_train}"
        test_msg = f"Testing points: {self.dataset.n_test}"
        self.logger.log(uuid_msg)
        self.logger.log(train_msg)
        self.logger.log(test_msg)
        if self.verbose:
            print(uuid_msg)
            print(train_msg)
            print(test_msg)

        for model_name, model in self.models.items():
            msg = f"------------------- {model_name} -------------------"
            self.logger.log(msg)
            if self.verbose:
                print(msg)

            self._train_model(model_name, model, num_epochs)
            self._eval_model(model_name, model)
            self._save_model(model_name, model)

    def _train_model(self, model_name: str, model: CAE, num_epochs: int) -> None:
        """Trains model on dataset"""
        for train_loader, val_loader in self.dataset.get_cv_splits(k_folds=5):
            self.losses[model_name]["train_loss"] = []
            self.losses[model_name]["val_loss"] = []

            losses, val_losses = model.fit(
                train_loader,
                epochs=num_epochs,
                val_loader=val_loader,
            )

            self.losses[model_name]["train_loss"].append(losses)
            self.losses[model_name]["val_loss"].append(val_losses)
            break  # Only train on first fold

    def _eval_model(self, model_name: str, model: CAE) -> None:
        """Evaluates all models currently in runner"""
        test_loader = self.dataset.get_test_dataloader()
        loss_fn = torch.nn.MSELoss()
        test_loss = model.eval_one_epoch(test_loader, loss_fn)
        self.losses[model_name]["test_loss"] = test_loss
        self.logger.log(f"Test loss: {test_loss}")
        if self.verbose:
            print(f"Test loss: {test_loss}")

    def _save_model(self, model_name: str, model: CAE) -> None:
        """Saves trained model parameters"""
        torch.save(model.state_dict(), f"saved_models/{model_name}_{self._UUID}.pth")
        self.logger.log(f"Trained model saved at saved_models/{model_name}_{self._UUID}.pth")
