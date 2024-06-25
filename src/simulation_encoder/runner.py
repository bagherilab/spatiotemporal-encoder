import os
import uuid
from copy import deepcopy

import torch
import pandas as pd

from simulation_encoder.loader import ARCADELoader, CSVLoader, AlphaNumericLoader, Loader
from simulation_encoder.logger import ExperimentLogger
from simulation_encoder.writer import Writer
from simulation_encoder.models.cae import CAE
from simulation_encoder.models.vae import VAE 
from simulation_encoder.models.emulator import Emulator
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
    dataset : Loader
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
        self.models: dict[str, CAE|VAE] = {}
        self.dataset: Loader = None

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

        params_dict = dataset_params.__dict__
        loader = params_dict.pop("loader")
        if loader == "ARCADE":
            self.dataset = ARCADELoader(
                **params_dict,
                logger=self.logger,
            )
        elif loader == "alphanumeric":
            del params_dict["label_dir"]
            del params_dict["labels"]
            self.dataset = AlphaNumericLoader(
                **params_dict,
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
            params_dict = model_param_set.__dict__
            model_type = params_dict.pop("model_type")
            if model_type == "CAE":
                model = CAE(**deepcopy(model_param_set.__dict__), logger=self.logger)
            elif model_type == "VAE":
                model = VAE(**deepcopy(model_param_set.__dict__), logger=self.logger)
            else:
                raise ValueError(f"Model type {model_type} not recognized")
            

            latent_dim = model_param_set.params["latent_dim"]
            model_id = f"{model_param_set.name}_{latent_dim}d_{model_num}"
            while model_id in self.models:
                model_num += 1
                model_id = f"{model_param_set.name}_{latent_dim}d_{model_num}"
            self.models[model_id] = model
            self.losses[model_id] = LossData()

        device = model.device
        self.logger.log(f"{model_param_set.name} models added to runner. Device: {device}")

    def run_encoder(self) -> None:
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
            encoded_dataset = self._encode_dataset(model)
            self.writer.write_encoded_data(model_id, encoded_dataset)

        best_model = min(self.losses, key=lambda x: self.losses[x].combined_loss_val)

        self.writer.write_results(
            "_best_model",
            self.models[best_model],
            self.dataset,
            self.losses[best_model],
        )

    def run_emulator(self) -> None:
        # labels = self.dataset.labels
        # exp_id = self.dataset.exp_id
        labels = ["activity", "growth", "symmetry"]
        exp_id = "092a30b5-db7a-4a13-b22d-5fe174d790ce"

        encoder_models = []
        for folder_name in os.listdir(f"results/{exp_id}"):
            if folder_name.startswith("1c_"):
                encoder_models.append(folder_name)
        encoder_models.sort()

        emulator_models = ["linear_regression", "random_forest", "svm"]

        for encoder_model in encoder_models:
            print(encoder_model)
            encoded_dataset = CSVLoader(exp_id=exp_id, model=encoder_model, labels=labels)
            models = {}
            for model_type in emulator_models:
                models[model_type] = Emulator(model_type=model_type)

            X_train, y_train = encoded_dataset.get_data("train")
            X_val, y_val = encoded_dataset.get_data("val")

            for label in labels:
                best_model_type = None
                best_model_params = None
                best_r2 = float("-inf")
                for model_type, model in models.items():
                    model_params = model.grid_search(X_train, y_train[label])
                    model = Emulator(model_type=model_type, params=model_params)

                    X_train = (X_train - X_train.mean()) / X_train.std()
                    y_train[label] = (y_train[label] - y_train[label].mean()) / y_train[label].std()
                    X_val = (X_val - X_train.mean()) / X_train.std()
                    y_val[label] = (y_val[label] - y_train[label].mean()) / y_train[label].std()

                    model.fit(X_train, y_train[label])
                    r2 = model.evaluate(X_val, y_val[label])
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_type = model_type
                        best_model_params = model_params

                self.writer.write_emulation_results(
                    best_model_type, label, best_model_params, best_r2
                )

    def _train_model(self, model_name: str, model: CAE) -> None:
        """Trains a model on the dataset"""
        train_loader = self.dataset.get_dataloader(dataset_type="train")
        val_loader = self.dataset.get_dataloader(dataset_type="val")
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
        test_loader = self.dataset.get_dataloader(dataset_type="test")
        test_loss = model.eval_one_epoch(test_loader)
        self.losses[model_name].add_test_loss(test_loss)
        self.logger.log(f"Test loss: {self.losses[model_name].combined_loss_test}")

    def _encode_dataset(self, model: CAE) -> dict[str, pd.DataFrame]:
        """Encodes the dataset using the model. Final dataframe includes labels and seed keys"""
        encoded_train = model.encode_loader(self.dataset.get_dataloader(dataset_type="train")).cpu()
        encoded_val = model.encode_loader(self.dataset.get_dataloader(dataset_type="val")).cpu()
        encoded_test = model.encode_loader(self.dataset.get_dataloader(dataset_type="test")).cpu()

        num_dims = model.latent_dim
        column_names = [f"dim_{i}" for i in range(num_dims)]

        encoded_train = pd.DataFrame(encoded_train, columns=column_names)
        encoded_val = pd.DataFrame(encoded_val, columns=column_names)
        encoded_test = pd.DataFrame(encoded_test, columns=column_names)

        try:
            for label in self.dataset.labels:
                encoded_train[label] = self.dataset.get_labels(label=label, dataset_type="train")
                encoded_val[label] = self.dataset.get_labels(label=label, dataset_type="val")
                encoded_test[label] = self.dataset.get_labels(label=label, dataset_type="test")
        except AttributeError:
            pass

        encoded_train["timepoint"] = self.dataset.get_timepoints(dataset_type="train")
        encoded_val["timepoint"] = self.dataset.get_timepoints(dataset_type="val")
        encoded_test["timepoint"] = self.dataset.get_timepoints(dataset_type="test")

        encoded_train["seed_key"] = self.dataset.get_seed_keys(dataset_type="train")
        encoded_val["seed_key"] = self.dataset.get_seed_keys(dataset_type="val")
        encoded_test["seed_key"] = self.dataset.get_seed_keys(dataset_type="test")

        return {"train": encoded_train, "val": encoded_val, "test": encoded_test}

    def _save_model(self, model_name: str, model: CAE) -> None:
        """Saves trained model parameters"""
        torch.save(model.state_dict(), f"results/{self._UUID}/{model_name}/trained_model.pth")
        self.logger.log(
            f"Trained model saved at results/{self._UUID}/{model_name}/trained_model.pth"
        )
