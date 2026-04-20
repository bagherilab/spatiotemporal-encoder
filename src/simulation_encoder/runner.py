<<<<<<< HEAD
import copy
from collections import defaultdict
from copy import deepcopy
import pandas as pd

from simulation_encoder.logger import Logger
from simulation_encoder.loaders.loader import Loader
from simulation_encoder.loaders.arcade_loader import ARCADELoader
from simulation_encoder.loaders.gastruloid_loader import GastruloidLoader
from simulation_encoder.loaders.alphanumeric_loader import AlphanumericLoader
from simulation_encoder.loaders.glims_loader import GlimsLoader

from simulation_encoder.models.ae import AE
from simulation_encoder.models.vae import VAE
from simulation_encoder.models.base_nn import BaseNN

from simulation_encoder.dataclass.loss_data import LossData
from simulation_encoder.dataclass.param_sets import ModelParams, DatasetParams
=======
import os
import json

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch

from simulation_encoder.models.cnn import CAE
from simulation_encoder.loader import PNGLoader
from simulation_encoder.logger import ExperimentLogger
>>>>>>> 3011307 (Refactor out runner class)


class Runner:
    """
    Class for managing the training and saving of models

    Parameters
    ----------
<<<<<<< HEAD
    pretrain : bool
        Controls if models should be pretrained
    logger : Logger
        Logger for tracking training progress
    verbose : bool
        Controls if model training is output to console

    Attributes
    ----------
    models : dict[str, ModelParams]
        Dictionary of model IDs and their corresponding parameter sets
    loader_params : dict[str, DatasetParams]
        Dictionary of loader names and their corresponding parameter sets
    losses : dict[str, LossData]
        Dictionary of model names and their corresponding loss data
    """

    def __init__(
        self, pretrain: bool = False, logger: Logger = None, verbose: bool = False
    ) -> None:
        self.pretrain = pretrain
        self.logger = logger
        self.verbose = verbose

        self.models: dict[str, ModelParams] = {}
        self.loader_params: dict[str, DatasetParams] = {}

        # Cache for created loaders to prevent reinitialization
        self._loader_cache: dict[str, Loader] = {}

        self.losses: dict[str, LossData] = {}

    def add_loader_params(self, loader_param_sets: dict[str, DatasetParams]) -> None:
        """
        Add loader parameter sets to be used for creating loaders

        Parameters
        ----------
        loader_param_sets : dict[str, DatasetParams]
            Dictionary mapping loader names to their parameter sets
        """
        self.loader_params.update(loader_param_sets)
        # Clear the cache when new loader params are added
        self._loader_cache = {}

    def add_model_params(self, model_param_sets: list[ModelParams]) -> None:
        """
        Add models to be trained by the runner

        Parameters
        ----------
        model_param_sets : list[ModelParams]
            List of model parameter sets to be trained
        """
        model_num = 0
        for model_param in model_param_sets:
            model_id = f"{model_param.name}_{model_num}"
            self.models[model_id] = model_param
            self.losses[model_id] = LossData()
            model_num += 1

    def create_loader(self, loader_name: str) -> Loader:
        """
        Returns a created loader instance from the loader parameters

        Parameters
        ----------
        loader_name : str
            Name of the loader to create

        Returns
        -------
        Loader
            Created loader instance
        """
        if loader_name in self._loader_cache:
            return self._loader_cache[loader_name]

        if loader_name not in self.loader_params:
            raise ValueError(f"Loader {loader_name} not found")

        loader = self._create_loader(loader_name)
        self._loader_cache[loader_name] = loader

        return loader

    def create_model(self, model_id: str) -> BaseNN:
        """
        Returns a created model instance from the model parameters

        Parameters
        ----------
        model_id : str
            ID of the model to create

        Returns
        -------
        BaseNN
            Created model instance
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model_params: ModelParams = self.models[model_id]
        params_dict = deepcopy(model_params.__dict__)
        model_type = params_dict.pop("model_type")

        if model_type == "AE":
            return AE(**params_dict, logger=self.logger)
        if model_type == "VAE":
            return VAE(**params_dict, logger=self.logger)
        raise ValueError(f"Model type {model_type} not recognized")

    def run_encoder(self, study_name: str) -> dict:
        """
        Runs the training and evaluation of autoencoder pipeline

        Parameters
        ----------
        study_name : str
            Name of the experiment to run

        Returns
        -------
        dict
            Dictionary of results
        """
        if not self.loader_params:
            raise ValueError("No loader parameters have been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")
        self.logger.set_study_name(study_name)

        results: dict = defaultdict(dict)

        for model_id, _ in self.models.items():
            model = self.create_model(model_id)
            self.logger.set_model_name(model_id)
            self._log(f"Training model {model_id} on device {model.device}")

            for loader_name in self.loader_params:
                loader = self.create_loader(loader_name)
                self._log(f"Training on dataset {loader_name}")
                self._log(f"Training points - {loader.n_train} Testing points - {loader.n_test}")

                losses, val_losses, grad_norms = self._train_model(model_id, model, loader)

                loss_data = LossData()
                loss_data.add_train_loss(losses)
                loss_data.add_val_loss(val_losses)

                model_snapshot = copy.deepcopy(model)
                dataset_results = {
                    "model": model_snapshot,
                    "losses": loss_data,
                    "grad_norms": grad_norms,
                }

                results[loader_name][model_id] = dataset_results

        return results

    def _train_model(self, model_id: str, model: BaseNN, loader: Loader) -> tuple:
        """Trains a model on the dataset"""
        train_loader = loader.get_dataloader(dataset_type="train")
        val_loader = loader.get_dataloader(dataset_type="val")

        losses, val_losses, grad_norms = model.fit(
            train_loader,
            val_loader=val_loader,
            pretrain=self.pretrain,
            patience=7,
            min_delta=0.001,
        )

        self.losses[model_id].add_train_loss(losses)
        self.losses[model_id].add_val_loss(val_losses)

        return losses, val_losses, grad_norms

    def _eval_model(self, model_name: str, model: BaseNN, loader: Loader) -> dict[str, float]:
        """Evaluates all models currently in runner"""
        test_loader = loader.get_dataloader(dataset_type="test")
        test_loss = model.eval_one_epoch(test_loader)
        self.losses[model_name].add_test_loss(test_loss)

        return test_loss

    def _create_loader(self, loader_name: str) -> Loader:
        """
        Helper method to create a loader from parameters

        Parameters
        ----------
        loader_name : str
            Name of the loader to create

        Returns
        -------
        Loader
            Created loader instance
        """
        dataset_params = self.loader_params[loader_name]
        params_dict = deepcopy(dataset_params.__dict__)
        loader_type = params_dict.pop("loader")

        self._log(f"Creating loader {loader_name} with type {loader_type}")

        if loader_type.lower() == "arcade":
            return ARCADELoader(
                **params_dict,
                logger=self.logger,
            )
        else:
            # Only ARCADE loaders have labels currently
            if "label_dir" in params_dict:
                del params_dict["label_dir"]
            if "labels" in params_dict:
                del params_dict["labels"]

            if loader_type.lower() == "alphanumeric":
                return AlphanumericLoader(
                    **params_dict,
                    logger=self.logger,
                )
            if loader_type.lower() == "gastruloid":
                return GastruloidLoader(
                    **params_dict,
                    logger=self.logger,
                )
            if loader_type.lower() == "glims":
                return GlimsLoader(
                    **params_dict,
                    logger=self.logger,
                )
        raise ValueError(f"Invalid loader type specified: {loader_type}")

    def _normalize_data(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> tuple:
        """Normalize the training and validation data"""
        X_train_norm = (X_train - X_train.mean()) / X_train.std()
        y_train_norm = (y_train - y_train.mean()) / y_train.std()
        X_val_norm = (X_val - X_train.mean()) / X_train.std()
        y_val_norm = (y_val - y_train.mean()) / y_train.std()
        return X_train_norm, y_train_norm, X_val_norm, y_val_norm

    @staticmethod
    def _encode_dataset(model: BaseNN, loader: Loader) -> dict[str, pd.DataFrame]:
        """Encodes the dataset using the model. Final dataframe includes labels and seed keys"""
        data_loaders = {
            "train": loader.get_dataloader(dataset_type="train"),
            "val": loader.get_dataloader(dataset_type="val"),
            "test": loader.get_dataloader(dataset_type="test"),
        }

        encoded_data = {}
        num_dims = model.latent_dim
        column_names = [f"dim_{i}" for i in range(num_dims)]

        for split, data_loader in data_loaders.items():
            encoded = model.encode_loader(data_loader).cpu() if data_loader else None
            if encoded is not None:
                encoded_df = pd.DataFrame(encoded, columns=column_names)

                if hasattr(loader, "labels") and loader.labels is not None:
                    for label in loader.labels:
                        encoded_df[label] = loader.get_labels(label=label, dataset_type=split)

                encoded_df["timepoint"] = loader.get_timepoints(dataset_type=split)
                encoded_df["sample_id"] = loader.get_sample_ids(dataset_type=split)
                encoded_df["split"] = split

                encoded_data[split] = encoded_df

        return encoded_data

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)
=======
    exp_name : str
        Name of the experiment
    verbose : bool
        Controls if model training is output to console
    """

    def __init__(self, exp_name: str, verbose: bool) -> None:
        self.models: dict[str, CAE] = {}
        self.dataset: PNGLoader = None
        self.logger = ExperimentLogger(exp_name)
        self.losses: dict[str, list[float]] = {}
        self.val_losses: dict[str, list[float]] = {}

        self.exp_name = exp_name
        self.verbose = verbose

    def add_model(self, model_yaml: str) -> None:
        """
        Add model to be trained by the runner
        """
        with open(model_yaml, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        model_name = os.path.splitext(os.path.basename(model_yaml))[0]
        model = CAE(params)
        self.models[model_name] = model

    def add_dataset(
        self, data_dir: str, test_split: float, batch_size: int, healthy_flag: bool = False
    ) -> None:
        """
        Add dataset on which models should be trained
        """
        self.dataset = PNGLoader(
            data_dir,
            test_split=test_split,
            batch_size=batch_size,
            logger=self.logger,
            healthy_flag=healthy_flag,
        )

    def run_models(self) -> None:
        """
        Runs all models currently in runner
        """
        if not self.dataset:
            raise ValueError("No dataset has been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")

        msg = f"Training on {len(self.dataset._get_train_indices())} examples. Testing on {len(self.dataset._get_test_indices())} examples."  # type: ignore
        self.logger.log(msg)
        if self.verbose:
            print(msg)

        for model_name, model in self.models.items():
            self.logger.log("----------------------------")
            self.logger.log(f"Training model {model_name}")
            self.logger.log(f"Model: {model}")
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
        with open(f"results/{self.exp_name}", "w", encoding="utf-8") as r_file:
            for model_name, _ in self.models.items():
                json.dump(
                    {
                        "train_loss": self.losses[model_name],
                        "val_loss": self.val_losses[model_name],
                    },
                    r_file,
                )

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

        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.MSELoss()

        losses, val_losses = model.fit(
            train_loader, epochs=10, optimizer=optimizer, loss_fn=loss_fn, val_loader=test_loader
        )

        self.losses[model_name] = losses
        self.val_losses[model_name] = val_losses
>>>>>>> 3011307 (Refactor out runner class)
