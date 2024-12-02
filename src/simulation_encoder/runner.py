import os
import copy
from collections import defaultdict
from typing import Optional, Union
from copy import deepcopy


import pandas as pd

from simulation_encoder.logger import Logger
from simulation_encoder.loaders.loader import Loader
from simulation_encoder.loaders.util_loaders import CSVLoader

from src.simulation_encoder.models.base_nn import BaseNN
from src.simulation_encoder.models.ae import AE
from simulation_encoder.models.pretrained_ae import PretrainedAE
from simulation_encoder.models.vae import VAE
from simulation_encoder.models.emulator import Emulator

from simulation_encoder.dataclass.loss_data import LossData
from simulation_encoder.dataclass.emulator_results import EmulationResults, EncoderModelResult
from simulation_encoder.dataclass.param_sets import ModelParams


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
    models : dict[str, AE]
        Dictionary of model names and their corresponding autoencoder models
    dataset : Loader
        Dataset to be used for training and evaluation
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
        self.datasets: dict[str, Loader] = {}

        self.losses: dict[str, LossData] = {}

    def add_datasets(self, datasets: dict[str, Loader]) -> None:
        """
        Set the dataset on which models should be trained

        Parameters
        ----------
        dataset : Loader
            Dataset to be used for training and evaluation
        """
        self.datasets = datasets

    def add_models(self, model_param_sets: list[ModelParams]) -> None:
        """
        Add models to be trained by the runner

        Parameters
        ----------
        model_param_sets : list[BaseNN]
            List of models to be trained
        """
        model_num = 0
        for model_param in model_param_sets:
            latent_dim = model_param.params.get("latent_dim")
            model_id = f"{model_param.name}_{latent_dim}d_{model_num}"
            while model_id in self.models:
                model_num += 1
                model_id = f"{model_param.name}_{latent_dim}d_{model_num}"
            self.models[model_id] = model_param
            self.losses[model_id] = LossData()

    def get_model(self, model_id: str) -> BaseNN:
        """Returns the specified model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model_params: ModelParams = self.models[model_id]
        params_dict = deepcopy(model_params.__dict__)
        model_type = params_dict.pop("model_type")

        if model_type == "AE":
            return AE(**params_dict, logger=self.logger)
        elif model_type == "PretrainedAE":
            return PretrainedAE(**params_dict, logger=self.logger)
        elif model_type == "VAE":
            return VAE(**params_dict, logger=self.logger)
        else:
            raise ValueError(f"Model type {model_type} not recognized")

    def get_losses(self) -> dict[str, LossData]:
        """Returns the loss data for all models"""
        return self.losses

    def get_dataset(self, dataset_name: str) -> Loader:
        """Returns the dataset"""
        return self.datasets[dataset_name]

    def run_encoder(self, experiment_name: str) -> dict:
        """Runs the training and evaluation of models"""
        if not self.datasets:
            raise ValueError("No dataset has been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")
        self.logger.set_experiment_name(experiment_name)

        results: dict = defaultdict(dict)

        for model_id, _ in self.models.items():
            model = self.get_model(model_id)
            self.logger.set_model_name(model_id)
            self._log(f"Training model {model_id} on device {model.device}")

            for dataset_name, dataset in self.datasets.items():
                self._log(f"Training on dataset {dataset_name}")
                self._log(f"Training points - {dataset.n_train} Testing points - {dataset.n_test}")

                losses, val_losses, grad_norms = self._train_model(model_id, model, dataset)

                loss_data = LossData()
                loss_data.add_train_loss(losses)
                loss_data.add_val_loss(val_losses)

                encoded_dataset = self._encode_dataset(model, dataset)
                model_snapshot = copy.deepcopy(model.state_dict())

                dataset_results = {
                    "encoded_data": encoded_dataset,
                    "model_state": model_snapshot,
                    "losses": loss_data,
                    "grad_norms": grad_norms,
                }

                results[dataset_name][model_id] = dataset_results

        return results

    def _train_model(self, model_id: str, model: AE, dataset: Loader) -> tuple:
        """Trains a model on the dataset"""
        train_loader = dataset.get_dataloader(dataset_type="train")
        val_loader = dataset.get_dataloader(dataset_type="val")

        losses, val_losses, grad_norms = model.fit(
            train_loader,
            val_loader=val_loader,
            pretrain=self.pretrain,
        )

        self.losses[model_id].add_train_loss(losses)
        self.losses[model_id].add_val_loss(val_losses)

        return losses, val_losses, grad_norms

    def _eval_model(self, model_name: str, model: AE, dataset: Loader) -> float:
        """Evaluates all models currently in runner"""
        test_loader = dataset.get_dataloader(dataset_type="test")
        test_loss = model.eval_one_epoch(test_loader)
        self.losses[model_name].add_test_loss(test_loss)

        return test_loss

    def _encode_dataset(self, model: AE, dataset: Loader) -> dict[str, pd.DataFrame]:
        """Encodes the dataset using the model. Final dataframe includes labels and seed keys"""
        data_loaders = {
            "train": dataset.get_dataloader(dataset_type="train"),
            "val": dataset.get_dataloader(dataset_type="val"),
            "test": dataset.get_dataloader(dataset_type="test"),
        }

        encoded_data = {}
        num_dims = model.latent_dim
        column_names = [f"dim_{i}" for i in range(num_dims)]

        for split, loader in data_loaders.items():
            # Encode data and convert to DataFrame
            encoded = model.encode_loader(loader).cpu() if loader else None
            if encoded is not None:
                encoded_df = pd.DataFrame(encoded, columns=column_names)

                # Add labels if they exist
                if dataset.labels:
                    for label in dataset.labels:
                        encoded_df[label] = dataset.get_labels(label=label, dataset_type=split)

                # Add timepoint and seed_key
                encoded_df["timepoint"] = dataset.get_timepoints(dataset_type=split)
                encoded_df["seed_key"] = dataset.get_seed_keys(dataset_type=split)

                # Store the dataframe in the dictionary
                encoded_data[split] = encoded_df

        return encoded_data

    def run_emulator(self, conf_name: str) -> Optional[EmulationResults]:
        """Runs emulation for the encoded datasets and returns the results"""
        datasets = self.datasets
        if not self.datasets:
            datasets = self._get_encoder_datasets(conf_name)
        else:
            any_labels = any(dataset.labels for dataset in self.datasets.values())
            if not any_labels:
                self._log("No datasets have labels; emulation will not be run.")
                return None

        emulation_results = EmulationResults()

        for dataset_name, dataset in datasets.items():
            if not self.datasets:
                labels = ["activity", "growth", "symmetry"]
            else:
                labels = dataset.labels

            if not labels:
                continue

            emulator_models = ["linear_regression", "random_forest", "mlp"]
            encoder_models = self._get_encoder_models(conf_name)

            self._log(f"Running emulation on {dataset_name} encoded datasets")

            for experiment, encoder_model_names in encoder_models.items():
                self.logger.set_experiment_name(experiment)
                for encoder_model_name in encoder_model_names:
                    self.logger.set_model_name(encoder_model_name)

                    encoded_dataset = CSVLoader(
                        conf_name=conf_name,
                        exp_id=experiment,
                        model=encoder_model_name,
                        dataset_name=dataset_name,
                        labels=labels,
                    )
                    models = self._initialize_models(emulator_models)

                    X_train, y_train = encoded_dataset.get_data("train")
                    X_val, y_val = encoded_dataset.get_data("val")

                    dataset_result = emulation_results.add_dataset_result(dataset_name)
                    encoder_model_result = dataset_result.add_encoder_model_result(
                        encoder_model_name
                    )

                    X = (X_train, X_val)
                    y = (y_train, y_val)
                    self._run_emulation_for_model(models, labels, X, y, encoder_model_result)

        return emulation_results

    def _run_emulation_for_model(
        self,
        models: dict,
        labels: list,
        X: tuple[pd.DataFrame, pd.DataFrame],
        y: tuple[pd.DataFrame, pd.DataFrame],
        encoder_model_result: EncoderModelResult,
    ) -> None:
        """Run emulation for a single encoder model"""
        X_train, X_val = X
        y_train, y_val = y

        for label in labels:
            self._log(f"Target - {label}")
            label_result = encoder_model_result.add_label_result(label)

            for model_type, model in models.items():
                best_params = model.grid_search(X_train, y_train[label])
                model = Emulator(model_type=model_type, params=best_params)

                X_train_norm, y_train_norm, X_val_norm, y_val_norm = self._normalize_data(
                    X_train, y_train[label], X_val, y_val[label]
                )

                model.fit(X_train_norm, y_train_norm)
                r2_score = model.evaluate(X_val_norm, y_val_norm)

                label_result.add_model_result(model_type, best_params, r2_score)

    def _get_encoder_models(self, conf_name: str) -> dict[str, list[str]]:
        """Get list of encoder model folder names"""
        encoder_models: dict[str, list[str]] = {}
        for experiment in os.listdir(f"results/{conf_name}"):
            encoder_models[experiment] = []
            for model in os.listdir(f"results/{conf_name}/{experiment}"):
                if self._is_model_folder(model):
                    encoder_models[experiment].append(model)
            encoder_models[experiment].sort()
        return encoder_models

    def _get_encoder_datasets(self, conf_name: str) -> dict[str, Optional[list[str]]]:
        """Get list of dataset folder names for each experiment."""
        datasets: dict[str, Optional[list[str]]] = {}

        for experiment in os.listdir(f"results/{conf_name}"):

            for model in os.listdir(f"results/{conf_name}/{experiment}"):
                model_dir = f"results/{conf_name}/{experiment}/{model}"
                if os.path.isdir(model_dir):
                    for dataset in os.listdir(model_dir):
                        if dataset == "figures":
                            continue
                        dataset_dir = f"{model_dir}/{dataset}"
                        if os.path.isdir(dataset_dir):
                            datasets[dataset] = None

        datasets = dict(sorted(datasets.items(), key=lambda x: x[0]))

        return datasets

    def _initialize_models(self, emulator_models: list[str]) -> dict[str, Emulator]:
        """Initialize emulator models"""
        models = {}
        for model_type in emulator_models:
            models[model_type] = Emulator(model_type=model_type, logger=self.logger)
        return models

    def _normalize_data(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> tuple:
        """Normalize the training and validation data"""
        X_train_norm = (X_train - X_train.mean()) / X_train.std()
        y_train_norm = (y_train - y_train.mean()) / y_train.std()
        X_val_norm = (X_val - X_train.mean()) / X_train.std()
        y_val_norm = (y_val - y_train.mean()) / y_train.std()
        return X_train_norm, y_train_norm, X_val_norm, y_val_norm

    def _is_model_folder(self, folder_name: str) -> bool:
        """Check if folder is a model folder"""
        folder_chunks = folder_name.split("_")
        if len(folder_chunks) < 3:
            return False

        model_id = folder_chunks[-1]
        dim = folder_chunks[-2]
        try:
            int(dim[:-1])
            int(model_id)
        except ValueError:
            return False

        return True

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)
