import copy
from collections import defaultdict
from copy import deepcopy

import pandas as pd

from simulation_encoder.dataclass.loss_data import LossData
from simulation_encoder.dataclass.param_sets import ModelParams
from simulation_encoder.loaders.loader import Loader
from simulation_encoder.logger import Logger
from simulation_encoder.models.ae import AE
from simulation_encoder.models.base_nn import BaseNN
from simulation_encoder.models.vae import VAE


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
        self.loaders: dict[str, Loader] = {}

        self.losses: dict[str, LossData] = {}

    def add_loaders(self, loaders: dict[str, Loader]) -> None:
        """
        Set the loaders on which models should be trained

        Parameters
        ----------
        dataset : Loader
            Dataset to be used for training and evaluation
        """
        self.loaders = loaders

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
            model_id = f"{model_param.name}_{model_num}"
            self.models[model_id] = model_param
            self.losses[model_id] = LossData()
            model_num += 1

    def get_model(self, model_id: str) -> BaseNN:
        """Returns the specified model"""
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

    def get_losses(self) -> dict[str, LossData]:
        """Returns the loss data for all models"""
        return self.losses

    def get_loader(self, loader_name: str) -> Loader:
        """Returns the dataset"""
        return self.loaders[loader_name]

    def run_encoder(self, experiment_name: str) -> dict:
        """Runs the training and evaluation of models"""
        if not self.loaders:
            raise ValueError("No loaders has been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")
        self.logger.set_experiment_name(experiment_name)

        results: dict = defaultdict(dict)

        for model_id, _ in self.models.items():
            model = self.get_model(model_id)
            self.logger.set_model_name(model_id)
            self._log(f"Training model {model_id} on device {model.device}")

            for loader_name, loader in self.loaders.items():
                self._log(f"Training on dataset {loader_name}")
                self._log(
                    f"Training points - {loader.n_train} Testing points - {loader.n_test}"
                )

                losses, val_losses, grad_norms = self._train_model(
                    model_id, model, loader
                )

                loss_data = LossData()
                loss_data.add_train_loss(losses)
                loss_data.add_val_loss(val_losses)

                encoded_dataset = self._encode_dataset(model, loader)
                model_snapshot = copy.deepcopy(model.state_dict())

                dataset_results = {
                    "encoded_data": encoded_dataset,
                    "model_state": model_snapshot,
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
            patience=5,
            min_delta=0.001,
        )

        self.losses[model_id].add_train_loss(losses)
        self.losses[model_id].add_val_loss(val_losses)

        return losses, val_losses, grad_norms

    def _eval_model(
        self, model_name: str, model: BaseNN, loader: Loader
    ) -> dict[str, float]:
        """Evaluates all models currently in runner"""
        test_loader = loader.get_dataloader(dataset_type="test")
        test_loss = model.eval_one_epoch(test_loader)
        self.losses[model_name].add_test_loss(test_loss)

        return test_loss

    def _encode_dataset(self, model: BaseNN, loader: Loader) -> dict[str, pd.DataFrame]:
        """Encodes the dataset using the model. Final dataframe includes labels and seed keys"""
        data_loaders = {
            "train": loader.get_dataloader(dataset_type="train"),
            "val": loader.get_dataloader(dataset_type="val"),
            "test": loader.get_dataloader(dataset_type="test"),
        }

        encoded_data = {}
        num_dims = model.latent_dim
        column_names = [f"dim_{i}" for i in range(num_dims)]

        for split, datat_loader in data_loaders.items():
            # Encode data and convert to DataFrame
            encoded = model.encode_loader(datat_loader).cpu() if datat_loader else None
            if encoded is not None:
                encoded_df = pd.DataFrame(encoded, columns=column_names)

                # Add labels if they exist
                if hasattr(loader, "labels") and loader.labels is not None:
                    for label in loader.labels:
                        encoded_df[label] = loader.get_labels(
                            label=label, dataset_type=split
                        )

                # Add timepoint and sample_id
                encoded_df["timepoint"] = loader.get_timepoints(dataset_type=split)
                encoded_df["sample_id"] = loader.get_sample_ids(dataset_type=split)

                # Store the dataframe in the dictionary
                encoded_data[split] = encoded_df

        return encoded_data

    def _normalize_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
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
