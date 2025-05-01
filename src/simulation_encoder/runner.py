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


class Runner:
    """
    Class for managing the training and saving of models

    Parameters
    ----------
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

        # Cache for created loaders
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
        Runs the training and evaluation of models

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
            train_loader, val_loader=val_loader, pretrain=self.pretrain, patience=5, min_delta=0.001
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
