import os
from collections import defaultdict
from typing import Optional, Any

import pandas as pd

from simulation_encoder.loader import CSVLoader, Loader
from simulation_encoder.logger import Logger

from simulation_encoder.models.abstract_cnn import BaseCNN
from simulation_encoder.models.cae import CAE
from simulation_encoder.models.vae import VAE
from simulation_encoder.models.emulator import Emulator

from simulation_encoder.dataclass.loss_data import LossData
from simulation_encoder.dataclass.emulator_results import EmulationResults, EncoderModelResult


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
    losses : dict[str, LossData]
        Dictionary of model names and their corresponding loss data
    """

    def __init__(self, pretrain: bool = False, verbose: bool = False) -> None:
        self.pretrain = pretrain
        self.verbose = verbose

        self.models: dict[str, CAE | VAE] = {}
        self.dataset: Loader = None

        self.losses: dict[str, LossData] = {}

    def add_dataset(self, dataset: Loader) -> None:
        """
        Set the dataset on which models should be trained

        Parameters
        ----------
        dataset : Loader
            Dataset to be used for training and evaluation
        """

        self.dataset = dataset

    def add_models(self, models: list[BaseCNN], logger: Logger) -> None:
        """
        Add models to be trained by the runner

        Parameters
        ----------
        model_param_sets : list[BaseCNN]
            List of models to be trained
        """
        model_num = 0
        for model in models:
            latent_dim = model.latent_dim
            model_id = f"{model.name}_{latent_dim}d_{model_num}"
            while model_id in self.models:
                model_num += 1
                model_id = f"{model.name}_{latent_dim}d_{model_num}"
            self.models[model_id] = model
            self.losses[model_id] = LossData()

        device = model.device
        logger.log(f"{model.name} models added to runner. Device: {device}")

    def get_losses(self) -> dict[str, LossData]:
        """Returns the loss data for all models"""
        return self.losses

    def get_model(self, model_name: str) -> CAE | VAE:
        """Returns the specified model"""
        return self.models[model_name]

    def get_dataset(self) -> Loader:
        """Returns the dataset"""
        return self.dataset

    def run_encoder(self, experiemnt_name: str, logger: Logger) -> dict:
        """Runs the training and evaluation of models"""
        if not self.dataset:
            raise ValueError("No dataset has been added to runner.")
        if not self.models:
            raise ValueError("No models have been added to runner.")

        logger.log(f"Experiment name: {experiemnt_name}")
        logger.log(f"Training points: {self.dataset.n_train} (including any augmented images)")
        logger.log(f"Testing points: {self.dataset.n_test}")

        results: dict = defaultdict(
            lambda: {
                "encoded_data": None,
                "model_state": None,
                "best_model": None,
                "losses": self.losses,
                "plot_data": {"losses": None, "val_losses": None, "grad_norms": None},
            }
        )

        for model_id, model in self.models.items():
            logger.log(f"------------------- {model_id} -------------------")
            losses, val_losses, grad_norms = self._train_model(model_id, model)
            # self._eval_model(model_id, model)

            results[model_id]["model_state"] = model.state_dict()
            encoded_dataset = self._encode_dataset(model)
            results[model_id]["encoded_data"] = encoded_dataset

            results[model_id]["plot_data"] = {
                "losses": losses,
                "val_losses": val_losses,
                "grad_norms": grad_norms,
            }

        best_model = min(self.losses, key=lambda x: min(self.losses[x].losses_val["combined"]))
        results["best_model"] = best_model

        return results

    def _train_model(self, model_name: str, model: CAE) -> tuple:
        """Trains a model on the dataset"""
        train_loader = self.dataset.get_dataloader(dataset_type="train")
        val_loader = self.dataset.get_dataloader(dataset_type="val")

        losses, val_losses, grad_norms = model.fit(
            train_loader,
            val_loader=val_loader,
            pretrain=self.pretrain,
        )

        self.losses[model_name].add_train_loss(losses)
        self.losses[model_name].add_val_loss(val_losses)

        return losses, val_losses, grad_norms

    def _eval_model(self, model_name: str, model: CAE) -> float:
        """Evaluates all models currently in runner"""
        test_loader = self.dataset.get_dataloader(dataset_type="test")
        test_loss = model.eval_one_epoch(test_loader)
        self.losses[model_name].add_test_loss(test_loss)

        return test_loss

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

        if self.dataset.labels:
            for label in self.dataset.labels:
                encoded_train[label] = self.dataset.get_labels(label=label, dataset_type="train")
                encoded_val[label] = self.dataset.get_labels(label=label, dataset_type="val")
                encoded_test[label] = self.dataset.get_labels(label=label, dataset_type="test")

        encoded_train["timepoint"] = self.dataset.get_timepoints(dataset_type="train")
        encoded_val["timepoint"] = self.dataset.get_timepoints(dataset_type="val")
        encoded_test["timepoint"] = self.dataset.get_timepoints(dataset_type="test")

        encoded_train["seed_key"] = self.dataset.get_seed_keys(dataset_type="train")
        encoded_val["seed_key"] = self.dataset.get_seed_keys(dataset_type="val")
        encoded_test["seed_key"] = self.dataset.get_seed_keys(dataset_type="test")

        return {"train": encoded_train, "val": encoded_val, "test": encoded_test}

    def run_emulator(self, conf_name: str) -> Optional[EmulationResults]:
        """Runs emulation for the encoded datasets and returns the results"""
        labels = self.dataset.labels

        if not self.dataset:
            raise ValueError("No dataset has been added to runner.")
        if not labels:
            return None

        emulator_models = ["linear_regression", "random_forest", "svm"]
        encoder_models = self._get_encoder_models(conf_name)

        emulation_results = EmulationResults()
        for experiment, encoder_model_names in encoder_models.items():
            for encoder_model_name in encoder_model_names:
                encoded_dataset = CSVLoader(
                    conf_name=conf_name, exp_id=experiment, model=encoder_model_name, labels=labels
                )
                models = self._initialize_models(emulator_models)

                X_train, y_train = encoded_dataset.get_data("train")
                X_val, y_val = encoded_dataset.get_data("val")

                encoder_model_result = emulation_results.add_encoder_model_result(
                    encoder_model_name
                )
                self._run_emulation_for_model(
                    models, labels, X_train, y_train, X_val, y_val, encoder_model_result
                )

        return emulation_results

    def _run_emulation_for_model(
        self,
        models: dict,
        labels: list,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        encoder_model_result: EncoderModelResult,
    ) -> None:
        """Run emulation for a single encoder model"""
        for label in labels:
            label_result = encoder_model_result.add_label_result(label)
            for model_type, model in models.items():
                model_params = model.grid_search(X_train, y_train[label])
                model = Emulator(model_type=model_type, params=model_params)

                X_train_norm, y_train_norm, X_val_norm, y_val_norm = self._normalize_data(
                    X_train, y_train[label], X_val, y_val[label]
                )

                model.fit(X_train_norm, y_train_norm)
                r2_score = model.evaluate(X_val_norm, y_val_norm)

                label_result.add_model_result(model_type, model_params, r2_score)

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

    def _initialize_models(self, emulator_models: list[str]) -> dict:
        """Initialize emulator models"""
        models = {}
        for model_type in emulator_models:
            models[model_type] = Emulator(model_type=model_type)
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
