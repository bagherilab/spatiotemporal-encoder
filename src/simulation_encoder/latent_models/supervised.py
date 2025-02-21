import os
from typing import Optional

import pandas as pd

from simulation_encoder.latent_models.supervised_model import SupervisedModel
from simulation_encoder.loaders.dataset_utils.label_loaders import CSVLoader
from simulation_encoder.dataclass.supervised_results import SupervisedResults, SupervisedModelResult


class SupervisedRunner:
    """Class for running supervised models on learned latent space."""

    def __init__():
        pass

    def run_emulator(self, conf_name: str) -> Optional[SupervisedResults]:
        """Runs emulation for the encoded datasets and returns the results"""
        datasets = self._get_encoder_datasets(conf_name)
        any_labels = any(dataset.labels for dataset in self.datasets.values())
        if not any_labels:
            self._log("No datasets have labels; emulation will not be run.")
            return None

        emulation_results = SupervisedResults()

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
        encoder_model_result: SupervisedModelResult,
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

    def _initialize_models(self, emulator_models: list[str]) -> dict[str, SupervisedModel]:
        """Initialize emulator models"""
        models = {}
        for model_type in emulator_models:
            models[model_type] = SupervisedModel(model_type=model_type, logger=self.logger)
        return models
