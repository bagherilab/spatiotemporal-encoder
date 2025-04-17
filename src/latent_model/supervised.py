import os
from typing import Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler

from simulation_encoder.logger import Logger
from simulation_encoder.loaders.loader_retrieval import load_loaders
from simulation_encoder.loaders.dataset_utils.label_loaders import CSVLoader
from simulation_encoder.dataclass.supervised_results import SupervisedResults, SupervisedModelResult

from latent_model.models.supervised_model import SupervisedRegressor


class SupervisedRunner:
    """Class for running supervised models on learned latent space."""

    def __init__(self, logger: Logger = None) -> None:
        """
        Initialize the supervised runner.
        
        Parameters
        ----------
        logger : Logger
            Logger object for tracking progress
        """
        self.logger = logger

    def run_emulator(self, conf_name: str) -> Optional[SupervisedResults]:
        """
        Runs emulation for the encoded datasets and returns the results.
        
        Parameters
        ----------
        conf_name : str
            Configuration name
            
        Returns
        -------
        Optional[SupervisedResults]
            Results of emulation, or None if no datasets have labels
        """
        results_path = f"results/{conf_name}"
        image_base_dir = "data"

        emulation_results = SupervisedResults()
        emulator_models = ["linear_regression", "random_forest", "svr"]
        encoder_models = self._get_encoder_models(conf_name)

        label_datasets = CSVLoader(
            conf_name=conf_name,
            exp_id="neurolop",
            model="_best_model",
            dataset_name="vascular_function_128",
            labels=["activity", "growth", "symmetry"],
        )

        print(label_datasets)

    def _run_emulation_for_model(
        self,
        models: dict[str, SupervisedRegressor],
        labels: list[str],
        X: tuple[pd.DataFrame, pd.DataFrame],
        y: tuple[pd.DataFrame, pd.DataFrame],
        encoder_model_result: SupervisedModelResult,
    ) -> None:
        """
        Run emulation for a single encoder model.
        
        Parameters
        ----------
        models : Dict[str, SupervisedRegressor]
            Dictionary of regression models
        labels : List[str]
            List of target labels
        X : Tuple[pd.DataFrame, pd.DataFrame]
            Training and validation features
        y : Tuple[pd.DataFrame, pd.DataFrame]
            Training and validation targets
        encoder_model_result : SupervisedModelResult
            Result object to store emulation results
        """
        X_train, X_val = X
        y_train, y_val = y

        for label in labels:
            self._log(f"Target - {label}")
            label_result = encoder_model_result.add_label_result(label)

            for model_type, model in models.items():
                # Create a fresh model instance for each label
                regressor = SupervisedRegressor(
                    model_type=model_type, 
                    logger=self.logger,
                    eval_metric="r2"  # Use R² score for regression evaluation
                )
                
                # Find best hyperparameters
                best_params = regressor.grid_search(X_train, y_train[label])
                
                # Normalize data for better model performance
                X_train_norm, y_train_norm, X_val_norm, y_val_norm = self._normalize_data(
                    X_train, y_train[label], X_val, y_val[label]
                )
                
                # Train model with best parameters on normalized data
                # (Note: grid_search already updates the model with best params)
                regressor.fit(X_train_norm, y_train_norm)
                
                # Evaluate model performance
                r2_score = regressor.evaluate(X_val_norm, y_val_norm, metric="r2")
                
                # Store results
                label_result.add_model_result(model_type, best_params, r2_score)

    def _normalize_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Normalize features and targets for better model performance.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation targets
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
            Normalized training and validation data
        """
        # Normalize features
        X_scaler = StandardScaler()
        X_train_norm = pd.DataFrame(
            X_scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_norm = pd.DataFrame(
            X_scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        # Normalize target
        y_scaler = StandardScaler()
        y_train_norm = pd.Series(
            y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(),
            index=y_train.index,
            name=y_train.name
        )
        y_val_norm = pd.Series(
            y_scaler.transform(y_val.values.reshape(-1, 1)).flatten(),
            index=y_val.index,
            name=y_val.name
        )
        
        return X_train_norm, y_train_norm, X_val_norm, y_val_norm

    def _get_encoder_models(self, conf_name: str) -> dict[str, list[str]]:
        """
        Get list of encoder model folder names.
        
        Parameters
        ----------
        conf_name : str
            Configuration name
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping experiment names to lists of model names
        """
        encoder_models: dict[str, list[str]] = {}
        for experiment in os.listdir(f"results/{conf_name}"):
            encoder_models[experiment] = []
            for model in os.listdir(f"results/{conf_name}/{experiment}"):
                if self._is_model_folder(model):
                    encoder_models[experiment].append(model)
            encoder_models[experiment].sort()
        return encoder_models

    def _get_encoder_datasets(self, conf_name: str) -> dict[str, Optional[list[str]]]:
        """
        Get list of dataset folder names for each experiment.
        
        Parameters
        ----------
        conf_name : str
            Configuration name
            
        Returns
        -------
        Dict[str, Optional[List[str]]]
            Dictionary mapping dataset names to their labels (if any)
        """
        datasets: dict[str, Optional[list[str]]] = {}

        for experiment in os.listdir(f"results/{conf_name}"):
            for _ in os.listdir(f"results/{conf_name}/{experiment}"):
                model_dir = f"results/{conf_name}/{experiment}/_best_model"
                for dataset in os.listdir(model_dir):
                    datasets[dataset] = None

        datasets = dict(sorted(datasets.items(), key=lambda x: x[0]))
        return datasets

    def _initialize_models(self, emulator_models: list[str]) -> dict[str, SupervisedRegressor]:
        """
        Initialize emulator models.
        
        Parameters
        ----------
        emulator_models : List[str]
            List of model types to initialize
            
        Returns
        -------
        Dict[str, SupervisedRegressor]
            Dictionary mapping model types to initialized models
        """
        models = {}
        for model_type in emulator_models:
            models[model_type] = SupervisedRegressor(
                model_type=model_type, 
                logger=self.logger,
                eval_metric="r2"
            )
        return models

    def _is_model_folder(self, folder_name: str) -> bool:
        """
        Check if folder is a model folder.
        
        Parameters
        ----------
        folder_name : str
            Name of folder to check
            
        Returns
        -------
        bool
            True if folder is a model folder, False otherwise
        """
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
        """
        Log a message using the provided logger.
        
        Parameters
        ----------
        msg : str
            Message to log
        level : str
            Log level ("info" or "warning")
        """
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)