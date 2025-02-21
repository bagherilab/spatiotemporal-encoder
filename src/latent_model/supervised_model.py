from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.base import clone

from simulation_encoder.logger import Logger


class SupervisedModel:
    def __init__(
        self,
        model_type: str = "linear_regression",
        params: Optional[dict[str, Any]] = None,
        logger: Optional[Logger] = None,
    ):
        self.model_type = model_type
        self.logger = logger

        default_param_grids = {
            "linear_regression": {
                "model": LinearRegression,
                "param_grid": {
                    "fit_intercept": [True, False],
                },
            },
            "random_forest": {
                "model": RandomForestRegressor,
                "param_grid": {
                    "n_estimators": [10, 25, 50],
                    "max_depth": [10, 20],
                    "min_samples_split": [2],
                },
            },
            "svm": {
                "model": SVR,
                "param_grid": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "poly"],
                    "degree": [2, 3, 5],
                },
            },
            "mlp": {
                "model": MLPRegressor,
                "param_grid": {
                    "hidden_layer_sizes": [(10, 10), (10, 20), (25,), (25, 25)],
                    "activation": ["relu", "tanh"],
                    "solver": ["adam"],
                    "alpha": [0.0001, 0.001],
                },
            },
        }

        if model_type not in default_param_grids:
            raise ValueError(f"Invalid model type: {model_type}")

        model_class = default_param_grids[model_type]["model"]
        self.model = model_class(**params) if params else model_class()

        self.param_grid = default_param_grids[model_type]["param_grid"]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        return self.model.score(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def grid_search(self, X: pd.DataFrame, y: pd.DataFrame) -> Optional[dict]:
        """Perform a grid search to find the best hyperparameters."""
        param_grid = ParameterGrid(self.param_grid)
        best_score = -np.inf
        best_params: Optional[dict] = None
        n_models = len(param_grid)

        for i, params in enumerate(param_grid):
            self._log(f"{i+1}/{n_models} - {self.model_type}")
            model = clone(self.model).set_params(**params)
            scores = []
            for train_idx, val_idx in KFold(n_splits=5).split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model.fit(X_train, y_train)
                scores.append(model.score(X_val, y_val))
            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        return best_params

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)
