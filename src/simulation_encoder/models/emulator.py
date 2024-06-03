from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


class Emulator:
    def __init__(
        self,
        model_type: str = "linear_regression",
        params: Optional[dict[str, Any]] = None,
    ):
        self.model_type = model_type

        # Define default parameter grids
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
                    "n_estimators": [10, 50, 100],
                    "max_depth": [10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "bootstrap": [True, False],
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

    def grid_search(self, X: pd.DataFrame, y: pd.DataFrame) -> dict:
        """Perform a grid search to find the best hyperparameters."""
        grid_search = GridSearchCV(
            estimator=self.model, param_grid=self.param_grid, cv=5, scoring="r2"
        )
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_
