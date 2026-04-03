from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import ParameterGrid, StratifiedKFold, KFold
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class SupervisedModel:
    """
    Base class for supervised learning models with standardized interface for
    training, evaluation, prediction, and hyperparameter optimization.
    """

    def __init__(
        self,
        model_type: str,
        model_class: type[BaseEstimator],
        param_grid: dict[str, list[Any]],
        params: Optional[dict[str, Any]] = None,
        logger: Optional[Any] = None,
        eval_metric: str = "default",
    ):
        self.model_type = model_type
        self.logger = logger
        self.eval_metric = eval_metric
        self.param_grid = param_grid

        self.model = model_class(**params) if params else model_class()

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        """
        Fit the model to the training data.
        """
        self._log(f"Fitting {self.model_type} model")
        self.model.fit(X, y)
        self._log(f"Finished fitting {self.model_type} model")

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        return self.model.predict(X)

    def grid_search(
        self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray, cv_folds: int = 5
    ) -> Optional[dict]:
        """
        Perform a grid search to find the best hyperparameters.
        """
        param_grid = ParameterGrid(self.param_grid)
        best_score = -np.inf
        best_params: Optional[dict] = None
        n_models = len(param_grid)

        if hasattr(self, "is_classification") and self.is_classification:
            kf = StratifiedKFold(n_splits=cv_folds)
        else:
            kf = KFold(n_splits=cv_folds)

        for i, params in enumerate(param_grid):
            self._log(f"{i+1}/{n_models} - {self.model_type} with params: {params}")

            if hasattr(
                self, "_is_invalid_param_combination"
            ) and self._is_invalid_param_combination(params):
                self._log(f"Skipping invalid parameter combination: {params}", level="warning")
                continue

            try:
                model = clone(self.model).set_params(**params)
                scores = []

                # Split data differently based on task type
                if hasattr(self, "is_classification") and self.is_classification:
                    cv_splits = kf.split(X, y)
                else:
                    cv_splits = kf.split(X)

                for train_idx, val_idx in cv_splits:
                    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    else:
                        X_train, X_val = X[train_idx], X[val_idx]

                    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    else:
                        y_train, y_val = y[train_idx], y[val_idx]

                    try:
                        model.fit(X_train, y_train)
                        score = self._calculate_score(model, X_val, y_val)
                        scores.append(score)
                    except Exception as e:
                        self._log(f"Error in fold: {str(e)}", level="warning")
                        continue

                if scores:
                    mean_score = np.mean(scores)
                    self._log(f"Mean {self.eval_metric} score: {mean_score:.4f}")

                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
                else:
                    self._log(f"No valid scores for parameters: {params}", level="warning")

            except Exception as e:
                self._log(f"Error with parameter set {params}: {str(e)}", level="warning")
                continue

        if best_params:
            self._log(f"Best params: {best_params}, score: {best_score:.4f}")
            # Update the model with best parameters
            self.model = clone(self.model).set_params(**best_params)
        else:
            self._log("No valid parameter combinations found", level="warning")

        return best_params

    def _log(self, msg: str, level: str = "info") -> None:
        """
        Log a message using the provided logger if available.

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
                self.logger.info(msg)


class SupervisedClassifier(SupervisedModel):
    """
    A wrapper class for various classification models with standardized interface for
    training, evaluation, prediction, and hyperparameter optimization.
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",
        params: Optional[dict[str, Any]] = None,
        logger: Optional[Any] = None,
        eval_metric: str = "accuracy",
    ):
        """
        Initialize a supervised classification model.

        Parameters
        ----------
        model_type : str
            Type of classification model to use. Options: "logistic_regression",
            "random_forest", "svm", "mlp"
        params : Optional[Dict[str, Any]]
            Parameters to initialize the model with, overriding defaults
        logger : Optional[Any]
            Logger object for tracking training progress
        eval_metric : str
            Evaluation metric for model selection. Options: "accuracy", "f1",
            "precision", "recall"
        """
        self.is_classification = True

        # Define default parameter grids for classification models
        default_param_grids = {
            "logistic_regression": {
                "model": LogisticRegression,
                "param_grid": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2", "elasticnet"],
                    "solver": ["liblinear", "saga"],
                    "max_iter": [1000],
                },
            },
            "random_forest": {
                "model": RandomForestClassifier,
                "param_grid": {
                    "n_estimators": [10, 25],
                    "max_depth": [5, 10, 20],
                    "min_samples_split": [5, 10],
                },
            },
            "svm": {
                "model": SVC,
                "param_grid": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "poly"],
                    "degree": [2, 3],
                    "max_iter": [1000],
                },
            },
            "mlp": {
                "model": MLPClassifier,
                "param_grid": {
                    "hidden_layer_sizes": [(50,), (100,)],
                    "activation": ["relu"],
                    "solver": ["adam"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate": ["constant", "adaptive"],
                    "max_iter": [500],
                },
            },
        }

        if model_type not in default_param_grids:
            raise ValueError(
                f"Invalid model type: {model_type}. Available types: {list(default_param_grids.keys())}"
            )

        model_config = default_param_grids[model_type]
        super().__init__(
            model_type=model_type,
            model_class=model_config["model"],
            param_grid=model_config["param_grid"],
            params=params,
            logger=logger,
            eval_metric=eval_metric,
        )

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        metric: Optional[str] = None,
    ) -> float:
        """
        Evaluate the model on the given data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix
        y : Union[pd.Series, np.ndarray]
            True class labels
        metric : Optional[str]
            Metric to use for evaluation. If None, uses the metric specified during initialization.

        Returns
        -------
        float
            Score based on the specified metric
        """
        metric = metric or self.eval_metric
        y_pred = self.predict(X)

        if metric == "accuracy":
            return accuracy_score(y, y_pred)
        elif metric == "f1":
            return f1_score(y, y_pred, average="weighted")
        elif metric == "precision":
            return precision_score(y, y_pred, average="weighted")
        elif metric == "recall":
            return recall_score(y, y_pred, average="weighted")
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Predict probability estimates for samples.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix

        Returns
        -------
        np.ndarray
            Probability estimates
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(
                f"Model {self.model_type} does not support probability predictions"
            )

    def _is_invalid_param_combination(self, params: dict) -> bool:
        """
        Check if the parameter combination is invalid.

        Parameters
        ----------
        params : dict
            Parameter combination to check

        Returns
        -------
        bool
            True if the parameter combination is invalid, False otherwise
        """
        if self.model_type == "logistic_regression":
            if params.get("penalty") == "elasticnet" and params.get("solver") != "saga":
                return True

            if params.get("penalty") == "l1" and params.get("solver") not in ["liblinear", "saga"]:
                return True

        return False

    def _calculate_score(self, model, X_val, y_val):
        """
        Calculate score based on the evaluation metric.

        Parameters
        ----------
        model : BaseEstimator
            Trained model
        X_val : Union[pd.DataFrame, np.ndarray]
            Validation features
        y_val : Union[pd.Series, np.ndarray]
            Validation targets

        Returns
        -------
        float
            Score based on the evaluation metric
        """
        y_pred = model.predict(X_val)

        if self.eval_metric == "accuracy":
            return accuracy_score(y_val, y_pred)
        elif self.eval_metric == "f1":
            return f1_score(y_val, y_pred, average="weighted")
        elif self.eval_metric == "precision":
            return precision_score(y_val, y_pred, average="weighted")
        elif self.eval_metric == "recall":
            return recall_score(y_val, y_pred, average="weighted")
        else:
            return model.score(X_val, y_val)


class SupervisedRegressor(SupervisedModel):
    """
    A wrapper class for various regression models with standardized interface for
    training, evaluation, prediction, and hyperparameter optimization.
    """

    def __init__(
        self,
        model_type: str = "linear_regression",
        params: Optional[dict[str, Any]] = None,
        logger: Optional[Any] = None,
        eval_metric: str = "r2",
    ):
        """
        Initialize a supervised regression model.

        Parameters
        ----------
        model_type : str
            Type of regression model to use. Options: "linear_regression",
            "ridge", "lasso", "elastic_net", "random_forest", "svr", "mlp"
        params : Optional[Dict[str, Any]]
            Parameters to initialize the model with, overriding defaults
        logger : Optional[Any]
            Logger object for tracking training progress
        eval_metric : str
            Evaluation metric for model selection. Options: "r2", "mse", "rmse", "mae"
        """
        self.is_classification = False

        # Define default parameter grids for regression models
        default_param_grids = {
            "linear_regression": {
                "model": LinearRegression,
                "param_grid": {
                    "fit_intercept": [True, False],
                    **(
                        {"normalize": [True, False]}
                        if hasattr(LinearRegression, "normalize")
                        else {}
                    ),
                },
            },
            "elastic_net": {
                "model": ElasticNet,
                "param_grid": {
                    "alpha": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.7, 0.9],
                    "selection": ["cyclic", "random"],
                    "max_iter": [1000],
                },
            },
            "random_forest": {
                "model": RandomForestRegressor,
                "param_grid": {
                    "n_estimators": [10, 50],
                    "max_depth": [10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [2, 4],
                },
            },
            "svr": {
                "model": SVR,
                "param_grid": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["poly", "rbf"],
                    "gamma": ["scale", "auto"],
                    "epsilon": [0.1, 0.2, 0.5],
                },
            },
            "mlp": {
                "model": MLPRegressor,
                "param_grid": {
                    "hidden_layer_sizes": [(50,), (100,)],
                    "activation": ["relu", "tanh"],
                    "solver": ["adam"],
                    "alpha": [0.0001, 0.01],
                    "learning_rate": ["constant", "adaptive"],
                    "max_iter": [500],
                },
            },
        }

        if model_type not in default_param_grids:
            raise ValueError(
                f"Invalid model type: {model_type}. Available types: {list(default_param_grids.keys())}"
            )

        model_config = default_param_grids[model_type]
        super().__init__(
            model_type=model_type,
            model_class=model_config["model"],
            param_grid=model_config["param_grid"],
            params=params,
            logger=logger,
            eval_metric=eval_metric,
        )

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        metric: Optional[str] = None,
    ) -> float:
        """
        Evaluate the model on the given data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix
        y : Union[pd.Series, np.ndarray]
            True values
        metric : Optional[str]
            Metric to use for evaluation. If None, uses the metric specified during initialization.

        Returns
        -------
        float
            Score based on the specified metric
        """
        metric = metric or self.eval_metric
        y_pred = self.predict(X)

        if metric == "r2":
            return r2_score(y, y_pred)
        elif metric == "mse":
            return -mean_squared_error(y, y_pred)
        elif metric == "rmse":
            return -np.sqrt(mean_squared_error(y, y_pred))
        elif metric == "mae":
            return -mean_absolute_error(y, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _calculate_score(self, model, X_val, y_val):
        """
        Calculate score based on the evaluation metric.

        Parameters
        ----------
        model : BaseEstimator
            Trained model
        X_val : Union[pd.DataFrame, np.ndarray]
            Validation features
        y_val : Union[pd.Series, np.ndarray]
            Validation targets

        Returns
        -------
        float
            Score based on the evaluation metric
        """
        y_pred = model.predict(X_val)

        if self.eval_metric == "r2":
            return r2_score(y_val, y_pred)
        elif self.eval_metric == "mse":
            return -mean_squared_error(y_val, y_pred)  # Negative because higher is better
        elif self.eval_metric == "rmse":
            return -np.sqrt(mean_squared_error(y_val, y_pred))  # Negative because higher is better
        elif self.eval_metric == "mae":
            return -mean_absolute_error(y_val, y_pred)  # Negative because higher is better
        else:
            return model.score(X_val, y_val)
