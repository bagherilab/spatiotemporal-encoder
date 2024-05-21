import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, PredefinedSplit

class Emulator:
    def __init__(self, model_type='linear_regression'):
        self.model_type = model_type
        if model_type == 'linear_regression':
            self.model = LinearRegression()
            self.param_grid = {
                'fit_intercept': [True, False],
            }
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor()
            self.param_grid = {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20, 30]
            }
        elif model_type == 'svm':
            self.model = SVR()
            self.param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def evaluate(self, X, y) -> float:
        return self.model.score(X, y)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def grid_search(self, X, y) -> dict:
        """Perform a grid search to find the best hyperparameters."""
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=5, scoring='r2')
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_
