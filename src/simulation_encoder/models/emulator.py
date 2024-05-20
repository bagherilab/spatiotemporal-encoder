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
                'normalize': [True, False]
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

    def grid_search(self, X_train, y_train, X_val, y_val) -> dict:
        """Perform a grid search to find the best hyperparameters."""
        test_fold = [-1] * len(X_train) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold=test_fold)

        X = np.concatenate((X_train, X_val))
        y = np.concatenate((y_train, y_val))
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=ps, scoring='r2')
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_
