"""Author: Johannes Krämer"""

import logging
import optuna
import sklearn.metrics as skm
from sklearn.base import BaseEstimator
from typing import Any, Dict
import sklearn.model_selection as skms


def tune_model(
    X: Any, y: Any, model: BaseEstimator, param_grid: Dict[str, Any]
) -> BaseEstimator:
    """
    Tunes the given model using GridSearchCV with the provided parameter grid.

    Parameters:
    X (Any): Feature matrix.
    Y (Any): Labels.
    model (BaseEstimator): The machine learning model to be tuned.
    param_grid (Dict[str, Any]): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.

    Returns:
    BaseEstimator: The best estimator found by GridSearchCV.
    """
    logging.info("%s - Tuning", model)
    grid_search = skms.GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X, y)
    logging.info("%s - Best params: %s", model, grid_search.best_params_)
    logging.info("%s - Best score: %s", model, grid_search.best_score_)
    return grid_search.best_estimator_


def tune_model_optuna(
    X: Any,
    y: Any,
    model_class: BaseEstimator,
    param_grid: Dict[str, Any],
    n_trials: int = 100,
) -> BaseEstimator:
    """
    Tunes the given model using Optuna with the provided parameter grid.

    Parameters:
    X (Any): Feature matrix.
    Y (Any): Labels.
    model_class (BaseEstimator): The machine learning model to be tuned.
    param_grid (Dict[str, Any]): Dictionary with parameter names (str) as keys and lists of parameter settings to try as values.
    n_trials (int): Number of trials for the optimization. Default is 100.

    Returns:
    BaseEstimator: The best estimator found by Optuna.
    """

    def objective(trial):
        params = {}
        for key, value in param_grid.items():
            if isinstance(value, list):
                params[key] = trial.suggest_categorical(key, value)
            else:
                params[key] = trial.suggest_float(key, *value)
        trial_model = model_class(**params)
        X_train, X_test, y_train, y_test = skms.train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        trial_model.fit(X_train, y_train)
        y_pred = trial_model.predict(X_test)
        return skm.accuracy_score(y_test, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    best_params = study.best_params
    best_model = model_class(**best_params)
    best_model.fit(X, y)
    logging.info("%s - Best params: %s", model_class, best_params)
    logging.info("%s - Best score: %s", model_class, study.best_value)
    return best_model
