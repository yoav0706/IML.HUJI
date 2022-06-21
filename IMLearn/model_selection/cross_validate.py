from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    groups = np.remainder(np.arange(X.shape[0]), cv)
    train_score = 0
    validation_score = 0
    for k in range(cv):
        x_train = X[groups != k]
        y_train = y[groups != k]
        x_val = X[groups == k]
        y_val = y[groups == k]
        estimator.fit(x_train, y_train)
        y_pred_train = estimator.predict(x_train)
        y_pred_val = estimator.predict(x_val)
        validation_score += scoring(y_val, y_pred_val)
        train_score += scoring(y_train, y_pred_train)
    train_score /= cv
    validation_score /= cv
    return train_score, validation_score
