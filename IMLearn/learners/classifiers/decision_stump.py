from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        curr_loss = np.inf
        for sign in [-1, 1]:
            for ind in range(X.shape[1]):
                thresh, loss = self._find_threshold(X[:, ind], y, sign)
                if loss < curr_loss:
                    curr_loss = loss
                    self.sign_ = sign
                    self.threshold_ = thresh
                    self.j_ = ind

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        res = np.ones(X.shape[0]) * self.sign_
        res[X[:, self.j_] < self.threshold_] = -self.sign_
        return res

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        ind_sort = np.argsort(values)  # sort values
        values = values[ind_sort]
        labels = labels[ind_sort]
        # define possible thresholds
        thresh = np.concatenate([[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        # define loss of -inf threshold (labels also includes weights)
        curr_loss = np.sum(np.abs(labels)[np.sign(labels) == sign])
        # calculate the loss of all thresholds using cumulative sum
        # each loss is related to the loss of the previous iteration
        losses = np.append(curr_loss, curr_loss - np.cumsum(labels * sign))
        ind_min = np.argmin(losses)
        return thresh[ind_min], losses[ind_min]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return np.count_nonzero(self.predict(X) != np.sign(y))
