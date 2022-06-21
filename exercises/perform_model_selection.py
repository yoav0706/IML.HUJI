from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    max_val = 2
    min_val = -1.2
    x = np.random.rand(n_samples) * (max_val - min_val) + min_val
    y_true = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y_noisy = y_true + np.random.randn(n_samples) * noise
    x_train, y_train, x_test, y_test = split_train_test(pd.DataFrame(x), pd.Series(y_noisy, name="y"), 2 / 3)
    x_train = x_train.to_numpy().squeeze()
    x_test = x_test.to_numpy().squeeze()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    fig = go.Figure([
        go.Scatter(
            name='Clean data',
            x=x,
            y=y_true,
            mode='markers'
        ),
        go.Scatter(
            name='Train data',
            x=x_train,
            y=y_train,
            mode='markers'
        ),
        go.Scatter(
            name='Test data',
            x=x_test,
            y=y_test,
            mode='markers'
        )
    ])
    fig.show()
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    cross_val_train = []
    cross_val_test = []
    for k in range(11):
        pol = PolynomialFitting(k)
        train_loss, test_loss = cross_validate(pol, x_train, y_train, mean_square_error)
        cross_val_train.append(train_loss)
        cross_val_test.append(test_loss)
    fig = go.Figure([
        go.Scatter(
            name='cross_val_train',
            y=cross_val_train,
            mode='lines+markers'
        ),
        go.Scatter(
            name='cross_val_test',
            y=cross_val_test,
            mode='lines+markers'
        )
    ])
    fig.show()
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(cross_val_test).item()
    pol = PolynomialFitting(best_k)
    pol.fit(x_train, y_train)
    y_approx = pol.predict(x_test)
    print(f"best k is: {best_k} mse: {mean_square_error(y_test, y_approx)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    x_train = X[:n_samples, :]
    y_train = y[:n_samples]
    x_test = X[n_samples:, :]
    y_test = y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_range = np.linspace(0.001, 2, num=n_evaluations)
    ridge_cross_val_train = []
    ridge_cross_val_test = []
    lasso_cross_val_train = []
    lasso_cross_val_test = []
    for lam in lam_range:
        ridge = RidgeRegression(lam, True)
        train_loss, test_loss = cross_validate(ridge, x_train, y_train, mean_square_error)
        ridge_cross_val_train.append(train_loss)
        ridge_cross_val_test.append(test_loss)
        lasso = Lasso(alpha=lam)
        train_loss, test_loss = cross_validate(lasso, x_train, y_train, mean_square_error)
        lasso_cross_val_train.append(train_loss)
        lasso_cross_val_test.append(test_loss)
    fig = go.Figure([
        go.Scatter(
            name='ridge_cross_val_train',
            x=lam_range,
            y=ridge_cross_val_train,
            mode='lines'
        ),
        go.Scatter(
            name='ridge_cross_val_test',
            x=lam_range,
            y=ridge_cross_val_test,
            mode='lines'
        ),
        go.Scatter(
            name='lasso_cross_val_train',
            x=lam_range,
            y=lasso_cross_val_train,
            mode='lines'
        ),
        go.Scatter(
            name='lasso_cross_val_test',
            x=lam_range,
            y=lasso_cross_val_test,
            mode='lines'
        )
    ])
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = lam_range[np.argmin(ridge_cross_val_test)]
    best_lam_lasso = lam_range[np.argmin(lasso_cross_val_test)]
    lst = LinearRegression(include_intercept=True)
    lst.fit(x_train, y_train)
    lst_loss = mean_square_error(y_test,lst.predict(x_test))
    ridge = RidgeRegression(lam=best_lam_ridge, include_intercept=True)
    ridge.fit(x_train, y_train)
    ridge_loss = mean_square_error(y_test, ridge.predict(x_test))
    lasso = Lasso(alpha=best_lam_lasso)
    lasso.fit(x_train, y_train)
    lasso_loss = mean_square_error(y_test, lasso.predict(x_test))
    print(best_lam_ridge,best_lam_lasso)
    print(f"lst_loss: {lst_loss} ridge_loss: {ridge_loss} lasso_loss: {lasso_loss}")


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10.)
    select_regularization_parameter()
