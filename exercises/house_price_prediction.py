from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)
    data = data[(data['price'] > 0) & (data['bedrooms'] > 0) & (data['bathrooms'] > 0) & (data['sqft_living'] > 0)]
    data['status'] = 0
    data.loc[data['yr_renovated'] > 2010, 'status'] = 1
    data.loc[data['yr_built'] > 2000, 'status'] = 2
    features = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode','status']
    data.dropna()
    data = data[features]
    # data = data.drop(columns=['id', 'date', 'lat', 'long', 'yr_built', 'yr_renovated', 'zipcode'], axis=1)
    return data


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_name = str(y.name)
    for column in X.columns:
        x = X[column]
        pearson = x.cov(y) / (x.std() * y.std())
        fig = go.Figure([go.Scatter(x=x, y=y, name="column correlation", mode='markers')],
                        layout=go.Layout(title=r"pearson correlation = " + str(pearson),
                                         xaxis={"title": "feature - " + column},
                                         yaxis={"title": "response - " + y_name},
                                         height=400))
        fig.write_image(output_path + "/" + column + " + " + y_name + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, X['price'])

    # Question 3 - Split samples into training- and testing sets.
    y = X.pop('price')
    train_x, train_y, test_x, test_y = split_train_test(X, y, train_proportion=0.75)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    l_reg = LinearRegression()
    x_per = np.arange(0.1, 1, 0.01)
    mean_y = []
    var_y = []
    for percent in x_per:
        per_loss = []
        for i in range(10):
            part_train_x, part_train_y, _, _ = split_train_test(train_x, train_y, train_proportion=percent)
            l_reg.fit(part_train_x.to_numpy(), part_train_y.to_numpy())
            per_loss.append(l_reg.loss(test_x.to_numpy(), test_y.to_numpy()))
        mean_y.append(np.mean(per_loss))
        var_y.append(np.var(per_loss))
    mean_y = np.array(mean_y)
    var_y = np.array(var_y)
    fig = go.Figure([
        go.Scatter(
            name='Mean Loss',
            x=x_per,
            y=mean_y,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Mean+2*STD',
            x=x_per,
            y=mean_y + 2 * var_y,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Mean-2*STD',
            x=x_per,
            y=mean_y - 2 * var_y,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        xaxis_title='Percents of training data',
        yaxis_title='Loss over percentage',
        title='Continuous, variable value error bars'
    )
    fig.show()
