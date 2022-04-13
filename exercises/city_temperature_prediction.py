import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date'])
    data.dropna()
    data = data[
        (data['Month'] > 0) & (data['Month'] <= 12) & (data['Day'] > 0) & (data['Day'] <= 31) & (data['Temp'] > -50)]
    data['DayOfYear'] = data['Date'].dt.dayofyear
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    IS_X = X[X['Country'] == 'Israel']
    temp = IS_X.copy()
    temp["Year"] = temp["Year"].astype(str)
    fig = px.scatter(temp, x="Month", y="Temp", color="Year")
    fig.show()
    month_grouped = IS_X.groupby(['Month']).agg('std')
    fig = px.bar(x=month_grouped.index, y=month_grouped['Temp'])
    fig.update_layout(title="STD of temp per month",
                      xaxis={"title": "Month"},
                      yaxis={"title": "STD Temp"})
    fig.show()
    # Question 3 - Exploring differences between countries
    country_month = X.groupby(["Country", "Month"]).agg('mean').reset_index()
    temp = X.groupby(["Country", "Month"]).agg('std').reset_index()
    country_month['temp_std'] = temp['Temp']
    fig = px.line(country_month, x="Month", y="Temp", error_y='temp_std', color='Country')
    fig.show()
    # Question 4 - Fitting model for different values of `k`
    IS_X = X[X['Country'] == 'Israel']
    IS_y = IS_X.pop('Temp')
    train, train_y, test, test_y = split_train_test(IS_X, IS_y, train_proportion=0.75)
    deg_k = np.arange(1, 11, 1)
    loss_per_k = []
    for k in deg_k:
        polyfit = PolynomialFitting(k)
        polyfit.fit(train['DayOfYear'].to_numpy(), train_y.to_numpy())
        loss_per_k.append(polyfit.loss(test['DayOfYear'].to_numpy(), test_y.to_numpy()))
        print(k,loss_per_k[-1])
    fig = px.bar(x=deg_k, y=loss_per_k)
    fig.update_layout(title="Loss over degree of k",
                      xaxis={"title": "Degree of k"},
                      yaxis={"title": "Loss"})
    fig.show()
    # Question 5 - Evaluating fitted model on different countries
    polyfit = PolynomialFitting(6)
    polyfit.fit(train['DayOfYear'].to_numpy(), train_y.to_numpy())
    countries = []
    countries_err = []
    for country in X.Country.unique():
        if country != "Israel":
            countries.append(country)
            temp_X = X[X['Country'] == country]
            countries_err.append(polyfit.loss(temp_X['DayOfYear'].to_numpy(), temp_X['Temp'].to_numpy()))
    fig = px.bar(x=countries, y=countries_err)
    fig.update_layout(title="Loss over different countries",
                      xaxis={"title": "Countries"},
                      yaxis={"title": "Loss"})
    fig.show()
