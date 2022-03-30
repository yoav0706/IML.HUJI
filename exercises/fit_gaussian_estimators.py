from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    X1 = np.random.normal(mu, sigma, 1000)
    uni1 = UnivariateGaussian()
    uni1.fit(X1)
    print(uni1.mu_, uni1.var_)

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mu = []
    uni = UnivariateGaussian()
    ms = np.arange(10, 1000, 10)
    for i in ms:
        X = np.random.normal(mu, sigma, size=int(i))
        uni.fit(X)
        estimated_mu.append(uni.mu_)

    go.Figure([go.Scatter(x=ms, y=abs(estimated_mu - uni.mu_), mode='markers+lines', name=r'$\widehat\sigma^2$')],
              layout=go.Layout(title=r"$\text{Error of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="number of samples",
                               yaxis_title="|estimated-expected|",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    go.Figure([go.Scatter(x=X1, y=uni1.pdf(X1), mode='markers', name=r'$\widehat\sigma^2$')],
              layout=go.Layout(title=r"empirical pdf",
                               xaxis_title="value",
                               yaxis_title="estimated pdf",
                               height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    X = np.random.multivariate_normal(mean, cov, 1000)
    multi = MultivariateGaussian()
    multi.fit(X)
    print("Estimated mean:\n", multi.mu_)
    print("Estimated cov:\n", multi.cov_)

    # Question 5 - Likelihood evaluation
    xs = np.linspace(-10, 10, 200)
    ys = np.linspace(-10, 10, 200)
    res = np.zeros((len(xs), len(ys)))
    for x, f1 in enumerate(xs):
        for y, f3 in enumerate(ys):
            mu = [f1, 0, f3, 0]
            res[x, y] = MultivariateGaussian.log_likelihood(mu, cov, X)
    go.Figure([go.Heatmap(x=xs, y=ys, z=res)],
              layout=go.Layout(title=r"$\text{Heatmap of f1 and f3 log likelihood}$",
                               xaxis_title="f1",
                               yaxis_title="f2",
                               height=300)).show()

    # Question 6 - Maximum likelihood
    ind = np.unravel_index(np.argmax(res, axis=None), res.shape)
    print(xs[ind[0]], ys[ind[1]])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
