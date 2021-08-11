from .core import NullModel
import numpy as np
from scipy import linalg
from scipy import optimize

class LinearRegression(NullModel):
    """
    Ordinary Least Squares Regression model

    Fits a linear model such that coefficients minimize the
    residual sum of squares between the observed target and
    the predicted target.

    Parameters
    ----------
    :param intercept:

    Attributes
    ----------
    :att X:

    """
    def __init__(self, intercept=True, positive=False):
        if not isinstance(intercept, (boolean, int)) and k > 0:
            raise ValueError("`k` must be a positive integer")
        self.intercept = intercept
        self.positive = positive

    def fit(self, X,y):
        """
        Fits model coefficients using OLS estimation
        :param X: Data matrix of shape (n_samples, n_features)
        :param y: Target vector of shape (n_samples, 1)
        :return: self
        """

        # Adding column of ones for models with fitted intercept
        if intercept:
            X = np.vstack([X, np.ones(len(X))]).T
        np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

        self.coef, self.residuals, self.rank, self.singular = linalg.lstsq(X, y)
        self.coef = self.residuals.T
        return self
    def predict(self, X):
        pass