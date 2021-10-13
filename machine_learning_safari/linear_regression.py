import numpy as np

from mlsafari.core import _SupervisedModel

class LinearRegression(_SupervisedModel):
    """
    Ordinary Least Squares Regression model

    Fits a linear model such that coefficients minimize the
    residual sum of squares between the observed target and
    the predicted target.
     """
    def __init__(self, intercept=True):
        """
        Initialise the model.

        Args:
            intercept: Whether or not the model is fitted with an intercept.
        """
        # Input validation
        self._validate_inputs(intercept)
        # Attributes    
        self.intercept = intercept 
        self.coef = None
        self.resid = None
        super(LinearRegression, self).__init__()

    @staticmethod
    def _validate_inputs(intercept):
        if not isinstance(intercept, bool):
            raise ValueError("`intercept` must be boolean") 

    def _fit(self, X,y):
        """
        Fits model coefficients using OLS estimation.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples, 1)

        Returns:
             self
        """
        # Adding column of ones for models with fitted intercept
        if self.intercept:
            X = np.vstack([X, np.ones(len(X))]).T
        # Computing coefficients
        self.coef = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.resid = y - np.dot(X, self.coef)
        return self

    def _apply(self, X):
        preds = np.dot(X, self.coef)
        return preds

    def _inspect(self):
        raise NotImplementedError