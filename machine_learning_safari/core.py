"""Core machine learning algorithms and models."""

from abc import ABC, abstractmethod

import numpy as np
import scipy.stats as sts


class SupervisedModel(ABC):

    def __init__(self):
        self.fitted = False

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict(self, X):
        if not self.fitted:
            # Todo: define custom error class
            raise Error("Must fit model before predicting")

    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod
    def inspect(self):
        pass


class NullModel(SupervisedModel):

    def __init__(self, objective):
        # Validation
        if not objective in ('classification', 'regression'):
            raise ValueError(
                "`objective` must be one of 'classification' or 'regression'"
            )
        self.obj = objective
        self.val = None
        super(NullModel, self).__init__()

    def fit(self, X, y):
        self.val = sts.mode(y) if self.obj == 'classification' else np.mean(y)

    def predict(self, X):
        return np.full(X.shape[0], self.val)

    def inspect(self):
        raise NotImplementedError
