"""Core machine learning algorithms and models."""

from abc import ABC, abstractmethod

import numpy as np

from machine_learning_safari.exceptions import NotFittedError


class SupervisedModel(ABC):
    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        self._fit(X, y)
        self.fitted = True
        return self

    @abstractmethod
    def _fit(self, X, y):
        pass

    def predict(self, X):
        if not self.fitted:
            # Todo: define custom error class
            raise NotFittedError("Must fit model before predicting")
        return self._predict(X)

    @abstractmethod
    def _predict(self, X):
        pass

    def inspect(self):
        self._inspect()

    @abstractmethod
    def _inspect(self):
        pass


class NullModel(SupervisedModel):
    def __init__(self, objective):
        # Validation
        if objective not in ('classification', 'regression'):
            raise ValueError(
                "`objective` must be one of 'classification' or 'regression'"
            )
        self.obj = objective
        self.val = None
        super(NullModel, self).__init__()

    def _fit(self, X, y):
        self.val = np.mean(y) if self.obj == 'regression' else self._mode(y)

    def _predict(self, X):
        return np.full(X.shape[0], self.val)

    @staticmethod
    def _mode(x):
        values, counts = np.unique(x, return_counts=True)
        return values[np.argmax(counts)]

    def _inspect(self):
        raise NotImplementedError
