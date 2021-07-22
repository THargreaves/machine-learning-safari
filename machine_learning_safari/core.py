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

class UnsupervisedModel(ABC):

    def __init__(self):
        self.fitted = False

    def fit(self, X):
        self._fit(X)
        self.fitted = True

    @abstractmethod
    def _fit(self, X):
        pass

    def transform(self, X):
        if not self.fitted:
            # Todo: define custom error class
            raise Error("Must fit model before transforming")
        return self._transform(X)

    @abstractmethod
    def _transform(self, X):
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


class KMeans(UnsupervisedModel):

    def __init__(self, k=2, max_iter=100, tol=1e-4, seed=None):
        # Todo: multiple reps
        # Validation
        if not isinstance(k, int) and k > 0:
            raise ValueError("`k` must be a positive integer")
        if not isinstance(max_iter, int) and max_iter > 0:
            raise ValueError("`max_iter` must be a positive integer")
        if not isinstance(tol, (int, float)) and tol > 0:
            raise ValueError("`tol` must be a positive real number")
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.means = None

    def _fit(self, X):
        np.random.seed(self.seed )
        means = X[np.random.choice(X.shape[0], self.k)]
        for __ in range(self.max_iter):
            # Compute closest means
            distances = np.sqrt(((X - means[:, np.newaxis]) ** 2).sum(axis=2))
            closest = np.argmin(distances, axis=0)
            # Update means
            new_means = np.array([X[closest==k].mean(axis=0)
                                 for k in range(means.shape[0])])
            if np.all(new_means - means < self.tol):
                break
            means = new_means
        else:
            # Todo: add convergence error
            raise Error("Failed to converge")
        self.means = means

    def _transform(self, X):
        # Todo: repetitive code; should modularise
        distances = np.sqrt(((X - self.means[:, np.newaxis]) ** 2).sum(axis=2))
        closest = np.argmin(distances, axis=0)
        return closest

    def inspect(self):
        pass
