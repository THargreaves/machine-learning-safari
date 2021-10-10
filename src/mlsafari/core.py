"""Core machine learning algorithms and models."""

from abc import ABC, abstractmethod

import numpy as np

from mlsafari.exceptions import NotFittedError


class _SupervisedModel(ABC):
    """A abstract model used for supervised learning."""

    def __init__(self):
        """Initialise the model and mark as unfitted."""
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model.

        Args:
            X: Array of training set predictors
            y: Array of training set responses

        Returns:
            self: The fitted model
        """
        # Fit using child method and mark as fitted
        self._fit(X, y)
        self.fitted = True
        return self

    @abstractmethod
    def _fit(self, X, y):
        """Fit the model using internal method to be overridden by child."""
        pass  #pragma: no cover

    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the model.

        Args:
            X: Array of test set predictors

        Returns:
            y: The result of applying the model.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if not self.fitted:
            raise NotFittedError("Must fit model before predicting")
        return self._apply(X)

    @abstractmethod
    def _apply(self, X):
        """Apply the model using internal method to be overridden by child."""
        pass  #pragma: no cover

    def inspect(self):
        self._inspect()

    @abstractmethod
    def _inspect(self):
        pass  #pragma: no cover


class _UnsupervisedModel(ABC):
    """A abstract model used for unsupervised learning."""

    def __init__(self):
        """Initialise the model and mark as unfitted."""
        self.fitted = False

    def fit(self, X: np.ndarray):
        """
        Fit the model.

        Args:
            X: Array of training set features

        Returns:
            self: The fitted model
        """
        # Fit using child method and mark as fitted
        self._fit(X)
        self.fitted = True
        return self

    @abstractmethod
    def _fit(self, X):
        """Fit the model using internal method to be overridden by child."""
        pass  #pragma: no cover

    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the model.

        Args:
            X: Array of test set features

        Returns:
            y: The result of applying the model.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if not self.fitted:
            raise NotFittedError("Must fit model before predicting")
        return self._apply(X)

    @abstractmethod
    def _apply(self, X):
        """Apply the model using internal method to be overridden by child."""
        pass  #pragma: no cover

    def inspect(self):
        self._inspect()

    @abstractmethod
    def _inspect(self):
        pass  #pragma: no cover


class NullModel(_SupervisedModel):
    """
    A null model.

    The null model is the optimum model that can be trained by only considering
    the response variable in the training set (ignoring the predictors). This
    should be used as a benchmark performance, with any model that is better
    than chance, obtaining a better test score.

    For classification, the null model will predict the most common class in
    the training data for every prediction (regardless of the predictor
    values). For regression, the mean is used.
    """

    def __init__(self, objective):
        """
        Initialise the model.

        Args:
            objective: The model objecive, either 'classification' or
                'regression'.
        """
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

    def _apply(self, X):
        return np.full(X.shape[0], self.val)

    @staticmethod
    def _mode(x):
        values, counts = np.unique(x, return_counts=True)
        return values[np.argmax(counts)]

    def _inspect(self):
        raise NotImplementedError
