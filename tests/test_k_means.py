"""Tests for the KMeans class."""
import numpy as np
import pytest

from mlsafari import KMeans
from mlsafari.exceptions import ConvergenceError


def test_validation() -> None:
    """It properly validates initialisation arguments."""
    # k must be a postive integer
    with pytest.raises(ValueError):
        KMeans(k=3.14)
    with pytest.raises(ValueError):
        KMeans(k=0)
    # max_iter must be a postive integer
    with pytest.raises(ValueError):
        KMeans(max_iter='spam')
    with pytest.raises(ValueError):
        KMeans(max_iter=0)
    # tol must be a postive real number
    with pytest.raises(ValueError):
        KMeans(max_iter='spam')
    with pytest.raises(ValueError):
        KMeans(max_iter=0.0)


def test_k_means() -> None:
    """It outputs the correct centroids for two circular clusters."""
    # Data
    np.random.seed(1729)
    X_train = np.concatenate([
        np.random.multivariate_normal(np.array([-2, 0]), np.eye(2), size=100),
        np.random.multivariate_normal(np.array([2, 0]), np.eye(2), size=100)
    ])
    X_test = np.array([
        [-1, 0],
        [1, 0]
    ])
    # Model
    mod = KMeans()
    mod.fit(X_train)
    # Testing
    y_pred = mod.apply(X_test)
    print(y_pred)
    assert np.array_equal(np.sort(y_pred), np.array([0, 1]))


def test_convergence() -> None:
    """It returns error if not converged in time."""
    with pytest.raises(ConvergenceError):
        np.random.seed(1729)
        X_train = np.expand_dims(np.random.normal(size=100), -1)
        mod = KMeans(max_iter=1)
        mod.fit(X_train)
