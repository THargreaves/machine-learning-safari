"""Tests for the NullModel class."""
import numpy as np
import pytest

from mlsafari import NullModel


@pytest.mark.parametrize(
    'objective,expectation', [('regression', 3), ('classification', 2)]
)
def test_null_model(objective: str, expectation: float) -> None:
    """
    It outputs the mean/modal training value for all training predictors.

    Args:
        objective: The objective of the model (classification or regression).
        expectation: The expected prediction of the model.
    """
    # Data
    X_train = np.empty((5, 2))
    y_train = np.array([2, 2, 3, 4, 4])
    X_test = np.empty((2, 2))
    # Model
    mod = NullModel(objective=objective)
    mod.fit(X_train, y_train)
    y_pred = mod.apply(X_test)
    # Testing
    y_test = np.full(2, expectation)
    assert np.allclose(y_pred, y_test)
