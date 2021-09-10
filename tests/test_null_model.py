import numpy as np
import pytest

from machine_learning_safari import NullModel

@pytest.mark.parametrize('objective,expectation', [
    ('regression', 3),
    ('classification', 2)
])
def test_null_model(objective, expectation):
    # Data
    X_train = np.empty((5, 2))
    y_train = np.array([2, 2, 3, 4, 4])
    X_test = np.empty((2, 2))
    # Model
    mod = NullModel(objective=objective)
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)
    # Testing
    y_test = np.full(2, expectation)
    assert np.allclose(y_pred, y_test)
