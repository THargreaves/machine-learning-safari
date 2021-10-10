"""Tests for the core package functionality."""

import numpy as np
import pytest

from mlsafari import NullModel, KMeans
from mlsafari.exceptions import NotFittedError


def test_apply_before_fit() -> None:
    """It raises an error when attempting to apply a model before fitting."""
    X = np.empty((1, 1))
    # Supervised model
    with pytest.raises(NotFittedError):
        mod1 = NullModel(objective='regression')
        mod1.apply(X)
    # Unsupervised model
    with pytest.raises(NotFittedError):
        mod2 = KMeans()
        mod2.apply(X)
