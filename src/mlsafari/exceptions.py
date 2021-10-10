"""Package-specific exceptions."""


class NotFittedError(Exception):
    """Exception raised when predicting using an unfitted estimator."""

    pass


class ConvergenceError(Exception):
    """Exception raised when a model fails to converge."""

    pass
