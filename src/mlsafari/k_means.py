"""Core machine learning algorithms and models."""

import numpy as np

from mlsafari.core import _UnsupervisedModel
from mlsafari.exceptions import ConvergenceError


class KMeans(_UnsupervisedModel):
    """
    A K-Means model.

    The k-means model is used for clustering data points. It is an iterative
    algorithm which alternates between two steps:
      1. Assign each data point to the nearest cluster centroid according to
         the Euclidean distance.
      2. Update each centroid to minimise the sum of squared distances to its
         assigned data points.

    Given enough time, the centroids are guaranteed to converge to within a
    tolerance.
    """

    # TODO: Add multiple starts
    def __init__(self, k=2, max_iter=100, tol=1e-4, seed=None):
        """
        Initialise the model.

        Args:
            k: The number of clusters.
            max_iter: The maxium number of iterations to run.
            tol: The tolerance used to determine convergence. If all updated
                centroids are within a distance `tol` of their previous value
                we conisder the model to have converged.
            seed: A random seed used to ensure reproducibility.
        """
        self._validate_inputs(k, max_iter, tol)
        # Attributes
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.means = None
        super(KMeans, self).__init__()

    @staticmethod
    def _validate_inputs(k, max_iter, tol):
        if not (isinstance(k, int) and k > 0):
            raise ValueError("`k` must be a positive integer")
        if not (isinstance(max_iter, int) and max_iter > 0):
            raise ValueError("`max_iter` must be a positive integer")
        if not (isinstance(tol, (int, float)) and tol > 0):
            raise ValueError("`tol` must be a positive real number")

    def _fit(self, X):
        original_state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            # Initialise centroids by randomly chosing k data points
            means = X[np.random.choice(X.shape[0], self.k)]
            for __ in range(self.max_iter):
                # Compute closest means
                closest = self._closest_centroids(X, means)
                # Update means
                new_means = np.array([
                    X[closest == k].mean(axis=0)
                    for k in range(means.shape[0])
                ])
                # Check for convergence
                if np.all(np.abs(new_means - means) < self.tol):
                    break
                means = new_means
            else:
                raise ConvergenceError("Failed to converge")
            self.means = means
        finally:
            np.random.set_state(original_state)

    def _apply(self, X):
        return self._closest_centroids(X, self.means)

    @staticmethod
    def _closest_centroids(X, means):
        distances = np.sqrt(((X - means[:, np.newaxis]) ** 2).sum(axis=2))
        closest = np.argmin(distances, axis=0)
        return closest

    def _inspect(self):
        raise NotImplementedError
