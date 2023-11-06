from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin


class KernelFunction(Protocol):
    """Callable interface for kernel functions used by Nadaraya-Watson estimator."""

    def __call__(self, memory: NDArray, points: NDArray, bandwidth: float) -> NDArray:
        ...


def gaussian_kernel(memory: NDArray, points: NDArray, bandwidth: float) -> NDArray:
    diff = memory - points
    return np.exp(-0.5 * (diff / bandwidth) ** 2)


def nearest_neighbor_kernel(memory: NDArray, points: NDArray, bandwidth: float) -> NDArray:
    diff = memory - points
    return (np.abs(diff) <= bandwidth).astype(float)


class NadarayaWatsonEstimator(RegressorMixin, BaseEstimator):
    """
    Nadaraya-Watson estimator for regression, it is a non-parametric estimator that uses kernel functions.
    Fitting is done by storing all the training data, so it is not recommended for large datasets.
    Inference is done by computing the weighted average of the stored training data using given kernel function.
    """

    def __init__(self, kernel: KernelFunction, bandwidth: float, max_memory: int | None = None):
        """
        :param kernel: any function implementing the KernelFunction protocol
        :param bandwidth: parameter for kernel function
        :param max_memory: maximal number of samples to keep in memory, used with partial_fit
                           defaults to None, which does not truncate stored
        """
        super().__init__()

        self.kernel = kernel
        self.bandwidth = bandwidth
        self.max_memory = max_memory

        self.x: NDArray | None = None
        self.y: NDArray | None = None

    def fit(self, x: NDArray, y: NDArray) -> None:
        self.x = x
        self.y = y

    def predict(self, x: NDArray) -> NDArray:
        memory = np.expand_dims(self.x, axis=-1)
        points = np.expand_dims(x, axis=0)

        weights = self.kernel(memory, points, bandwidth=self.bandwidth)
        return (weights * np.expand_dims(self.y, axis=-1)).sum(axis=0) / weights.sum(axis=0)

    def fit_predict(self, x: NDArray, y: NDArray) -> NDArray:
        self.fit(x, y)
        return self.predict(x)

    def partial_fit(self, x: NDArray, y: NDArray) -> None:
        """Partial fit, used for online learning. Calling this does not change the stored data, only appends to it."""
        if self.x is None or self.y is None:
            self.fit(x, y)  # first run of fit
        else:
            # keep only the last `max_memory` samples
            self.x = np.concatenate([self.x, x])[self.max_memory :]  # noqa: E203
            self.y = np.concatenate([self.y, y])[self.max_memory :]  # noqa: E203
