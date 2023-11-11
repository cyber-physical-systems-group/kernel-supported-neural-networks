import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin


class NumpyNadarayaWatsonEstimator(RegressorMixin, BaseEstimator):
    """
    Nadaraya-Watson estimator for regression, it is a non-parametric estimator that uses kernel functions.
    Fitting is done by storing all the training data, so it is not recommended for large datasets.
    Inference is done by computing the weighted average of the stored training data using given kernel function.

    This implementation supports returning confidence bounds for the predictions, which requires the knowing Lipschitz
    constant and noise variance of the function being estimated.

    :note: two versions of Nadaraya-Watson estimator are implemented, one using numpy and one using pytorch.
    """

    def __init__(self, bandwidth: float, max_memory: int | None = None):
        """
        :param bandwidth: parameter for kernel function
        :param max_memory: maximal number of samples to keep in memory, used with partial_fit
                           defaults to None, which does not truncate stored
        """
        super().__init__()

        self.bandwidth = bandwidth
        self.max_memory = max_memory

        self.x: NDArray | None = None
        self.y: NDArray | None = None

    @staticmethod
    def kernel(memory: NDArray, points: NDArray, bandwidth: float) -> NDArray:
        """Box kernel function is fixed as it minimizes the theoretical bounds"""
        diff = memory - points
        return (np.abs(diff) / bandwidth <= 1).astype(float)

    def fit(self, x: NDArray, y: NDArray) -> None:
        self.x = x
        self.y = y

    def compute_bounds(self, kappa: NDArray, delta: float, lipschitz_constant: float, noise_variance: float) -> NDArray:
        """
        The bounds are distance from the predicted value, so the true value is in [prediction - bound] with
        probability `1 - delta`, depending on the Lipschitz constant and noise variance of the function being estimated.

        :param kappa: sum of weights for each point
        :param delta: confidence, probability of true value being in bounds is `1 - delta`
        :param lipschitz_constant: Lipschitz constant of the function being estimated
        :param noise_variance: variance of the noise in the data

        :return: ND array of bound with the same dimensions as function
        """
        lower = (kappa <= 1) * np.sqrt(np.log(np.sqrt(2) / delta))
        upper = (kappa > 1) * np.sqrt(kappa * np.log(np.sqrt(1 + kappa) / delta))
        alpha = lower + upper

        return lipschitz_constant * self.bandwidth + 2 * noise_variance * alpha / kappa

    def predict(
        self,
        x: NDArray,
        with_bounds: bool = False,
        delta: float | None = None,
        lipschitz_constant: float | None = None,
        noise_variance: float | None = None,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """
        :param x: points where function is to be predicted from training data
        :param with_bounds: if True, returns 2D array of lower and upper bounds
        :param delta: confidence, probability of true value being in bounds is `1 - delta`
        :param lipschitz_constant: Lipschitz constant of the function being estimated
        :param noise_variance: variance of the noise in the data
        """
        memory = np.expand_dims(self.x, axis=-1)
        points = np.expand_dims(x, axis=0)

        weights = self.kernel(memory, points, bandwidth=self.bandwidth)
        predictions = (weights * np.expand_dims(self.y, axis=-1)).sum(axis=0) / weights.sum(axis=0)

        if with_bounds:
            if delta is None or lipschitz_constant is None or noise_variance is None:
                message = "delta, lipschitz_constant and noise_variance must be provided when with_bounds is True!"
                raise ValueError(message)

            kappa = weights.sum(axis=0)
            bounds = self.compute_bounds(kappa, delta, lipschitz_constant, noise_variance)
            return predictions, bounds

        return predictions

    def fit_predict(self, x: NDArray, y: NDArray, **kwargs) -> NDArray:
        self.fit(x, y)
        return self.predict(x, **kwargs)
