import math

import torch


class TorchKernelRegression(torch.nn.Module):
    """
    Fitting is done by storing all the training data, so it is not recommended for large datasets.
    Inference is done by computing the weighted average of the stored training data using given kernel function.

    This implementation supports returning confidence bounds for the predictions, which requires the knowing Lipschitz
    constant and noise variance of the function being estimated.

    :note: two versions of kernel regression are implemented, one using numpy and one using pytorch.
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

        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None

    @staticmethod
    def kernel(memory: torch.Tensor, points: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """Box kernel function is fixed as it minimizes the theoretical bounds"""
        diff = memory - points
        return (torch.abs(diff) / bandwidth <= 1).to(torch.float32)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def compute_bounds(
        self, kappa: torch.Tensor, delta: float, lipschitz_constant: float, noise_variance: float
    ) -> torch.Tensor:
        """
        The bounds are distance from the predicted value, so the true value is in [prediction - bound] with
        probability `1 - delta`, depending on the Lipschitz constant and noise variance of the function being estimated.

        :param kappa: sum of weights for each point
        :param delta: confidence, probability of true value being in bounds is `1 - delta`
        :param lipschitz_constant: Lipschitz constant of the function being estimated
        :param noise_variance: variance of the noise in the data

        :return: ND array of bound with the same dimensions as function
        """
        lower = (kappa <= 1) * math.sqrt(math.log(math.sqrt(2) / delta))
        upper = (kappa > 1) * torch.sqrt(kappa * torch.log(torch.sqrt(1 + kappa) / delta))
        alpha = lower + upper

        return lipschitz_constant * self.bandwidth + 2 * noise_variance * alpha / kappa

    def predict(
        self,
        x: torch.Tensor,
        with_bounds: bool = False,
        delta: float | None = None,
        lipschitz_constant: float | None = None,
        noise_variance: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """x
        :param x: points where function is to be predicted from training data
        :param with_bounds: if True, returns 2D array of lower and upper bounds
        :param delta: confidence, probability of true value being in bounds is `1 - delta`
        :param lipschitz_constant: Lipschitz constant of the function being estimated
        :param noise_variance: variance of the noise in the data
        """
        memory = torch.unsqueeze(self.x, dim=-1)
        points = torch.unsqueeze(x, dim=0)

        weights = self.kernel(memory, points, bandwidth=self.bandwidth)
        predictions = (weights * torch.unsqueeze(self.y, dim=-1)).sum(axis=0) / weights.sum(dim=0)

        if with_bounds:
            if delta is None or lipschitz_constant is None or noise_variance is None:
                message = "delta, lipschitz_constant and noise_variance must be provided when with_bounds is True!"
                raise ValueError(message)

            kappa = weights.sum(dim=0)
            bounds = self.compute_bounds(kappa, delta, lipschitz_constant, noise_variance)
            return predictions, bounds

        return predictions

    def fit_predict(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        self.fit(x, y)
        return self.predict(x, **kwargs)
