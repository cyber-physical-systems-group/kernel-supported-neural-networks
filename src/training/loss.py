import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class BoundedMSELoss(Module):
    """MSE loss with penalty for crossing bounds of the nonparametric estimator"""

    def __init__(self, gamma: float):
        """
        :param gamma: penalty factor for crossing bounds, defaults to 0.0
        """
        self.gamma = gamma

        super(BoundedMSELoss, self).__init__()

    def forward(
        self, y_true: Tensor, y_pred: Tensor, lower: Tensor | None = None, upper: Tensor | None = None
    ) -> float | Tensor:
        loss = F.mse_loss(y_pred, y_true, reduction="mean")

        # if gamma is given as 0 or if bounds are not given,
        # no penalty is applied and BoundedMSELoss is equivalent to MSE, for example when used as validation loss
        if self.gamma == 0 or lower is None or upper is None:
            return loss

        penalty = torch.where(
            (y_pred < lower) | (y_pred > upper),  # find predictions outside of bounds
            torch.min(torch.abs(y_pred - upper), torch.abs(y_pred - lower)),  # calculate distance to closer bound
            torch.tensor(0.0, device=y_pred.device),  # zero-fill for predictions inside bounds
        )

        # returns loss as sum of MSE and cumulated penalty for crossing bounds with gamma factor
        return loss + self.gamma * torch.sum(penalty)
