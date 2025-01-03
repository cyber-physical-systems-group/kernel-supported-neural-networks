import pytest
import torch
from torch import Tensor

from src.training.loss import BoundedMSELoss


@pytest.mark.parametrize(
    ["y_true", "y_pred", "lower", "upper", "gamma", "expected"],
    (
        # perfect prediction
        (torch.ones(5), torch.ones(5), torch.zeros(5), 5 * torch.ones(5), 1.0, 0.0),
        # loss falls back to MSE if gamma is 0, in this case error is 1.0
        (torch.ones(5), torch.Tensor([2, 2, 2, 2, 2]), torch.zeros(5), 5 * torch.ones(5), 1.0, 1.0),
        # loss is 0 if prediction is inside bounds
        (torch.ones(5), torch.Tensor([2, 2, 2, 2, 2]), torch.zeros(5), 5 * torch.ones(5), 1.0, 1.0),
        # loss is 1 + 0.5 for crossing bounds in the last element
        (torch.ones(5), torch.Tensor([2, 2, 2, 2, 2]), torch.zeros(5), torch.Tensor([5, 5, 5, 5, 1.5]), 1.0, 1.5),
        # loss is 1 + 5 for crossing bounds in each element (for test purpose true value and bound are equal)
        (torch.ones(5), torch.Tensor([2, 2, 2, 2, 2]), torch.zeros(5), torch.ones(5), 1.0, 6.0),
        # loss is 1 + 2 * 0.5 (gamma = 2) for crossing bounds in the last element
        (torch.ones(5), torch.Tensor([2, 2, 2, 2, 2]), torch.zeros(5), torch.Tensor([5, 5, 5, 5, 1.5]), 2.0, 2.0),
        # loss is 1 + 0.5 * 0.5 (gamma = 1/2) for crossing bounds in the last element
        (torch.ones(5), torch.Tensor([2, 2, 2, 2, 2]), torch.zeros(5), torch.Tensor([5, 5, 5, 5, 1.5]), 0.5, 1.25),
        # gamma = 0 so loss falls back to regular MSE
        (torch.ones(5), torch.Tensor([2, 2, 2, 2, 2]), torch.zeros(5), torch.Tensor([5, 5, 5, 5, 1.5]), 0.0, 1.0),
    ),
)
def test_bounded_mse_loss(y_true: Tensor, y_pred: Tensor, lower: Tensor, upper: Tensor, gamma: float, expected: float):
    loss = BoundedMSELoss(gamma)

    with torch.no_grad():
        result = loss(y_true, y_pred, lower, upper)  # type: ignore
        assert result.item() == expected
