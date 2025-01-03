import pytest
import torch
from torch import Tensor

from src.training.module import bounded_linear_unit


@pytest.mark.parametrize(
    ["inputs", "lower", "upper", "expected"],
    [
        # all elements in scalar bounds - output is the same as input
        (torch.ones(5), -1, 2, torch.ones(5)),
        # all elements outside upper scalar bound - output is upper bound
        (torch.ones(5), -1, 0, torch.zeros(5)),
        # all elements outside lower scalar bound - output is lower bound
        (torch.ones(5), 2, 3, 2 * torch.ones(5)),
        # all elements in varying bounds - output is the same as input
        (torch.ones(5), torch.Tensor([0, -1, 0, -1, 0]), torch.Tensor([2, 3, 5, 6, 7]), torch.ones(5)),
        # all elements outside upper varying bounds - output is upper bound with every element different
        (
            10 * torch.ones(5),
            torch.Tensor([0, -1, 0, -1, 0]),
            torch.Tensor([0, 2, 5, 6, 7]),
            torch.Tensor([0, 2, 5, 6, 7]),
        ),
        # all elements outside lower varying bounds - output is lower bound with every element different
        (
            -10 * torch.ones(5),
            torch.Tensor([0, -1, 0, -1, 0]),
            torch.Tensor([2, 3, 5, 6, 1]),
            torch.Tensor([0, -1, 0, -1, 0]),
        ),
        # some elements outside upper varying bounds - output contains elements from input and upper bound
        (
            torch.Tensor([1, 2, 3, 4, 5]),
            torch.Tensor([0, 0, 0, 0, 0]),
            torch.Tensor([2, 1, 2, 1, 2]),
            torch.Tensor([1, 1, 2, 1, 2]),
        ),
        # some elements outside lower varying bounds - output contains elements from input and lower bound
        (
            torch.Tensor([1, 2, 3, 4, 5]),
            torch.Tensor([0, 5, 0, 5, 0]),
            torch.Tensor([10, 10, 10, 10, 10]),
            torch.Tensor([1, 5, 3, 5, 5]),
        ),
        # some elements outside upper and lower varying bounds - output contains elements from input and both bounds
        (
            torch.Tensor([1, 2, 3, 4, 5]),
            torch.Tensor([5, 0, 5, 0, 0]),
            torch.Tensor([10, 10, 10, 3, 3]),
            torch.Tensor([5, 2, 5, 3, 3]),
        ),
    ],
)
def test_bounded_linear_unit(inputs: Tensor, lower: float | Tensor, upper: float | Tensor, expected: Tensor):
    outputs = bounded_linear_unit(inputs, lower, upper)
    assert torch.allclose(outputs, expected)
