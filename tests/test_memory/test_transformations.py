import pytest
import torch
from torch import Tensor

from src.nonparametric.memory import transformations


def batch_arange(start: int, end: int, batch_size: int = 1) -> Tensor:
    """Return batch of arange tensors from start to end for given batch size, always as float32"""
    return torch.arange(start, end).unsqueeze(dim=0).repeat(batch_size, 1).to(torch.float32)


@pytest.mark.parametrize(
    ["points", "decay", "expected"],
    [
        # decay with 0.9 factor for 5 time-steps
        (torch.ones(1, 5), 0.9, torch.Tensor([[1, 0.9, 0.9**2, 0.9**3, 0.9**4]])),
        # decay series of twos with 0.9 factor
        (2 * torch.ones(1, 5), 0.9, torch.Tensor([[2, 2 * 0.9, 2 * 0.9**2, 2 * 0.9**3, 2 * 0.9**4]])),
        # decay series of twos with 0.5 factor
        (2 * torch.ones(1, 5), 0.5, torch.Tensor([[2, 2 * 0.5 * 1, 2 * 0.5**2, 2 * 0.5**3, 2 * 0.5**4]])),
        # decay with 0.99 factor
        (torch.ones(1, 5), 0.99, torch.Tensor([[1, 0.99, 0.99**2, 0.99**3, 0.99**4]])),
        # decay monotonic series with 0.9 factor
        (batch_arange(1, 6), 0.9, torch.Tensor([[1, 2 * 0.9, 3 * 0.9**2, 4 * 0.9**3, 5 * 0.9**4]])),
    ],
)
def test_decay_time_series(points: Tensor, decay: float, expected: Tensor):
    transform = transformations.DecayTimeSeries(decay, inverse=True)
    # memory and query undergo the same transformation of decaying time series
    memory = transform.before_prepare(points)
    query = transform.before_query(points)
    returned = transform.after_query(memory)
    # check if memory and query are decayed and back-transformed query is the same as original query
    torch.testing.assert_close(memory, expected)
    torch.testing.assert_close(query, expected)
    torch.testing.assert_close(returned, points)


@pytest.mark.parametrize(
    ["points", "expected"],
    [
        # mean = 1 and std = 1
        (torch.Tensor([[0, 0, 2, 2]]), torch.Tensor([[-1, -1, 1, 1]])),
        # mean = 0 and std = 1
        (torch.Tensor([[-1, -1, 1, 1]]), torch.Tensor([[-1, -1, 1, 1]])),
        # mean = 0 and std = 10
        (torch.Tensor([[-10, -10, 10, 10]]), torch.Tensor([[-1, -1, 1, 1]])),
        # mean = 1 and std = 1
        (torch.Tensor([[1, 1, 3, 3]]), torch.Tensor([[-1, -1, 1, 1]])),
    ],
)
def test_normalize(points: Tensor, expected: Tensor):
    transform = transformations.Normalize()
    # transform memory and query points and back-transform query to the returned representation
    memory = transform.before_prepare(points)
    query = transform.before_query(points)
    returned = transform.after_query(query)
    # check if memory and query are normalized and back-transformed query is the same as original query
    torch.testing.assert_close(memory, expected)
    torch.testing.assert_close(query, expected)
    torch.testing.assert_close(returned, points)


@pytest.mark.parametrize(
    ["points", "target_dim", "expected"],
    [
        # return last element for single item batch of points
        (batch_arange(1, 11), 1, torch.Tensor([[10]])),
        # return last 3 elements for single item batch of points
        (batch_arange(1, 11), 3, batch_arange(8, 11)),
        # return last 5 elements for single item batch of points
        (batch_arange(1, 11), 5, batch_arange(6, 11)),
    ],
)
def test_truncate_delay_line(points: Tensor, target_dim: int, expected: Tensor):
    transform = transformations.TruncateDelayLine(target_dim)
    # memory and query undergo the same transformation of keeping last `target_dim` elements
    memory = transform.before_prepare(points)
    query = transform.before_query(points)

    assert memory.size(-1) == target_dim
    assert query.size(-1) == target_dim
    torch.testing.assert_close(memory, expected)
    torch.testing.assert_close(query, expected)
