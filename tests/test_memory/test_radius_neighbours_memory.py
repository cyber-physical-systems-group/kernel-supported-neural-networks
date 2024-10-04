import pytest
import torch
from pytest_lazy_fixtures import lf  # noqa: F401
from torch import Tensor

from src.nonparametric.memory.abstract import MemoryManager


@pytest.mark.parametrize(
    "memory_manager",
    [
        pytest.lazy_fixtures("exact_memory_manager"),
        pytest.lazy_fixtures("scikit_auto_memory_manager"),
        pytest.lazy_fixtures("scikit_kd_tree_memory_manager"),
        pytest.lazy_fixtures("scikit_ball_tree_memory_manager"),
    ],
)
@pytest.mark.parametrize(
    "points, r, expected",
    (
        # query for single point, since they are spaced 0.01 apart, radius of 0.001 should return only the point itself
        (torch.tensor([[0.5]]), 0.001, torch.tensor([[0.5]])),
        # query for multiple points, r = 0.15 to catch 3 points in [0.49, 0.51] range
        (torch.tensor([[0.5]]), 0.015, torch.tensor([[0.49], [0.5], [0.51]])),
        # query on the edge of range
        (torch.tensor([[0.0]]), 0.001, torch.tensor([[0.0]])),
        (torch.tensor([[1.0]]), 0.001, torch.tensor([[1.0]])),
        # query for multiple points on the edge of range
        (torch.tensor([[0.0]]), 0.05, torch.tensor([[0.0], [0.01], [0.02], [0.03], [0.04], [0.05]])),
        # for numerical reasons, the point 0.95 will not be returned by scikit-learn radius query, if exactly r=0.05
        (torch.tensor([[1.0]]), 0.051, torch.tensor([[0.95], [0.96], [0.97], [0.98], [0.99], [1.0]])),
        # batch query
        (torch.tensor([[0.5], [0.25]]), 0.001, torch.tensor([[0.25], [0.5]])),
        # batch query with multiple points
        (torch.tensor([[0.5], [0.25]]), 0.015, torch.tensor([[0.24], [0.25], [0.26], [0.49], [0.5], [0.51]])),
        # batch query with multiple points without exact matches to memory
        (torch.tensor([[0.501], [0.2501]]), 0.015, torch.tensor([[0.24], [0.25], [0.26], [0.49], [0.5], [0.51]])),
    ),
)
def test_radius_query(memory_manager: MemoryManager, points: Tensor, r: int, expected: Tensor):
    memory, _, _ = memory_manager.query(points, r=r)  # ignore targets
    # sort memory to ensure all points are in the same order in result and expected, tensor is 1D
    memory, _ = torch.sort(memory)
    torch.testing.assert_close(memory, expected)
