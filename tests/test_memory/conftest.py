import pytest
import torch
from torch import Tensor

from src.nonparametric.memory import ExactMemoryManager, NNDescentMemoryManager, SklearnMemoryManager


@pytest.fixture(scope="module")
def memory() -> Tensor:
    # 101 points in [0, 1] range spaced by 0.01 and shape [101, 1]
    return torch.linspace(0, 1, 101).unsqueeze(-1)


@pytest.fixture(scope="module")
def targets(memory: Tensor) -> Tensor:
    return 2 * memory  # dummy targets


@pytest.fixture(scope="module")
def exact_memory_manager(memory: Tensor, targets: Tensor):
    manager = ExactMemoryManager()
    manager.prepare(memory, targets)

    return manager


@pytest.fixture(scope="module")
def nn_descent_memory_manager(memory: Tensor, targets: Tensor):
    manager = NNDescentMemoryManager(epsilon=1.0, metric="euclidean")  # high epsilon to get certain results in tests
    manager.prepare(memory, targets)

    return manager


@pytest.fixture(scope="module")
def scikit_auto_memory_manager(memory: Tensor, targets: Tensor):
    # k will over-written by query method in tests, this is used to build index
    manager = SklearnMemoryManager(k=10, algorithm="ball_tree")
    manager.prepare(memory, targets)

    return manager


@pytest.fixture(scope="module")
def scikit_ball_tree_memory_manager(memory: Tensor, targets: Tensor):
    # k will over-written by query method in tests, this is used to build index
    manager = SklearnMemoryManager(k=10, algorithm="ball_tree")
    manager.prepare(memory, targets)

    return manager


@pytest.fixture(scope="module")
def scikit_kd_tree_memory_manager(memory: Tensor, targets: Tensor):
    # k will over-written by query method in tests, this is used to build index
    manager = SklearnMemoryManager(k=10, algorithm="kd_tree")
    manager.prepare(memory, targets)

    return manager
