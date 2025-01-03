import pytest
import torch

from src.training.utils import unbatch


def tensor_batch_iterable(n_batches: int, batch_size: int, batch_shape: tuple[int, ...], n_tensors: int):
    """
    Testing util for defining batched iterable for Tensors returning given
    number of batches with given batch size all with ones tensors
    """
    for _ in range(n_batches):
        yield tuple([torch.ones((batch_size,) + batch_shape) for _ in range(n_tensors)])


@pytest.mark.parametrize(
    ["n_batches", "batch_size", "batch_shape", "n_tensors", "expected_shape"],
    [
        # single batch with 32 items with shape (10, 1)
        (1, 32, (10, 1), 1, (32, 10, 1)),
        # single batch with 32 items with shape (10,) for features and targets
        (1, 32, (10,), 2, (32, 10)),
        # 10 batches with 32 items with shape (10, 1)
        (10, 32, (10, 1), 1, (320, 10, 1)),
        # 10 batches with 32 items with shape (10,) for features and targets
        (10, 32, (10,), 2, (320, 10)),
        # single item batches
        (10, 1, (10, 1), 1, (10, 10, 1)),
        # 4-tuple dataloader
        (10, 32, (10, 1), 4, (320, 10, 1)),
    ],
)
def test_unbatch(n_batches: int, batch_size: int, batch_shape: tuple, n_tensors: int, expected_shape: tuple):
    iterable = tensor_batch_iterable(n_batches, batch_size, batch_shape, n_tensors)

    for tensor in unbatch(iterable):
        assert tensor.shape == expected_shape  # in the test all input tensors have the same shape
