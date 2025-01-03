from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import Module


@torch.no_grad()
def reset_parameters(module: Module):
    """
    Reset parameters of given module, if it has reset_parameters method implemented

    :param module: any torch module, including Sequential and composite modules to reset parameters

    :example:
        >>> model.apply(reset_parameters)
    """
    reset_func = getattr(module, "reset_parameters", None)

    if callable(reset_func):
        module.reset_parameters()


def unbatch(batched: Iterable[tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
    """
    Converts batched dataset given as iterable (usually lazy iterable) to tuple of tensors

    :example:
    >>> loader: DataLoader = get_loader(batch_size=32)  # assume get_loader is implemented
    >>> x, y = unbatch(loader)
    >>> x.shape
    ... (320, 10, 1)  # (BATCH_SIZE * N_BATCHES, *DATA_SHAPE)
    """
    for batch_idx, batch in enumerate(batched):
        if batch_idx == 0:  # initialize unbatched list of first batch
            n_tensors = len(batch) if isinstance(batch, Sequence) else 1
            unbatched = [Tensor() for _ in range(n_tensors)]

        for i, tensor in enumerate(batch):
            unbatched[i] = torch.cat([unbatched[i], tensor])

    return tuple(unbatched)
