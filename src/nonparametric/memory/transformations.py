from abc import ABC, abstractmethod

import torch
from torch import Tensor


class MemoryTransformation(ABC):
    """
    Interface for memory transformations, which can be used to modify memory and query points before they are used in
    memory manager and after, to return them to kernel regression model.

    This can be used to decrease dimension of memory and query points, normalize the data for the nearest neighbour
    search or to make data points further in time (using delay-line) further apart in memory space,
    since they are less important for prediction.
    """

    @abstractmethod
    def before_prepare(self, memory: Tensor) -> Tensor: ...

    @abstractmethod
    def before_query(self, query: Tensor) -> Tensor: ...

    @abstractmethod
    def after_query(self, memory: Tensor) -> Tensor: ...


class DecayTimeSeries(MemoryTransformation):
    """
    Decay time series memory points to make points further in time further apart in memory space, since they are less
    important for prediction.

    This memory transformation changes points both for nearest neighbour search and kernel regression.
    """

    def __init__(self, decay: float, inverse: bool = False):
        """
        :param decay: decay factor for time series
        :param inverse: whether to back-transform memory after searching,
                        so kernel regression works in the original space
        """
        if decay <= 0 or decay >= 1:
            raise ValueError("Decay factor must be in range (0, 1)")

        self.decay = decay
        self.inverse = inverse

        self.query_decay: Tensor | None = None

    def before_prepare(self, memory: Tensor) -> Tensor:
        decays = self.decay ** torch.arange(memory.size(-1), dtype=memory.dtype, device=memory.device)
        return memory * decays.unsqueeze(0)

    def before_query(self, query: Tensor) -> Tensor:
        decays = self.decay ** torch.arange(query.size(-1), dtype=query.dtype, device=query.device)
        self.query_decay = decays
        return query * decays.unsqueeze(0)

    def after_query(self, memory: Tensor) -> Tensor:
        if self.inverse:
            return memory / self.query_decay

        return memory


class Normalize(MemoryTransformation):
    """
    Normalize memory and query points to have zero mean and unit variance.

    This memory transformation changes points both for nearest neighbour search only and uses inverse transformation
    before kernel regression.
    """

    def __init__(self):
        self.mean: float | None = None
        self.std: float | None = None

    def before_prepare(self, memory: Tensor) -> Tensor:
        self.std, self.mean = torch.std_mean(memory, dim=-1, correction=0)
        return (memory - self.mean) / self.std

    def before_query(self, query: Tensor) -> Tensor:
        return (query - self.mean) / self.std

    def after_query(self, memory: Tensor) -> Tensor:
        return memory * self.std + self.mean


class TruncateDelayLine(MemoryTransformation):
    """
    Truncate delay line to last `target_dim` elements in order to make the nearest neighbour search faster and allow
    kernel model to use only the most recent data for prediction, therefore decreasing the effective bandwidth.

    This memory transformation changes points both for nearest neighbour search and kernel regression.
    """

    def __init__(self, target_dim: int):
        self.target_dim = target_dim

    def before_prepare(self, memory: Tensor) -> Tensor:
        return memory[:, -self.target_dim :]  # noqa E203

    def before_query(self, query: Tensor) -> Tensor:
        return query[:, -self.target_dim :]  # noqa E203

    def after_query(self, memory: Tensor) -> Tensor:
        return memory  # memory is modified during before_prepare call
