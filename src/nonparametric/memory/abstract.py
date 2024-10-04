from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .transformations import MemoryTransformation


class MemoryManager(ABC):
    """
    Interface for memory manager for non-parametric models (such as kernel regression).

    This class is used to enable non-parametric models to be used with large datasets, since they need to store all
    training data. Memory managers can be used to apply selection or search algorithms to reduce the number of samples
    return for each prediction, for example for kernel model returning only samples near the input points in given
    prediction batch.
    """

    def __init__(self, transform: MemoryTransformation | None = None):
        ...

    @abstractmethod
    def prepare(self, memory: Tensor, targets: Tensor | tuple[Tensor, ...]):
        """Prepare memory manager for use, for example build index for nearest neighbors search"""
        ...

    @abstractmethod
    def to(self, device: torch.device):
        """Move memory manager to given device"""
        ...

    @abstractmethod
    def query(self, points: Tensor, **kwargs) -> tuple[Tensor, ...]:  # type: ignore
        """Default call method, can be different for different memory managers"""
        ...
