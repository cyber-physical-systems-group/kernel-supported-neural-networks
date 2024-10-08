import torch
from torch import Tensor

from .abstract import MemoryManager
from .transformations import MemoryTransformation


class ExactMemoryManager(MemoryManager):
    """
    Exact memory manager that stores all data points in memory and returns them for each query.

    This memory manager is not suitable for large datasets, but can be used for testing and debugging and small
    datasets or low-dimensional systems, where the memory is not too large.

    It can be used by default with k-nn mode or radius mode.
    """

    def __init__(self, transform: MemoryTransformation | None = None):
        super().__init__()

        self.transform = transform
        # placeholders stored in prepare method
        self.memory: Tensor | None = None
        self.targets: Tensor | None = None

    def prepare(self, memory: Tensor, targets: Tensor | tuple[Tensor, ...]):
        """
        :param memory: tensor of indexed data points to search for nearest neighbors
        :param targets: tensor of target values corresponding to the memory points, can be any number of tensors
        """
        if self.transform is not None:
            memory = self.transform.before_prepare(memory)

        self.memory = memory  # store entire tensors in memory for exact search
        self.targets = targets

    def to(self, device: torch.device):
        """Move memory manager to given device"""
        self.memory = self.memory.to(device)
        self.targets = self.targets.to(device)

    def query_nearest(self, points: Tensor, k: int) -> tuple[Tensor, ...]:
        """
        Query for K-nearest neighbors in memory given input points.

        :param points: input points for which to find nearest neighbours
        :param k: number of nearest neighbours to return
        """
        distances = torch.cdist(points, self.memory)
        # get k nearest points for each query point
        _, index = torch.topk(distances, k=k, largest=False, dim=1)
        # flatten index tensor and remove duplicates, order of points to return does not matter
        index = torch.unique(torch.flatten(index))
        # return found nearest points from memory and collect from all target tensors corresponding to them
        return self.memory[index, :], self.targets[index, :]  # type: ignore

    def query_radius(self, points: Tensor, r: float) -> tuple[Tensor, Tensor]:
        """
        Query for all points in memory within given radius of input points.

        :param points: input points for which to find neighbours
        :param r: radius of the neighbourhood
        """
        distances = torch.cdist(points, self.memory)
        # check if any of the points in memory is within given radius of any query point
        mask = (distances <= r).any(axis=0)
        (index,) = torch.where(mask)
        index = torch.unique(index)

        return self.memory[index, :], self.targets[index, :]

    def query(self, points: Tensor, *, k: int | None = None, r: float | None = None, **kwargs) -> tuple[Tensor, ...]:
        if self.transform is not None:
            points = self.transform.before_query(points)

        if not (k is None) ^ (r is None):
            raise ValueError("Exactly one of: k and r parameter must be specified!")

        if k is not None:
            memory, targets = self.query_nearest(points, k=k)

        if r is not None:
            memory, targets = self.query_radius(points, r=r)

        if self.transform is not None:
            memory = self.transform.after_query(memory)  # type: ignore

        return memory, targets, points  # type: ignore
