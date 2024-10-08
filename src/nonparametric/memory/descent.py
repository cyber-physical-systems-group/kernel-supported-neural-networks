import torch
from pynndescent import NNDescent
from torch import Tensor

from .abstract import MemoryManager
from .transformations import MemoryTransformation


class NNDescentMemoryManager(MemoryManager):
    def __init__(self, epsilon: float, transform: MemoryTransformation | None = None, **parameters):
        """
        :param epsilon: search parameter for NNDescent, see: https://pynndescent.readthedocs.io/en/latest/api.html
                        given statically for all queries
        :param parameters: parameters for NNDescent algorithm,
                           see: https://pynndescent.readthedocs.io/en/latest/api.html
        """
        super().__init__()

        # placeholders stored in prepare method
        self.memory: Tensor | None = None
        self.targets: Tensor | None = None

        self.parameters = parameters
        self.epsilon = epsilon

        self.transform = transform
        self.index = None  # build deferred until first query, it takes significant amount of time

    def prepare(self, memory: Tensor, targets: Tensor | tuple[Tensor, ...]):
        """
        Build index for nearest neighbors search over memory using `NNDescent`

        :param memory: tensor of indexed data points to search for nearest neighbors
        :param targets: tensor of target values corresponding to the memory points, can be any number of tensors
        """
        if self.transform is not None:
            memory = self.transform.before_prepare(memory)

        self.memory = memory
        self.targets = targets  # store targets as tuple

        self.index = NNDescent(self.memory, **self.parameters)
        self.index.prepare()

    @property
    def neighbor_graph(self) -> tuple[Tensor, Tensor]:
        """Returns the neighbor graph of the memory points"""
        if self.index is None:
            raise RuntimeError("Index is not built, call prepare method first!")

        return self.index.neighbor_graph

    def to(self, device: torch.device):
        if device != torch.device("cpu"):
            raise ValueError("NNDescentMemoryManager only supports CPU device!")

    def query(self, points: Tensor, *, k: int, **kwargs) -> tuple[Tensor, ...]:  # type: ignore
        """
        Query for K-nearest neighbors in memory given input points.

        :param points: input points for which to find nearest neighbours
        :param k: number of nearest neighbours to return
        """
        return_device = self.memory.device

        if points.device != self.memory.device:
            return_device = points.device  # remember device where points came from
            # move points to memory device, since NNDescent only supports CPU
            points = points.detach().to(self.memory.device)

        if self.index is None:
            raise RuntimeError("Index is not built, call prepare method first!")

        if self.transform is not None:
            points = self.transform.before_query(points)

        indexed, _ = self.index.query(points, k=k, epsilon=self.epsilon)
        # memory manager returns flat memory for all query points
        # duplicates are removed and the dimensionality is reduced to 1
        indexed = torch.unique(torch.from_numpy(indexed.flatten()))
        # return found nearest points from memory and collect from all target tensors corresponding to them
        memory = self.memory[indexed, :].to(return_device)  # cast back to device where points came from
        targets = self.targets[indexed, :].to(return_device)

        if self.transform is not None:
            memory = self.transform.after_query(memory)  # targets are not transformed

        return memory, targets, points
