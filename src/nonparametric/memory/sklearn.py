import warnings
from typing import Literal

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from .abstract import MemoryManager
from .transformations import MemoryTransformation


class SklearnMemoryManager(MemoryManager):
    """
    Memory manager wrapping sklearn's NearestNeighbors. This memory manager can be used for relatively large datasets,
    however for very large datasets it is recommended to use approximated nearest neighbors algorithms.

    It can be used by default with k-nn mode or radius mode.
    """

    def __init__(
        self,
        k: int,
        transform: MemoryTransformation | None = None,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        **parameters,
    ):
        """
        :param k: number of nearest neighbors used to initialize the model, it can be over-written
        :param algorithm: algorithm used to compute nearest neighbors
        :param parameters: additional parameters passed to NearestNeighbors model,
                           for more details see: see: https://scikit-learn.org/stable/modules/neighbors.html#neighbors
        """
        super().__init__()

        # placeholders stored in prepare method
        self.memory: Tensor | None = None
        self.targets: Tensor | None = None
        self.index: NearestNeighbors | None = None

        self.k = k
        self.transform = transform
        self.algorithm = algorithm
        self.parameters = parameters

    def prepare(self, memory: Tensor, targets: Tensor | tuple[Tensor, ...]):
        """
        :param memory: tensor of indexed data points to search for nearest neighbors
        :param targets: tensor of target values corresponding to the memory points, can be any number of tensors
        """
        if self.transform is not None:
            memory = self.transform.before_prepare(memory)

        self.memory = memory
        self.targets = targets

        self.index = NearestNeighbors(n_neighbors=self.k, algorithm=self.algorithm, **self.parameters)
        self.index.fit(memory)

    def to(self, device: torch.device):
        """Move memory manager to given device"""
        self.memory = self.memory.to(device)
        self.targets = tuple(target.to(device) for target in self.targets)

    def query_nearest(self, points: Tensor, k: int | None = None) -> tuple[Tensor, ...]:
        if k != self.k:
            warnings.warn(f"Using k different from the one used to fit the model! {k} != {self.k}")

        k = k or self.k  # default k was used to fit the neighbors model

        index = self.index.kneighbors(points, n_neighbors=k, return_distance=False)
        index = np.unique(np.concatenate(index).flatten())

        return self.memory[index, :], self.targets[index, :]

    def query_radius(self, points: Tensor, r: float) -> tuple[Tensor, Tensor]:
        index = self.index.radius_neighbors(points, radius=r, return_distance=False)
        index = np.unique(np.concatenate(index).flatten())

        return self.memory[index, :], self.targets[index, :]

    def query(
        self, points: Tensor, *, k: int | None = None, r: float | None = None, **kwargs
    ) -> tuple[Tensor, ...]:  # type: ignore
        return_device = self.memory.device

        if points.device != self.memory.device:
            return_device = points.device  # remember device where points came from
            # move points to memory device, since NNDescent only supports CPU
            points = points.detach().to(self.memory.device)

        if self.index is None:
            raise RuntimeError("Index is not built, call prepare method first!")

        if not (k is None) ^ (r is None):
            raise ValueError("Exactly one of: k and r parameter must be specified!")

        if self.transform is not None:
            points = self.transform.before_query(points)

        if k is not None:
            memory, targets = self.query_nearest(points, k=k)

        if r is not None:
            memory, targets = self.query_radius(points, r=r)

        if self.transform is not None:
            memory = self.transform.after_query(memory)

        memory = memory.to(return_device)  # cast back to device where points came from  # noqa: E501
        targets = self.targets.to(return_device)
        return memory, targets, points
