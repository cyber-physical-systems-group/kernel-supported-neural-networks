import faiss
import numpy as np
import torch
from torch import Tensor

from .abstract import MemoryManager
from .transformations import MemoryTransformation


class FaissMemoryManager(MemoryManager):
    """
    Memory manager wrapping FAISS library for fast nearest neighbors search. This memory manager is optimized for
    large datasets and can be used for both CPU and GPU search, also using multiple GPUs.

    This memory manager can be used in k-nearest neighbors mode only.

    For details on FAISS, see:
    * https://github.com/facebookresearch/faiss
    * https://github.com/facebookresearch/faiss/wiki
    """

    def __init__(self, transform: MemoryTransformation | None = None, gpu: bool = False, n_gpus: int = 1):
        """
        :param gpu: whether to use GPU for search
        :param n_gpus: number of GPUs to use for search
        """
        super().__init__()

        self.gpu = gpu
        self.n_gpus = n_gpus

        self.transform = transform
        self.memory: Tensor | None = None
        self.targets: Tensor | None = None
        self.faiss_index: faiss.Index | None = None  # type: ignore

    def to(self, device: torch.device):
        """
        Move memory manager to given device. This does not affect the device algorithm is running on,
        just the device on which memory is returned. To run algorithm on GPU, use `gpu` parameter in constructor.
        """
        self.memory = self.memory.to(device)
        self.targets = self.targets.to(device)

    def prepare(self, memory: Tensor, targets: Tensor | tuple[Tensor, ...]):
        if self.transform is not None:
            memory = self.transform.before_prepare(memory)

        self.memory = memory
        self.targets = targets if isinstance(targets, tuple) else (targets,)

        self.faiss_index = faiss.IndexFlatL2(memory.size(-1))  # type: ignore

        if self.gpu:
            if self.n_gpus == 1:
                resource = faiss.StandardGpuResources()  # type: ignore
                self.faiss_index = faiss.index_cpu_to_gpu(resource, 0, self.faiss_index)  # type: ignore
            else:
                resources = [faiss.StandardGpuResources() for _ in range(self.n_gpus)]  # type: ignore
                self.faiss_index = faiss.index_cpu_to_gpu_multiple_py(resources, self.faiss_index)  # type: ignore

        # make sure memory is passed to FAISS as numpy array  on CPU
        # it will manage GPU itself, if gpu=True
        self.faiss_index.add(memory.cpu().numpy())

    def query(self, points: Tensor, *, k: int, **kwargs) -> tuple[Tensor, ...]:  # type: ignore
        if self.transform is not None:
            points = self.transform.before_query(points)

        _, index = self.faiss_index.search(points.detach().cpu().numpy(), k)  # type: ignore
        index = np.unique(np.concatenate(index).flatten())

        memory = self.memory[index, :]
        targets = self.targets[index, :]

        if self.transform is not None:
            memory = self.transform.after_query(memory)

        return memory, targets, points
