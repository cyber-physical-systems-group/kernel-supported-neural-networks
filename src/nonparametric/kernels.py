import math
from typing import Callable

import torch
from torch import Tensor

KernelCallable = Callable[[Tensor, float, float], Tensor]


@torch.no_grad()
def box_kernel(x: Tensor, width: float, offset: float) -> Tensor:
    """
    Computes box kernel density for each input point and memory point.

    :param x: points where function is to be estimated
    :param width: bandwidth of the kernel
    :param offset: center of the kernel, offset from 0
    """
    return (torch.abs(x - offset) <= width).to(x.dtype)  # type: ignore


@torch.no_grad()
def gaussian_kernel(x: Tensor, width: float, offset: float) -> Tensor:
    """
    Computes gaussian kernel density for each input point and memory point.

    :param x: points where function is to be estimated
    :param width: bandwidth of the kernel
    :param offset: center of the kernel, offset from 0
    """
    x = x / width
    x = x - offset
    return torch.exp(-0.5 * x**2) / (width * torch.sqrt(2 * torch.tensor(math.pi)))


@torch.no_grad()
def epanechnikov_kernel(x: Tensor, width: float, offset: float) -> Tensor:
    """
    Computes epanechnikov kernel density for each input point and memory point.

    :param x: points where function is to be estimated
    :param width: bandwidth of the kernel
    :param offset: center of the kernel, offset from 0
    """
    x = x / width
    x = x - offset
    kernel = 0.75 * (1 - x**2) * (torch.abs(x) <= 1)
    return kernel.to(x.dtype)


@torch.no_grad()
def triangular_kernel(x: Tensor, width: float, offset: float) -> Tensor:
    """
    Computes triangular kernel density for each input point and memory point.

    :param x: points where function is to be estimated
    :param width: bandwidth of the kernel
    :param offset: center of the kernel, offset from 0
    """
    x = x / width
    x = x - offset
    kernel = (1 - torch.abs(x)) * (torch.abs(x) <= 1)
    return kernel.to(x.dtype)


@torch.no_grad()
def quartic_kernel(x: Tensor, width: float, offset: float) -> Tensor:
    """
    Computes quartic kernel density for each input point and memory point.

    :param x: points where function is to be estimated
    :param width: bandwidth of the kernel
    :param offset: center of the kernel, offset from 0
    """
    x = x / width
    x = x - offset
    kernel = (15 / 16 * (1 - x**2) ** 2) * (torch.abs(x) <= 1)
    return kernel.to(x.dtype)


@torch.no_grad()
def triweight_kernel(x: Tensor, width: float, offset: float) -> Tensor:
    """
    Computes triweight kernel density for each input point and memory point.

    :param x: points where function is to be estimated
    :param width: bandwidth of the kernel
    :param offset: center of the kernel, offset from 0
    """
    x = x / width
    x = x - offset
    kernel = 35 / 32 * (1 - (x**2) ** 3) * (torch.abs(x) <= 1)
    return kernel.to(x.dtype)


@torch.no_grad()
def tricube_kernel(x: Tensor, width: float, offset: float) -> Tensor:
    """
    Computes tricube kernel density for each input point and memory point.

    :param x: points where function is to be estimated
    :param width: bandwidth of the kernel
    :param offset: center of the kernel, offset from 0
    """
    x = x / width
    x = x - offset
    kernel = 70 / 81 * (1 - torch.abs(x) ** 3) ** 3 * (torch.abs(x) <= 1)
    return kernel.to(x.dtype)


@torch.no_grad()
def cosine_kernel(x: Tensor, width: float, offset: float) -> Tensor:
    """
    Computes cosine kernel density for each input point and memory point.

    :param x: points where function is to be estimated
    :param width: bandwidth of the kernel
    :param offset: center of the kernel, offset from 0
    """
    x = x / width
    x = x - offset
    kernel = math.pi / 4 * torch.cos(x * math.pi / 2) * (torch.abs(x) <= 1)
    return kernel.to(x.dtype)


@torch.no_grad()
def logistic_kernel(x: Tensor, width: float, offset: float) -> Tensor:
    """
    Computes logistic kernel density for each input point and memory point.

    :param x: points where function is to be estimated
    :param width: bandwidth of the kernel
    :param offset: center of the kernel, offset from 0
    """
    x = x / width
    x = x - offset
    return 1 / (torch.exp(x) + 2 + torch.exp(-1 * x))


@torch.no_grad()
def sigmoid_kernel(x: Tensor, width: float, offset: float) -> Tensor:
    """
    Computes sigmoid kernel density for each input point and memory point.

    :param x: points where function is to be estimated
    :param width: bandwidth of the kernel
    :param offset: center of the kernel, offset from 0
    """
    x = x / width
    x = x - offset
    return 2 / math.pi / (torch.exp(x) + torch.exp(-1 * x))


COMPACT_CARRIER_KERNELS = (
    box_kernel,
    epanechnikov_kernel,
    triangular_kernel,
    quartic_kernel,
    triweight_kernel,
    tricube_kernel,
    cosine_kernel,
)
