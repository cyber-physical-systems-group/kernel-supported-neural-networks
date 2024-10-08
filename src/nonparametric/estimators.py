import torch.nn.functional as F
from torch import Tensor


def noise_variance(signal: Tensor, kernel_size: int) -> float:
    """
    Estimates the noise variance of a signal by smoothing it with a moving average filter and computing the variance
    of the difference between the signal and the smoothed signal.

    :param signal: 1D tensor with signal values
    :param kernel_size: size of moving average kernel
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd!")

    if signal.dim() != 1:
        raise RuntimeError("Input signal for noise variance must be 1D!")

    padding = kernel_size // 2  # padding needs to compensate for kernel size so signal has the same length
    # adding artificial batch dimension to allow using nn functions
    smoothed = F.avg_pool1d(signal.unsqueeze(dim=0), kernel_size=kernel_size, stride=1, padding=padding).squeeze()
    # estimate noise as difference between signal and smoothed signal
    noise = signal - smoothed
    return noise.var().item()
