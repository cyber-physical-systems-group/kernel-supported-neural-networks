import torch
from torch import Tensor


def bounded_linear(inputs: Tensor, bounds: float | Tensor, inplace: bool = False) -> Tensor:
    """
    Bounded linear activation function. It means that the output is linear in range [-bounds, bounds] and clamped
    outside of it to the values of the bounds. Bounds can be scalar of tensor of the same shape as inputs.
    """
    out = inputs if inplace else None
    return torch.clamp(inputs, min=-bounds, max=bounds, out=out)
