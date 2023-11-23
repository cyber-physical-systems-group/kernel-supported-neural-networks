from typing import Callable

import numpy as np
from numpy.typing import NDArray


def generate_data(
    nonlinear: Callable[[NDArray], NDArray], limits: NDArray, n_samples: int, noise_variance: float
) -> tuple[NDArray, NDArray]:
    """
    Simple function generating noised data for given nonlinear-static system, in limits with known noise variance.
    Supports 1D or ND static nonlinear systems

    :param nonlinear: function describing the nonlinear-static system
                      needs to return the same number of outputs as it has inputs
    :param limits: limits of the input space
    :param n_samples: number of samples to generate
    :param noise_variance: variance of the noise in the data
    """
    dims = limits.shape[-1]
    x = np.random.uniform(limits[0, :], limits[1, :], (n_samples, dims))
    noise = np.random.normal(0, noise_variance, size=[n_samples, dims])
    y = nonlinear(*np.split(x, dims, axis=-1)) + noise

    return x, y  # type: ignore
