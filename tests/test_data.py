import numpy as np
import pytest

from src.data import generate_data


@pytest.mark.parametrize(
    "nonlinear, limits, n_samples, noise_variance",
    [
        (
            lambda x: np.sin(x),
            np.array([[-1], [1]]),
            100,
            0.1,
        ),
        (
            lambda x1, x2: np.column_stack([np.sin(x1) * np.cos(x2), x1**2]),
            np.array([[-1, -1], [1, 1]]),
            100,
            0.5,
        ),
        (
            lambda x1, x2, x3: np.column_stack([np.sin(x1) * np.cos(x2), np.sin(x3), x1**2]),
            np.array([[-1, -1, -1], [1, 1, 1]]),
            100,
            1,
        ),
    ],
)
def test_generate_data(nonlinear, limits, n_samples, noise_variance):
    x, y = generate_data(nonlinear, limits, n_samples, noise_variance)

    assert x.shape == (n_samples, limits.shape[-1])
    assert y.shape == (n_samples, limits.shape[-1])
