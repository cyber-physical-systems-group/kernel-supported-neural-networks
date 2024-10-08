import math

import pytest
import torch

from src.nonparametric.estimators import noise_variance


@pytest.fixture(scope="module")
def random_seed():
    torch.manual_seed(42)


@pytest.mark.parametrize("variance", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
def test_noise_variance_estimator(variance: float, random_seed: int):
    # create signal with noise with given variance
    x = torch.linspace(-2 * math.pi, 2 * math.pi, 1000)
    signal = torch.sin(x) + torch.randn_like(x) * math.sqrt(variance)
    # allow 20% error with variance estimation and fixed kernel size
    estimated_variance = noise_variance(signal, kernel_size=11)
    assert math.isclose(estimated_variance, variance, rel_tol=0.2)
