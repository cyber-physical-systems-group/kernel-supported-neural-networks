import numpy as np
import pandas as pd
import pytest
import torch

from src.models.kernel import NumpyKernelRegression, TorchKernelRegression

L = 2  # Lipschitz constant
NOISE_VARIANCE = 0.05
KERNEL_BANDWIDTH = 0.1
DELTA = 0.01  # probability 1 - delta


@pytest.fixture
def train_data():
    return pd.read_csv("tests/resources/train.csv")


@pytest.fixture
def test_data():
    return pd.read_csv("tests/resources/test.csv")


@pytest.mark.parametrize(
    "estimator_cls, forward_cast_fn, backward_cast_fn",
    [
        (NumpyKernelRegression, lambda x: np.expand_dims(x, axis=-1), lambda x: x),
        (TorchKernelRegression, lambda x: torch.unsqueeze(torch.from_numpy(x), dim=-1), lambda x: x.numpy()),
    ],
)
def test_kernel_model(train_data, test_data, estimator_cls, forward_cast_fn: callable, backward_cast_fn: callable):
    """
    Test covers numpy and torch implementations of kernel regression.

    :param estimator_cls: class of estimator to test
    :param forward_cast_fn: function to cast data to type expected by estimator from pandas DataFrame to Tensor or array
    :param backward_cast_fn: function to cast data back from type returned by estimator to numpy
    """
    estimator = estimator_cls(bandwidth=KERNEL_BANDWIDTH)

    # train non-parametric estimator
    train_x = forward_cast_fn(train_data["train_x"].values)
    train_y = forward_cast_fn(train_data["train_y"].values)
    estimator.fit(train_x, train_y)

    # predict
    test_x = forward_cast_fn(test_data["test_x"].values)
    predictions, bounds = estimator.predict(
        test_x, with_bounds=True, delta=DELTA, lipschitz_constant=L, noise_variance=NOISE_VARIANCE
    )

    predictions = backward_cast_fn(predictions)  # cast back to numpy (if torch is used)
    bounds = backward_cast_fn(bounds)

    # flatten - tested system is 1D
    predictions = predictions.flatten()
    bounds = bounds.flatten()
    lower_bound = predictions - bounds
    upper_bound = predictions + bounds

    # assert with 6 decimal places of precision
    np.testing.assert_allclose(predictions, test_data["y_pred"].values, atol=1e-6)
    np.testing.assert_allclose(lower_bound, test_data["y_lower_bound"].values, atol=1e-6)
    np.testing.assert_allclose(upper_bound, test_data["y_upper_bound"].values, atol=1e-6)
