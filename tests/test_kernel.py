import numpy as np
import pandas as pd
import pytest

from src.models.kernel import NadarayaWatsonEstimator, box_kernel

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


def test_kernel_model(train_data, test_data):
    estimator = NadarayaWatsonEstimator(kernel=box_kernel, bandwidth=KERNEL_BANDWIDTH)

    # train non-parametric estimator
    train_x = np.expand_dims(train_data["train_x"].values, axis=-1)
    train_y = np.expand_dims(train_data["train_y"].values, axis=-1)
    estimator.fit(train_x, train_y)

    # predict
    test_x = np.expand_dims(test_data["test_x"], axis=-1)
    predictions, bounds = estimator.predict(
        test_x, with_bounds=True, delta=DELTA, lipschitz_constant=L, noise_variance=NOISE_VARIANCE
    )

    # flatten - tested system is 1D
    predictions = predictions.flatten()
    bounds = bounds.flatten()
    lower_bound = predictions - bounds
    upper_bound = predictions + bounds

    # assert with 6 decimal places of precision
    np.testing.assert_allclose(predictions, test_data["y_pred"].values, atol=1e-6)
    np.testing.assert_allclose(lower_bound, test_data["y_lower_bound"].values, atol=1e-6)
    np.testing.assert_allclose(upper_bound, test_data["y_upper_bound"].values, atol=1e-6)
