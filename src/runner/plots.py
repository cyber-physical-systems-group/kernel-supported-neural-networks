import numpy as np
import plotly.graph_objects as go
import wandb
from numpy.typing import NDArray


def needs_1d_array(func: callable) -> callable:
    """Makes sure that all arguments, which are numpy arrays are 1D"""

    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, NDArray) and len(arg.shape) > 1:
                raise ValueError(f"Expected 1D array, got {arg.shape}")
        return func(*args, **kwargs)

    return wrapper


@needs_1d_array
def report_static_prediction_plot(
    inputs: NDArray,
    targets: NDArray,
    network_predictions: NDArray,
    nonparametric_predictions: NDArray,
    lower_bound: NDArray,
    upper_bound: NDArray,
) -> None:
    """
    Plots interactive prediction plot using plotly with network and nonparametric predictions and bounds.

    :param inputs: array with input variable, needs to be 1D
    :param targets: array with target variable, needs to be 1D
    :param network_predictions: array with neural network predictions, needs to be 1D
    :param nonparametric_predictions: array with kernel regression predictions, needs to be 1D
    :param lower_bound: array with lower bounds, needs to be 1D
    :param upper_bound: array with upper bounds, needs to be 1D
    """
    index = np.argsort(inputs)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=inputs[index], y=targets[index], mode="lines", name="targets"))
    figure.add_trace(
        go.Scatter(x=inputs[index], y=network_predictions[index], mode="lines", name="network predictions")
    )
    figure.add_trace(
        go.Scatter(x=inputs[index], y=nonparametric_predictions[index], mode="lines", name="nonparametric predictions")
    )

    figure.add_trace(
        go.Scatter(
            x=np.concatenate([inputs[index], inputs[index][::-1]]),
            y=np.concatenate([upper_bound[index], lower_bound[index][::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line_color="rgba(255,255,255,0)",
            name="bounds",
        )
    )

    wandb.log({"prediction_plot": figure})


@needs_1d_array
def report_dynamic_prediction_plot(
    targets: NDArray,
    network_predictions: NDArray,
    nonparametric_predictions: NDArray,
    lower_bound: NDArray,
    upper_bound: NDArray,
    time_delta: float = 1.0,
) -> None:
    """
    Plots interactive prediction plot using plotly with network and nonparametric predictions and bounds.

    :param targets: array with target variable, needs to be 1D
    :param network_predictions: array with neural network predictions, needs to be 1D
    :param nonparametric_predictions: array with kernel regression predictions, needs to be 1D
    :param lower_bound: array with lower bounds, needs to be 1D
    :param upper_bound: array with upper bounds, needs to be 1D
    :param time_delta: time delta between samples, defaults to 1.0
    """
    t = np.arange(len(targets)) * time_delta

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=t, y=targets, mode="lines", name="targets"))
    figure.add_trace(go.Scatter(x=t, y=network_predictions, mode="lines", name="network predictions"))
    figure.add_trace(go.Scatter(x=t, y=nonparametric_predictions, mode="lines", name="nonparametric predictions"))

    figure.add_trace(
        go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line_color="rgba(255,255,255,0)",
            name="bounds",
        )
    )

    wandb.log({"test/prediction_plot": figure})
