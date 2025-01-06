# isort:skip_file
# skip file from sorting due to use of try/except import
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import plotly.express as px  # noqa
import wandb


def _compose_prefix(prefix: str, key: str) -> str:
    """Composes prefix for WANDB log and given key"""
    return f"{prefix}/{key}" if bool(prefix) else key


def report_metrics(metrics: dict[str, float], prefix: str = ""):
    """
    :param metrics: dictionary of metrics computed for given experiment
    :param prefix: additional prefix to prepend to the metric key,
                   used, when two tests are conducted in single experiment
    """
    metrics = {_compose_prefix(prefix, key): value for key, value in metrics.items()}
    wandb.log(metrics)


def report_prediction_plot(targets: NDArray, predictions: NDArray, prefix: str = "", plot_error: bool = True):
    """
    :param targets: array of targets for given experiment test dataset
    :param predictions: array of model predictions
    :param prefix: additional prefix to prepend to the metric key,
                   used, when two tests are conducted in single experiment
    :param plot_error: if True, absolute error between targets and predictions will be added to plot
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Predictions and targets shaped must have the same! {targets.shape} != {predictions.shape}")

    for state_dim in range(targets.shape[-1]):  # multi-dimensional plot
        dim_targets = targets[:, :, state_dim].flatten()
        dim_predictions = predictions[:, :, state_dim].flatten()

        plot_data = {"targets": dim_targets, "predictions": dim_predictions}

        if plot_error:
            plot_data["error"] = np.abs(dim_targets - dim_predictions)

        figure = px.line(pd.DataFrame(plot_data))
        wandb.log({_compose_prefix(prefix, "prediction_plot"): figure})
