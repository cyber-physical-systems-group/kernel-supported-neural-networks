import warnings
from functools import wraps
from numbers import Number
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn import metrics
from sklearn.utils import check_array, check_consistent_length

NumericSequence = Sequence[Number]
MetricCallable = Callable[[NumericSequence, NumericSequence], Number]


def validate_params(func: MetricCallable) -> MetricCallable:
    """
    Wrapper validating input parameters by checking array type and length consistency
    Shorthand for sklearn `validate_param` with default values used in this module
    """

    @wraps(func)
    def wrapper(y_true: NumericSequence, y_pred: NumericSequence, *args, **kwargs) -> Number | NumericSequence:
        y_true = check_array(y_true, ensure_2d=False)
        y_pred = check_array(y_pred, ensure_2d=False)
        check_consistent_length(y_true, y_pred)
        return func(y_true, y_pred, *args, **kwargs)  # type: ignore

    return wrapper


def non_zero_std(func: MetricCallable) -> MetricCallable:
    """
    Wrapper checking if y_true does not have zero standard deviation.
    Returns NaN if yes and runs metric computation normally otherwise
    """

    @wraps(func)
    def wrapper(y_true: NumericSequence, y_pred: NumericSequence, *args, **kwargs) -> Number | NumericSequence:
        if np.std(y_true) == 0:
            warnings.warn(
                "The standard deviation of the ground truth is zero."
                "The normalized mean squared error cannot be computed."
                "Returning nan instead."
            )
            return float("nan")
        return func(y_true, y_pred, *args, **kwargs)  # type: ignore

    return wrapper


def assemble_multioutput(func: MetricCallable) -> MetricCallable:
    """
    Assembles metrics for different multi-output aggregation modes.

    Supports three modes:
        - 'raw_values' returns a full set of scores in case of multi-output input.
        - 'uniform_average' scores of all outputs are averaged with uniform weight.
        - 'dimension_average' scores of each dimension (for multidimensional systems) are averaged.
    """

    @wraps(func)
    def wrapper(
        y_true: NDArray,
        y_pred: NDArray,
        multioutput: Literal["raw_values", "uniform_average", "dimension_average"],
        *args,
        **kwargs,
    ) -> Number | NDArray:
        if multioutput == "uniform_average":
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

        errors: NDArray = func(y_true, y_pred, *args, **kwargs)  # type: ignore
        if multioutput == "dimension_average":
            errors = np.mean(errors)

        return errors.item() if errors.size == 1 else errors

    return wrapper  # type: ignore


def normalized_mean_squared_error(
    y_true: NumericSequence,
    y_pred: NumericSequence,
    multioutput: Literal["raw_values", "uniform_average", "dimension_average"] = "raw_values",
) -> Number:
    """
    Computes MSE normalized by standard deviation of ground truth values.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs) with  ground truth (correct) target values
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs) with estimated target values
    :param multioutput: defines aggregation in the case of multiple output scores.
                        Can be one of the following strings (default is 'raw_values'):
                        - 'raw_values' returns a full set of scores in case of multi-output input.
                        - 'uniform_average' scores of all outputs are averaged with uniform weight.
                        - 'dimension_average' scores of each dimension (for multidimensional systems) are averaged.

    :return: normalized mean squared error as single numeric value or array of values for multivariate case
    """

    @validate_params
    @non_zero_std
    @assemble_multioutput
    def compute(y_true: NDArray, y_pred: NDArray) -> Number:
        return metrics.mean_squared_error(y_true, y_pred, multioutput="raw_values") / np.std(y_true, axis=0)

    return compute(y_true, y_pred, multioutput=multioutput)  # multioutput parameter is handled by wrapper


def normalized_root_mean_squared_error(
    y_true: NumericSequence,
    y_pred: NumericSequence,
    multioutput: Literal["raw_values", "uniform_average", "dimension_average"] = "raw_values",
) -> Number:
    """
    Computes RMSE normalized by standard deviation of ground truth values.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs) with  ground truth (correct) target values
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs) with estimated target values
    :param multioutput: defines aggregation in the case of multiple output scores.
                        Can be one of the following strings (default is 'raw_values'):
                        - 'raw_values' returns a full set of scores in case of multioutput input.
                        - 'uniform_average' scores of all outputs are averaged with uniform weight.
                        - 'dimension_average' scores of each dimension (for multi-dimensional systems) are averaged.

    :return: normalized root mean squared error as single numeric value or array of values for multivariate case
    """

    @validate_params
    @non_zero_std
    @assemble_multioutput
    def compute(y_true: NDArray, y_pred: NDArray) -> Number:
        errors = metrics.root_mean_squared_error(y_true, y_pred, multioutput="raw_values")
        return errors / np.std(y_true, axis=0)

    return compute(y_true, y_pred, multioutput=multioutput)  # multioutput parameter is handled by wrapper


def normalized_mean_absolute_error(
    y_true: NumericSequence,
    y_pred: NumericSequence,
    multioutput: Literal["raw_values", "uniform_average", "dimension_average"] = "raw_values",
) -> Number:
    """
    Computes MAE normalized by standard deviation of ground truth values.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs) with  ground truth (correct) target values
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs) with estimated target values
    :param multioutput: defines aggregation in the case of multiple output scores.
                        Can be one of the following strings (default is 'raw_values'):
                        - 'raw_values' returns a full set of scores in case of multioutput input.
                        - 'uniform_average' scores of all outputs are averaged with uniform weight.
                        - 'dimension_average' scores of each dimension (for multi-dimensional systems) are averaged.

    :return: normalized mean absolute error as single numeric value or array of values for multivariate case
    """

    @validate_params
    @non_zero_std
    @assemble_multioutput
    def compute(y_true: NDArray, y_pred: NDArray) -> Number:
        return metrics.mean_absolute_error(y_true, y_pred, multioutput="raw_values") / np.std(y_true, axis=0)

    return compute(y_true, y_pred, multioutput=multioutput)  # multioutput parameter is handled by wrapper


@validate_params
@non_zero_std
def normalized_max_error(
    y_true: NumericSequence,
    y_pred: NumericSequence,
) -> Number:
    """
    Computes MAE normalized by standard deviation of ground truth values.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs) with  ground truth (correct) target values
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs) with estimated target values

    :warning: does not support multioutput or multivariate arrays

    :return: normalized max error as single numeric value
    """
    return metrics.max_error(y_true.flatten(), y_pred.flatten()) / np.std(y_true.flatten(), axis=0)  # type: ignore


@validate_params
def regression_metrics(y_true: NumericSequence, y_pred: NumericSequence) -> dict[str, float]:
    """
    Computes multiple regression scores and returns a dictionary with results.

    :param y_true: array-like of shape (n_samples,) or (n_samples, n_outputs) with  ground truth (correct) target values
    :param y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs) with estimated target values

    :warning: assumes `uniform_average` for multi-output
    """
    multioutput = "uniform_average"

    return {
        "mean_squared_error": metrics.mean_squared_error(y_true, y_pred, multioutput=multioutput),
        "root_mean_squared_error": metrics.root_mean_squared_error(y_true, y_pred, multioutput=multioutput),
        "mean_absolute_error": metrics.mean_absolute_error(y_true, y_pred, multioutput=multioutput),
        "max_error": metrics.max_error(y_true.flatten(), y_pred.flatten()),
        # sequences are cast to arrays in `validate_params` wrapper
        "normalized_mean_squared_error": normalized_mean_squared_error(y_true, y_pred, multioutput=multioutput),
        "normalized_root_mean_squared_error": normalized_root_mean_squared_error(
            y_true, y_pred, multioutput=multioutput
        ),
        "normalized_mean_absolute_error": normalized_mean_absolute_error(y_true, y_pred, multioutput=multioutput),
        # max_error is aggregated differently
        "normalized_max_error": normalized_max_error(y_true, y_pred),
        "r2": metrics.r2_score(y_true, y_pred),  # type: ignore
        "true_mean": np.mean(y_true),
        "pred_mean": np.mean(y_pred),
        "true_std": np.std(y_true),
        "pred_std": np.std(y_pred),
    }  # type: ignore


def regression_report(
    y_true: NumericSequence,
    y_pred: NumericSequence,
    *,
    precision: int = 4,
) -> str:
    """
    Returns detailed regression report as string similar to classification report from sklearn.

    :param y_true: sequence of ground truth values for regression problem
    :param y_pred: sequence of values predicted by the model
    :param precision: precision to which metrics are expressed, defaults to 4

    :return: regression metrics and data statistics as single string
    """
    computed_metrics = regression_metrics(y_true=y_true, y_pred=y_pred)  # type: ignore

    header_width = 32  # distance from left to first column in chars
    header_offset = 2 if precision <= 4 else 0  # just to make it look nicer
    column_width = 12  # distance between columns in chars, will break if precision > 10

    report = ""
    # add report header
    report += f"{'':<{header_width - header_offset}}{'absolute':<{column_width}}{'normalized':<{column_width}}\n"
    # add metrics
    report += (
        f"{'mean_squared_error:':<{header_width}}"
        f"{computed_metrics['mean_squared_error']:<{column_width}.{precision}f}"
        f"{computed_metrics['normalized_mean_squared_error']:<{column_width}.{precision}f}\n"
    )

    report += (
        f"{'root_mean_squared_error:':<{header_width}}"
        f"{computed_metrics['root_mean_squared_error']:<{column_width}.{precision}f}"
        f"{computed_metrics['normalized_root_mean_squared_error']:<{column_width}.{precision}f}\n"
    )

    report += (
        f"{'mean_absolute_error:':<{header_width}}"
        f"{computed_metrics['mean_absolute_error']:<{column_width}.{precision}f}"
        f"{computed_metrics['normalized_mean_absolute_error']:<{column_width}.{precision}f}\n"
    )

    report += (
        f"{'max_error:':<{header_width}}"
        f"{computed_metrics['max_error']:<{column_width}.{precision}f}"
        f"{computed_metrics['normalized_max_error']:<{column_width}.{precision}f}\n"
    )
    # add single column for R2 score
    report += f"{'r2':<{header_width}}{'':<{column_width}}{computed_metrics['r2']:<{column_width}.{precision}f}\n"
    report += "\n\n"
    # add empty row for some space
    report += f"{'':<{header_width}}{'true':<{column_width}}{'pred':<{column_width}}\n"
    # add true and predicted distribution statistics
    report += f"{'mean:':<{header_width}}{computed_metrics['true_mean']:<{column_width}.{precision}f}"
    report += f"{computed_metrics['pred_mean']:<{column_width}.{precision}f}\n"
    report += f"{'std:':<{header_width}}{computed_metrics['true_std']:<{column_width}.{precision}f}"
    report += f"{computed_metrics['pred_std']:<{column_width}.{precision}f}\n"

    return report
