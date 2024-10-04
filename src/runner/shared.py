import os
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from numpy.typing import NDArray

from pydentification.data.datamodules.simulation import SimulationDataModule  # isort:skip
from pydentification.experiment.reporters import report_metrics, report_trainable_parameters  # isort:skip
from pydentification.metrics import regression_metrics  # isort:skip

from .plots import report_dynamic_prediction_plot, report_static_prediction_plot  # isort:skip  # noqa: F401


def input_fn(data_config: dict[str, Any], parameters: dict[str, Any]) -> pl.LightningDataModule:
    """
    Creates pl.LightningDataMo from data_config and training parameters

    :param data_config: static dataset values, such as path and test size
    :param parameters: dynamic training parameters, such as batch size or input and output lengths in samples

    :return: pl.LightningDataModule supporting selected training
    """
    return SimulationDataModule.from_csv(  # type: ignore
        dataset_path=data_config["path"],
        input_columns=data_config["input_columns"],
        output_columns=data_config["output_columns"],
        test_size=data_config["test_size"],
        batch_size=parameters["batch_size"],
        validation_size=parameters["validation_size"],
        shift=parameters["shift"],
        forward_input_window_size=parameters["n_input_time_steps"],
        forward_output_window_size=parameters["n_input_time_steps"],
        # always predict one-step ahead
        forward_output_mask=parameters["n_input_time_steps"] - parameters["n_output_time_steps"],
    )


def report_fn(model: pl.LightningModule, trainer: pl.Trainer, dm: pl.LightningDataModule):  # noqa: F811
    """
    Logs the experiment results to W&B

    :param model: trained model, needs to be BoundedSimulationTrainingModule instance
    :param trainer: ignored, needed for compatibility
    :param dm: datamodule used for training, needs to be SimulationDataModule instance
    """

    def _tensor_to_flat_array(tensor: torch.Tensor) -> NDArray:
        return tensor.detach().cpu().numpy().flatten()

    model.setup(stage="predict")
    predictions = model.predict_datamodule(dm, with_targets=True)

    dm.setup("predict")
    inputs = torch.cat([x for x, y in dm.test_dataloader()])

    inputs = _tensor_to_flat_array(inputs)

    y_true = _tensor_to_flat_array(predictions["targets"])
    y_pred_network = _tensor_to_flat_array(predictions["network_predictions"])
    y_pred_nonparametric = _tensor_to_flat_array(predictions["nonparametric_predictions"])
    lower_bound = _tensor_to_flat_array(predictions["lower_bound"])
    upper_bound = _tensor_to_flat_array(predictions["upper_bound"])

    report_metrics(regression_metrics(y_pred=y_pred_network, y_true=y_true), prefix="test/network")  # type: ignore
    report_metrics(regression_metrics(y_pred=lower_bound, y_true=y_true), prefix="test/lower_bound")  # type: ignore
    report_metrics(regression_metrics(y_pred=upper_bound, y_true=y_true), prefix="test/upper_bound")  # type: ignore

    # filter-out missing predictions, but report how many are missing
    mask = np.isnan(y_pred_nonparametric)
    wandb.log({"test/nonparametric/missing_predictions": mask.sum() / len(y_true)})
    report_metrics(regression_metrics(y_pred=y_pred_nonparametric[~mask], y_true=y_true[~mask]), prefix="test/nonparametric")  # type: ignore  # noqa: E501

    # store trainable parameters of neural-network
    report_trainable_parameters(model)

    # toggle between dynamic and static plots depending on the dataset used
    # we felt adding configuration for something changed twice during whole development is overkill
    report_static_prediction_plot(
        inputs=inputs,
        targets=y_true.flatten(),
        network_predictions=y_pred_network.flatten(),
        nonparametric_predictions=y_pred_nonparametric.flatten(),
        lower_bound=lower_bound.flatten(),
        upper_bound=upper_bound.flatten(),
    )

    # uncomment for nonlinear-dynamics and wiener-hammerstein datasets
    # report_dynamic_prediction_plot(
    #     targets=y_true.flatten(),
    #     network_predictions=y_pred.flatten(),
    #     nonparametric_predictions=y_pred_kre.flatten(),
    #     lower_bound=lower_bound.flatten(),
    #     upper_bound=upper_bound.flatten(),
    # )


def save_fn(name: str, model: pl.LightningModule):
    path = f"models/{name}/trained-model.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.network, path)  # save only neural-network to disk
    wandb.save(path)
