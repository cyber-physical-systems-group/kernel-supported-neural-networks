from typing import Any

import lightning.pytorch as pl
import torch
import wandb

from pydentification.models.modules.feedforward import TimeSeriesLinear  # isort:skip
from pydentification.models.nonparametric import kernels, memory  # isort:skip
from pydentification.training.bounded.module import BoundedSimulationTrainingModule  # isort:skip

# activation and kernel mappings are needed to access functional implementations using config
KERNELS = {  # only compact carrier kernels
    "box_kernel": kernels.box_kernel,
    "epanechnikov_kernel": kernels.epanechnikov_kernel,
    "triangular_kernel": kernels.triangular_kernel,
    "quartic_kernel": kernels.quartic_kernel,
    "triweight_kernel": kernels.triweight_kernel,
    "tricube_kernel": kernels.tricube_kernel,
    "cosine_kernel": kernels.cosine_kernel,
}

ACTIVATIONS = {
    "leaky_relu": torch.nn.LeakyReLU,
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
}


def network_fn(config: dict[str, Any]) -> torch.nn.Module:
    layers = []

    layers.append(
        TimeSeriesLinear(
            n_input_time_steps=config["n_input_time_steps"],
            n_output_time_steps=config["n_hidden_time_steps"],
            n_input_state_variables=config["n_input_state_variables"],
            n_output_state_variables=config["n_hidden_state_variables"],
        )
    )

    layers.append(ACTIVATIONS[config["activation"]]())

    for _ in range(config["n_hidden_layers"]):
        layers.append(
            TimeSeriesLinear(
                n_input_time_steps=config["n_hidden_time_steps"],
                n_output_time_steps=config["n_hidden_time_steps"],
                n_input_state_variables=config["n_hidden_state_variables"],
                n_output_state_variables=config["n_hidden_state_variables"],
            )
        )

        layers.append(ACTIVATIONS[config["activation"]]())

    layers.append(
        TimeSeriesLinear(
            n_input_time_steps=config["n_hidden_time_steps"],
            n_output_time_steps=config["n_output_time_steps"],
            n_input_state_variables=config["n_hidden_state_variables"],
            n_output_state_variables=config["n_output_state_variables"],
        )
    )

    return torch.nn.Sequential(*layers)


def model_fn(
    project_name: str, config: dict[str, Any], parameters: dict[str, Any]
) -> tuple[pl.LightningModule, pl.Trainer]:
    # config is static config and parameters is changing config during sweep
    # for single run config is training config and parameters contains model settings
    network = network_fn(parameters)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)  # fixed initial LR
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config["lr_patience"], verbose=True)

    timer = pl.callbacks.Timer(duration=config["timeout"], interval="epoch")
    stopping = pl.callbacks.EarlyStopping(
        monitor="training/validation_loss", patience=config["patience"], mode="min", verbose=True
    )

    path = f"models/{wandb.run.id}"
    epoch_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=path, monitor="training/validation_loss", every_n_epochs=10)

    model = BoundedSimulationTrainingModule(
        network=network,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        bound_during_training=config["bound_during_training"],
        bound_crossing_penalty=parameters["bound_crossing_penalty"],  # sweep parameter
        # memory manager is not configuration parameter, toggle manually depending on the dataset and hardware
        memory_manager=memory.ExactMemoryManager(),
        bandwidth=parameters["bandwidth"],
        kernel=KERNELS[parameters["kernel"]],
        lipschitz_constant=parameters["lipschitz_constant"],
        delta=parameters["delta"],
        noise_variance=parameters["noise_variance"],
        k=parameters["k"],
        p=2,
        # toggle devices manually depending on the available hardware
        memory_device="cpu",  # default is CPU
        predict_device="cpu",
    )

    trainer = pl.Trainer(
        max_epochs=config["n_epochs"],
        default_root_dir=path,
        callbacks=[timer, stopping, epoch_checkpoint],
        logger=pl.loggers.WandbLogger(project=project_name),
    )

    return model, trainer
