from abc import ABC, abstractmethod
from typing import Any

import lightning.pytorch as pl


class RuntimeContext(ABC):
    """
    This interface defined the runtime context needed for experiment execution by provided entrypoints.
    It can be used to define custom experiment execution flow.

    The interface can be implemented as module or namespace
    """

    @staticmethod
    @abstractmethod
    def input_fn(config: dict[str, Any], parameters: dict[str, Any]) -> pl.LightningDataModule:
        """
        :param config: static dataset configuration
        :param parameters: dynamic experiment configuration, for example delay-line length for dynamical systems

        :return: LightningDataModule instance, which is used to load and prepare data for training
        """
        ...

    @staticmethod
    @abstractmethod
    def model_fn(
        name: str, config: dict[str, Any], parameters: dict[str, Any], checkpoint_path: str | None = None
    ) -> tuple[pl.LightningModule, pl.Trainer]:
        """
        :param name: name of the W&B project, will be used for logging with callbacks
        :param config: static configuration, for example timeout, validation-size etc.
        :param parameters: dynamic experiment configuration, for example model settings or batch-size
        :param checkpoint_path: path to checkpoint for resuming training, can be None
        """
        ...

    @staticmethod
    @abstractmethod
    def train_fn(
        model: pl.LightningModule, trainer: pl.Trainer, dm: pl.LightningDataModule, checkpoint_path: str | None = None
    ) -> tuple[pl.LightningModule, pl.Trainer]:
        """
        :param model: LightningModule instance, returned from model_fn
        :param trainer: Trainer instance, returned from model_fn
        :param dm: LightningDataModule instance, returned from input_fn
        :param checkpoint_path: path to checkpoint for resuming training, can be None

        :return: trained model and trainer
        """
        ...

    @staticmethod
    @abstractmethod
    def report_fn(model: pl.LightningModule, trainer: pl.Trainer, dm: pl.LightningDataModule):
        """
        :param model: LightningModule instance, returned from train_fn (needs to be trained)
        :param trainer: Trainer instance, returned from train_fn, can be used for easier prediction
        :param dm: LightningDataModule instance, returned from input_fn, used for prediction on test data
        """
        ...

    @staticmethod
    @abstractmethod
    def save_fn(name: str, model: pl.LightningModule):
        """
        :param name: name of the run, returned from wandb.run.id or config
        :param model: trained LightningModule instance to be saved, returned from train_fn
        """
        ...
