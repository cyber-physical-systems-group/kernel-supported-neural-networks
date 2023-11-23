import math
from typing import Literal

import lightning.pytorch as pl
import torch
from torch import Tensor

from .kernel import TorchKernelRegression
from .network.activations import bounded_linear


class HybridResidualModule(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Sequential,
        bandwidth: float,
        delta: float,
        lipschitz_constant: float,
        noise_variance: float,
        readout_init: Literal["zeros", "bounded", "default"],
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param network: torch Sequential neural network predicting bounded corrections for the non-parametric estimator
        :param bandwidth: kernel bandwidth for non-parametric estimator, for more details see `models/kernel/torch.py`
        :param delta: probability of the true value being in the bounds of the non-parametric estimator, required for
                      computing the bounds of the non-parametric estimator
        :param lipschitz_constant: Lipschitz constant of the function being estimated, required for computing the bounds
        :param noise_variance: variance of the noise in the data, required for computing the bounds
        :param readout_init: initialization scheme for the last layer of the network, can be:
                             `zeros` - setting all parameters to zeros
                             `default` - uniform initialization for last layer
                             `bounded` - setting parameters to random values from uniform distribution multiplied by
                                         the mean value of the bounds computed during adapt
        """
        super(HybridResidualModule, self).__init__()

        self.delta = delta
        self.lipschitz_constant = lipschitz_constant
        self.noise_variance = noise_variance
        self.readout_init = readout_init
        self.dtype = dtype

        self.network = network.to(dtype)
        self.estimator = TorchKernelRegression(bandwidth=bandwidth)

    def set_readout_parameters(self, scale: float) -> None:
        """Sets initial values for the readout (last layer) of the network."""
        stdv = 1.0 / math.sqrt(self.network[-1].weight.size(dim=1))
        torch.nn.init.uniform_(self.network[-1].weight, -scale * stdv, scale * stdv)

        if (bias := self.network[-1].bias) is not None:
            stdv = 1.0 / math.sqrt(bias.size(dim=0))
            torch.nn.init.uniform_(bias, -scale * stdv, scale * stdv)

    def adapt(self, non_parametric_x: torch.Tensor, non_parametric_y: torch.Tensor) -> None:
        """
        Stores the training data for the non-parametric estimator needed to generate predictions for the residual
        network training. The points should not be reused for training the two models.
        """
        non_parametric_x = non_parametric_x.to(self.dtype)
        non_parametric_y = non_parametric_y.to(self.dtype)

        # estimator itself has no parameters, so only dtype of training data needs to be changed
        self.estimator.fit(non_parametric_x, non_parametric_y)
        # compute bounds on training set to use for initialization gain
        _, bounds = self.estimator.predict(
            non_parametric_x,
            with_bounds=True,
            delta=self.delta,
            lipschitz_constant=self.lipschitz_constant,
            noise_variance=self.noise_variance,
        )

        # initialize readout layer - optionally using bounds
        if self.readout_init == "zeros":
            torch.nn.init.zeros_(self.network[-1].weight)
            torch.nn.init.zeros_(self.network[-1].bias)
        if self.readout_init == "bounded":
            scale = torch.std(bounds).numpy().item()
            self.set_readout_parameters(scale=scale)
        if self.readout_init == "default":
            self.set_readout_parameters(scale=1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        predictions, bounds = self.estimator.predict(
            inputs,
            with_bounds=True,
            delta=self.delta,
            lipschitz_constant=self.lipschitz_constant,
            noise_variance=self.noise_variance,
        )

        residuals = self.network(inputs)
        residuals = bounded_linear(residuals, bounds)
        return predictions + residuals


class TrainableHybridModel(pl.LightningModule):
    """
    Simple wrapper using lightning allowing easy training of the hybrid model.
    Predictions can be done by simply calling forward (ideally in no_grad context).
    """

    def __init__(self, model: HybridResidualModule):
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)  # type: ignore

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)  # type: ignore

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
