import torch
import lightning.pytorch as pl
from torch import Tensor

from .kernel import TorchNadarayaWatsonEstimator
from .network.activations import bounded_linear


class HybridResidualModule(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Sequential,
        bandwidth: float,
        delta: float,
        lipschitz_constant: float,
        noise_variance: float,
        residual_zero_init: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param network: torch Sequential neural network predicting bounded corrections for the non-parametric estimator
        :param bandwidth: kernel bandwidth for non-parametric estimator, for more details see `models/kernel/torch.py`
        :param delta: probability of the true value being in the bounds of the non-parametric estimator, required for
                      computing the bounds of the non-parametric estimator
        :param lipschitz_constant: Lipschitz constant of the function being estimated, required for computing the bounds
        :param noise_variance: variance of the noise in the data, required for computing the bounds
        :param residual_zero_init: if True, the last layer of the network is initialized to zero
        """
        super(HybridResidualModule, self).__init__()

        self.delta = delta
        self.lipschitz_constant = lipschitz_constant
        self.noise_variance = noise_variance
        self.dtype = dtype

        self.network = network.to(dtype)
        self.estimator = TorchNadarayaWatsonEstimator(bandwidth=bandwidth)

        if residual_zero_init:
            for parameter in self.network[-1].parameters():
                torch.nn.init.zeros_(parameter)

    def adapt(self, non_parametric_x: torch.Tensor, non_parametric_y: torch.Tensor) -> None:
        """
        Stores the training data for the non-parametric estimator needed to generate predictions for the residual
        network training. The points should not be reused for training the two models.
        """
        non_parametric_x = non_parametric_x.to(self.dtype)
        non_parametric_y = non_parametric_y.to(self.dtype)
        # estimator itself has no parameters, so only dtype of training data needs to be changed
        self.estimator.fit(non_parametric_x, non_parametric_y)

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
