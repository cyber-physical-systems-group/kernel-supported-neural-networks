# Bounded Training

This package contains training module for neural network to identify nonlinear dynamical systems or static nonlinear
functions with guarantees by using bounded activation incorporating theoretical bounds from the kernel regression model.
The approach is limited to finite memory single-input single-output dynamical systems, which can be converted to
static multiple-input single-output systems by using delay-line. Bounds are computed using kernel regression working
with the same data, but we are able to guarantees of the estimation, which are used to activate a network during and
after training, in order to ensure that the predictions are never outside of those theoretical bounds.

Kernel regression, needs to compute distance from the point where the function is estimated to all the points in the
known dataset (explanatory data), which can lead to extreme memory consumption and slow computation. To mitigate this
issue, we use approximated nearest neighbours algorithm to select memory dynamically from entire known data. Bounds are
computed using those fetched points only, so the nearest neighbour algorithm does not affect the certainty.

## Dependencies

The implementation of hybrid training is based on `pydentification` packages in models, namely: `nonparametric`,
`modules.activations` and `modules.losses`. 

## Algorithm

The goal of bounded training is using kernel regression with bounds [1] to transfer them to neural network. This is done
using bounded activation, which is defined in `pydentification.modules.activations`. The activation is used in such a 
way, that network predictions will never go outside of those bounds. The details of the algorithm are described in our 
paper [2].

## Features

* Adding bounds to pre-trained neural network (use `from_pretrained` constructor)
* Training neural network with bounded activation and using bounds to improve convergence (use `bound_crossing_penalty > 0`)
* Training neural network with bounded activation and using bounds to re-initialize the network multiple times until the initialized in inside the bounds (use `max_reinit > 0` and `0 < reinit_relative_tolerance` < 1).

## Static Systems

The algorithm supports static systems as well, including multi-input single-output systems (MISO). This requires passing
tensors with shape `(BATCH, 1, SYSTEM_DIM)`, which are converted to `(BATCH, SYSTEM_DIM)` before passing to the network.
For static SISO or single-step dynamical systems, this does not make any difference. 

## Examples

Usage with default parameters is given in code example below.

To use with bounded activation. For zero-mean datasets,
the network is likely to be in bounds from start, so both `bound_crossing_penalty` and `max_reinit` might have little
effect. To run bounds crossing penalty, set the `bound_crossing_penalty` to some positive value and make sure that
`bound_during_training` is `False, since when it is set to `True`, the bounds are used to activate the network during
training, and it never goes outside of them, so bounds crossing penalty is never added.

To use with reinit, set `max_reinit` to some positive value and `reinit_relative_tolerance` to some value between 0 and 
1, parameter `reinit_relative_tolerance` is used to quantify the threshold for reinitialization, which is counted as the
ration of elements in the batch outside of bounds. Setting it to 1 would result in reinitialization only when all the
elements in the batch are outside of bounds, while setting it to 0 would result in reinitialization when any of elements
is outside of bounds. The `max_reinit` parameter is used to limit the number of reinitializations.

```python
import torch

from training import BoundedSimulationTrainingModule
from nonparametric import kernels, memory


model = BoundedSimulationTrainingModule(
    network=network,  # assume network exists and it is torch.nn.Module
    optimizer=torch.optim.Adam(network.parameters()),  # needs to be setup for network
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9),  # needs to be setup for optimizer
    memory_manager=memory.ExactMemoryManager(),
    bandwidth=0.1,
    kernel=kernels.box_kernel,  # and compact kernel function can be used
    lipschitz_constant=1,  # needs to be known
    delta=0.9,
    k=10,  # memory size for each kernel
)

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule=dm)  # assume datamodule exists and it is pydentification.datamodules.SimulationDataModule
```

It is possible to use GPU or other accelerators with `BoundedSimulationTrainingModule`, while keeping the memory on 
different or the same device and moving the model for prediction to any device. By default `lightning` trainer moves
the model back to CPU after training.

Note, that not all memory manager support GPU (`NNDescentMemoryManager` does not support GPU due to underlying library
implementation). Also note, that memory and predict device needs to be the same, when using different than CPU.

```python
import torch

from training import BoundedSimulationTrainingModule
from nonparametric import kernels, memory


model = BoundedSimulationTrainingModule(
    network=network,  # assume network exists and it is torch.nn.Module
    optimizer=torch.optim.Adam(network.parameters()),  # needs to be setup for network
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9),  # needs to be setup for optimizer
    memory_manager=memory.ExactMemoryManager(),
    bandwidth=0.1,
    kernel=kernels.box_kernel,  # and compact kernel function can be used
    lipschitz_constant=1,  # needs to be known
    delta=0.9,
    k=10,  # memory size for each kernel
    memory_device="cuda",  # move memory manager to CUDA for training
    predict_device="cuda",  # move prediction to CUDA for prediction
)

trainer = pl.Trainer(max_epochs=10, accelerator="gpu", gpus=1)
trainer.fit(model, datamodule=dm)  # assume datamodule exists and it is pydentification.datamodules.SimulationDataModule
```

### Implementation Details

The `BoundedSimulationTrainingModule` is meant to be used with `pl.Trainer` from `pytorch-lightning` package, however
not all options can be applied. Only `SingleDeviceStrategy` is supported, to check it, use following snippet, assuming
the call is made inside `pl.Trainer` (after passing the model to the trainer).

```python
if type(self.trainer.strategy).__name__ != "SingleDeviceStrategy": 
    raise RuntimeError(f"{self.__class__.__name__} can only be used with single device strategy!")
```

#### Memory Manager

When using `NNDescentMemoryManager`, sometimes internal issue pops up, with unclear message, which is given below. This
means that `epsilon` parameter was not given to call of pynndescent algorithm. It can be solved by passing
`memory_epsilon` set to some value (see docs https://pynndescent.readthedocs.io/en/latest/).

```log
TypingError: Failed in nopython mode pipeline (step: nopython frontend)
No implementation of function Function(<built-in function add>) found for signature:
 
 >>> add(float64, none)
 
There are 20 candidate implementations:
   - Of which 18 did not match due to:
   Overload of function 'add': File: <numerous>: Line N/A.
     With argument(s): '(float64, none)':
    No match.
   - Of which 2 did not match due to:
   Operator Overload in function 'add': File: unknown: Line unknown.
     With argument(s): '(float64, none)':
    No match for registered cases:
     * (int64, int64) -> int64
     * (int64, uint64) -> int64
     * (uint64, int64) -> int64
     * (uint64, uint64) -> uint64
     * (float32, float32) -> float32
     * (float64, float64) -> float64
     * (complex64, complex64) -> complex64
     * (complex128, complex128) -> complex128

During: typing of intrinsic-call at /mnt/c/Users/kzaja/Documents/Research/Github/pydentification/venv/lib/python3.10/site-packages/pynndescent/pynndescent_.py (1281)

File "venv/lib/python3.10/site-packages/pynndescent/pynndescent_.py", line 1281:
        def search_closure(query_points, k, epsilon, visited, rng_state):
            <source elided>
            result = make_heap(query_points.shape[0], k)
            distance_scale = 1.0 + epsilon
```

## References

<a id="1">[1]</a> 
Paweł Wachel and Krzysztof Kowalczyk and Cristian R. Rojas (2023)
*Decentralized diffusion-based learning under non-parametric limited prior knowledge*
https://arxiv.org/abs/2305.03295

<a id="2">[2]</a> 
Krzysztof Zając and Krzysztof Kowalczyk and Paweł Wachel (2024)
*Kernel-Supported Neural Modeling of Nonlinear Systems*
TBA
