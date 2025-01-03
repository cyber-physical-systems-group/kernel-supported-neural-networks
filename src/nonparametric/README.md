# Kernel Regression

Kernel regression is a non-parametric model, which can be used to estimate static-nonlinear functions and SISO dynamical
system with finite memory, by converting inputs using delay-line. This implementation is using only PyTorch, and it can
handle large dataset sizes with high dimensions by using `MemoryManagers`.

# Functional

## Kernel Regression

The submodule with implementation of the main kernel regression estimator functionality. It is using PyTorch tensors
and does not hold state. To use the function run following code:

```python
import torch

from pydentification.models.nonparametric.functional import kernel_regression
from pydentification.models.nonparametric.kernels import box_kernel


x = torch.rand(100, 2)
y = f(x)  # assume f exists and is SISO nonlinear function
test_x = torch.rand(100, 2)

y_hat = kernel_regression(memory=x, targets=y, points=test_x, kernel=box_kernel, bandwidth=0.1)
```

## Bounds

Bounds estimator is implemented using equations defined in [1]. It requires returning the kernel density tensor from
kernel regression. To compute bounds following characteristics of the estimated function need to be known:
* Lipschitz constant of the nonlinear function [2]
* Variance of the noise in measurements
* Probability of the true function being in bounds, given as 1 - \delta (delta is the constant set by the user)

To use the function run following code:

```python
import torch

from pydentification.models.nonparametric.functional import kernel_regression, kernel_regression_bounds
from pydentification.models.nonparametric.kernels import box_kernel


x = torch.rand(100, 2)
y = f(x)  # assume f exists and is SISO nonlinear function
test_x = torch.rand(100, 2)

L = 1  # Lipschitz constant is known for f
sigma = 0.1  # Variance of the noise in measurements
delta = 0.1  # Probability of the true function being in bounds

y_hat, kernel_density = kernel_regression(
    memory=x,
    targets=y,
    points=test_x,
    kernel=box_kernel,
    bandwidth=0.1,
    return_kernel_density=True
)

bounds = kernel_regression_bounds(
    kernel_density,
    dim=1,  # dimension of the function f
    bandwidth=0.1,  # the same as in kernel_regression
    delta=delta,
    lipschitz_constant=L,
    noise_variance=sigma,
)
```

## Kernels

Kernel functions are implemented as stand-alone function of single variable, which any dimension. All typically used
kernels are added to the package, for more information see [3].

*Note*: Kernels need to have compact carrier to be used with kernel regression bound computation, otherwise the theory
does not hold.

# Memory

Memory managers are defined in the package `memory` and they are used to query large datasets in batches for
non-parametric models. For use cases, where delay-line is used with relatively high length (over 100 samples), the 
kernel regression estimator needs to compute all pair-wise distances for the known points and its input batch. This can
be very memory and time-consuming, especially for large datasets. 

Memory managers use ANN algorithms (Approximate Nearest Neighbors) to query the known dataset (called memory) for pairs
of inputs and corresponding outputs, which are closest to points, in which function is to be estimated in the input 
space. This allows to reduce the number of computations and memory usage, while still providing good estimation quality,
since when `k_neighbours` or `radius` are set correctly, points not returned in memory, would be further away then
`bandwidth`, hence do not contribute to the estimation.

*Note*: Kernel regression bounds can still be used with memory managers. They are computed point-wise and depend on the
kernel density, which is still computed. When memory manager returns all points, which are closer than bandwidth for
each query, the bounds will be exactly the same as when using the full memory, in case some points are missed, due to
heuristic nature of ANN or setting too low `k_neighbours` or `radius`, the bounds will be wider then with full memory.

To use the memory manager with kernel regression, run following code:

```python
from itertools import batched

import torch

from src.nonparametric.functional import kernel_regression
from src.nonparametric.kernels import box_kernel
from src.nonparametric.memory import NNDescentMemoryManager


x = torch.rand(10_000, 100)  # very large dataset with high dimension
y = f(x)  # assume f exists and is SISO nonlinear function
test_x = torch.rand(50_000, 100)

L = 1  # Lipschitz constant is known for f
sigma = 0.1  # Variance of the noise in measurements
delta = 0.1  # Probability of the true function being in bounds

memory_manager = NNDescentMemoryManager(memory=x, targets=y)  # k is the number of neighbors to return

for batch in batched(test_x, batch_size=100):
    memory, targets = memory_manager.query_nearest(batch, k=10)
    y_hat = kernel_regression(memory=memory, targets=targets, points=batch, kernel=box_kernel, bandwidth=0.1)
```

Currently implemented memory managers:
* `ExactMemoryManager` - uses `torch.cdist` to query the memory, efficient for small datasets and low dimensions (can be run directly on GPU)
* `FAISSMemoryManager` - uses FAISS library to query the memory [4]
* `NNDescentMemoryManager` - uses NNDescent algorithm to query the memory [5]
* `SklearnMemoryManager` - uses sklearn library to query the memory, supporting all the algorithms from `sklearn.neighbors`

*Note*: We use `pip` to install dependencies and FAISS currently only supports `conda` installation, so it is not
included in automatic installing and by-extension testing. To use it, it is recommended to install manually from source,
using release newer than 1.7.4 (tested on python 3.10). FAISS is not used for any examples, as it does not yield better
results, but can significantly improve performance for really large datasets. We provided the implementation, as we used
it to experiment, but no results reported in the paper are using the FAISS memory manager.

For more information see: https://github.com/facebookresearch/faiss/wiki/Installing-Faiss.

# References

<a id="1">[1]</a> 
Paweł Wachel and Krzysztof Kowalczyk and Cristian R. Rojas (2023)
*Decentralized diffusion-based learning under non-parametric limited prior knowledge*
https://arxiv.org/abs/2305.03295

<a id="2">[2]</a>
Wikipedia (2023)
*Lipschitz continuity*
https://en.wikipedia.org/wiki/Lipschitz_continuity

<a id="3">[3]</a>
Wikipedia (2023)
*Kernel (statistics)*
https://en.wikipedia.org/wiki/Kernel_(statistics)

<a id="4">[4]</a>
Jeff Johnson and Matthijs Douze and Hervé Jégou (2017)
*Billion-scale similarity search with GPUs*
https://arxiv.org/abs/1702.08734

<a id="4">[5]</a>
Wei Dong and Moses Charikar and Kai Li (2011)
*Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures*
https://www.cs.princeton.edu/cass/papers/www11.pdf
