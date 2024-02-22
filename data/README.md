# Datasets

This directory contains datasets used in the paper and presentations related to it. Each dataset is stored as file
in `csv` directory.

| Dataset | Train Samples | Test Samples | Dynamics  | Sigma | L   |
|---------|---------------|--------------|-----------|-------|-----|
| TOY     | 1000          | 1000         | None      | 0.05  | 1   |
| STATIC  | 10 000        | 10 000       | None      | 0.1   | 1/4 |
| DYNAMIC | 50 000        | 50 000       | Nonlinear | 0.05  | 1   |

### TOY

This dataset is toy dataset mostly for visual explanation of the algorithms used. It contains data for single function,
which is R^1 to R^1 function, defined as `sin(x) exp(-gamma x)`, where `gamma` is set to `0.1`. Samples on `x` axis
are samples from uniform distribution on interval `[0, 12pi]` and samples of `y` axis have additional noise with 
Gaussian distribution with standard deviation `0.05`. 

Dataset is stored in `csv/toy.csv` file with feature column `x` and target column `y`.

### STATIC

This dataset represents high dimensional static system, R^8 to R^1. The function is defined as sum of 8 Gaussian density
functions (not distribution, but explicit PDF) with different C (given in table below). The inputs are samples uniformly
on range [-1, 1]. L is 1/4 for all Gaussian functions, so for its sum as well. Noise was added with sigma = 0.1.

| C1 | 1.42  |
| C2 | -1.32 |
| C3 | -1.65 |
| C4 | -1.04 |
| C5 | 0.61  |
| C6 | 1.98  |
| C7 | 0.31  |
| C8 | -0.31 |

### DYNAMIC

This dataset represents low dimensional dynamic system, R^1 to R^1 with nonlinear dynamics. The function is defined as
`x' = -k \sigma(x-1) u(t)`, where `sigma` is sigmoid function, `k` is constant equal to 0.9 and `u(t)` is input signal,
which was generated as sine wave. The system was numerically integrated and is stored in CSV files with 3 columns:
"t" for time, "u" for inputs and "y" for noised outputs. Initial condition was 1, time is set to 100 000 samples from 0
to 100 seconds and forcing was given as `sin(pi/5 t)`.
