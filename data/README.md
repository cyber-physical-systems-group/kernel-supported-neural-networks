# Datasets

This directory contains datasets used in the paper and presentations related to it. Each dataset is stored as file
in `csv` directory.

| Dataset | Train Samples | Test Samples | Dynamics | Sigma | L   |
|---------|---------------|--------------|----------|-------|-----|
| TOY     | 1000          | 1000         | None     | 0.1   | 1.0 | 


### TOY

This dataset is toy dataset mostly for visual explanation of the algorithms used. It contains data for single function,
which is R^1 to R^1 function, defined as `sin(x) exp(-gamma x)`, where `gamma` is set to `0.1`. Samples on `x` axis
are samples from uniform distribution on interval `[0, 12pi]` and samples of `y` axis have additional noise with 
Gaussian distribution with standard deviation `0.05`. 

Dataset is stored in `csv/toy.csv` file with feature column `x` and target column `y`.
