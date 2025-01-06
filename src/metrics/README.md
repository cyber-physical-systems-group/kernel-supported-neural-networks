# Metrics

Package containing utilities for regression models, which are missing in `scikit-learn`. It also contains utilities for
evaluating multiple metrics at once. 

## Example
```python
import numpy as np
from src.metrics import regression_report

example_y_true = np.random.rand(10)
example_y_pred = example_y_true + 0.1 * np.random.rand(10)

print(regression_report(y_true=example_y_true, y_pred=example_y_pred, precision=2))
```

```text
                              absolute    normalized  
mean_squared_error:             0.00        0.01        
root_mean_squared_error:        0.06        0.24        
mean_absolute_error:            0.05        0.20        
max_error:                      0.09        0.39        
r2                                          0.94        


                                true        pred        
mean:                           0.53        0.48        
std:                            0.24        0.25        
```
