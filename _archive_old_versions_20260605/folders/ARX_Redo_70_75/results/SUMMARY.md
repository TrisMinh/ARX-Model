# Summary

Selected shrink: `1.0`

| Model | Val FIT_sim | Test FIT_sim | Test RMSE | Test Bias |
| --- | ---: | ---: | ---: | ---: |
| ARX backbone | 68.788 | 66.492 | 0.9760 | 0.0079 |
| Conservative Hybrid ARX Residual | 71.562 | 70.294 | 0.8653 | 0.0111 |

Validation is used for shrink selection. Test is reported once after selection.
Residual features use ARX simulated trajectory and input lags `[2, 3, 6, 12, 24]` only.
