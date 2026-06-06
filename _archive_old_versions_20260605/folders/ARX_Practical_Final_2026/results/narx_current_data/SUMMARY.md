# Clean NARX Summary

All candidates use time-ordered split, train-only scaling, lagged inputs, and no future Soil_Moisture.

| Model | Val FIT_1 | Val FIT_12 | Val FIT_sim | Test FIT_1 | Test FIT_12 | Test FIT_sim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Delta_NNARX_MLP64_small_a10 | 89.325 | 69.107 | 66.199 | 88.885 | 67.950 | 64.674 |
| Delta_NNARX_MLP64_small_a01 | 89.852 | nan | 65.342 | 89.406 | nan | 64.317 |
| Delta_NNARX_MLP64_nomem_a01 | 89.935 | nan | 63.970 | 89.347 | nan | 61.098 |
| NNARX_MLP64_compact | 88.671 | nan | 41.415 | 88.349 | nan | 55.066 |

Selected by validation FIT_sim: `Delta_NNARX_MLP64_small_a10`.

Interpretation: one-step may be high because true output history is supplied at every step. Free-run `FIT_sim` is the strict deployment-style test.
