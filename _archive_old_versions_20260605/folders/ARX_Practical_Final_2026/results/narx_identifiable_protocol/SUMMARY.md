# End-to-End NARX 75 Summary

Dataset moi duoc sinh voi actuator schedule doc lap voi Soil_Moisture, co persistent excitation.

| Metric | Validation | Test |
| --- | ---: | ---: |
| FIT_1step | 97.006 | 85.894 |
| FIT_12 | 92.535 | 82.205 |
| FIT_sim | 86.453 | 76.295 |
| RMSE_sim | 0.3057 | 0.3306 |

Selected by robust validation score: `delta_nnarx_mlp64_a01_rs1`.
Robust validation score: `81.106`.
Target met: `True`.

Important: ket qua nay dat tren dataset moi co excitation dung nguyen tac. Khong nen tron voi benchmark data cu, noi ma clean NARX free-run bi drift.
