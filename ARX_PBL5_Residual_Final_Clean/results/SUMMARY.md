# Hybrid ARX Residual Final

Bản này dùng ARX 16 input làm backbone và thêm residual correction. Đây là bản bảo vệ chính nếu chọn hướng Hybrid ARX residual.

## Nguyên tắc không leakage

- Residual chỉ train trên train split.
- Chọn residual model và shrink bằng validation robust score.
- Test chỉ dùng để báo cáo cuối.
- Feature residual không dùng `Soil_Moisture` thật tương lai; chỉ dùng quỹ đạo `y_arx_sim` và input lag quá khứ.
- `shrink=0` nằm trong grid, nên nếu residual không giúp thì pipeline có quyền chọn không sửa gì.

## Kết quả

| Model | Test FIT_sim | Test RMSE | Test Bias |
| --- | ---: | ---: | ---: |
| ARX backbone | 82.496 | 0.1681 | -0.0381 |
| Hybrid selected | 83.433 | 0.1591 | -0.0207 |

- Candidate selected by validation: `hgb_leaf15_l2_0.01|0.25`.
- Test gain FIT_sim: `0.937` điểm.

Kết luận: residual có cải thiện thật trên test. Bản bảo vệ chính nên gọi đúng là Hybrid ARX residual correction, không gọi là ARX thuần.
