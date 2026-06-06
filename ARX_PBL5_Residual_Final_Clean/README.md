# Hybrid ARX Residual Final Clean

Đây là bản sạch dùng để bảo vệ nếu chọn hướng **Hybrid ARX residual correction** làm sản phẩm chính.

Tên đúng:

```text
Hybrid ARX residual correction
```

Không gọi là ARX thuần, vì model cuối gồm:

```text
y_hybrid = y_arx_sim + shrink * residual_model(features)
```

## Kết quả chính

| Vai trò | Model | FIT_sim | RMSE_sim | Ghi chú |
|---|---|---:|---:|---|
| Nền so sánh | ARX backbone 16 input | 82.496 | 0.1681 | ARX thuần |
| Bản bảo vệ chính | Hybrid ARX residual | 83.433 | 0.1591 | ARX backbone + sửa sai số |

Model residual được chọn bằng validation:

```text
hgb_leaf15_l2_0.01 | shrink = 0.25
```

## Cấu trúc folder

```text
ARX_PBL5_Residual_Final_Clean/
  src/
    arx_backbone_pipeline.py
    arx_residual_experiment.py
    debug_walkthrough.py
  results/
    metrics.json
    leaderboard.csv
    SUMMARY.md
    test_predictions.csv
  docs/
    README.md
    ly_thuyet/
    huong_dan_build/
    du_lieu/
    bao_ve/
```

## Đọc theo mục tiêu

Mục lục tổng docs:

[docs/README.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/README.md>)

Nếu cần lý thuyết:

[docs/ly_thuyet/README.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/ly_thuyet/README.md>)

Nếu cần hướng dẫn build code từ đầu:

[docs/huong_dan_build/README.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/README.md>)

Nếu cần thu dữ liệu và train/test:

[docs/du_lieu/README.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/du_lieu/README.md>)

Nếu cần bảo vệ:

[docs/bao_ve/README.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/bao_ve/README.md>)

## Chạy lại

Chạy bản final:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py
```

Chạy log từng bước để học code:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\debug_walkthrough.py
```

Chạy với CSV thật:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py --data-csv .\du_lieu_that.csv
```

## Câu nói khi bảo vệ

```text
Nhóm chọn Hybrid ARX residual correction làm bản kết quả chính vì vẫn giữ ARX làm backbone tuyến tính, nhưng thêm một tầng học sai số để giảm sai lệch free-run. Sau khi bỏ biến protocol không thực tế Phase_identification, ARX backbone đạt FIT_sim 82.50, còn Hybrid residual được chọn bằng validation đạt 83.43. Nhóm không gọi đây là ARX thuần; tên đúng là Hybrid ARX residual correction.
```
