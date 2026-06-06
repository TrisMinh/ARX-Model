# Mục lục và cách bảo vệ

## 1. Bản nào là bản chính?

Bản chính trong folder này là:

```text
Hybrid ARX residual correction
```

Nó dùng ARX làm backbone, sau đó thêm tầng residual để sửa sai số free-run.

Không nói:

```text
Đây là ARX thuần.
```

Nói đúng:

```text
Đây là mô hình lai: ARX backbone + residual correction.
```

## 2. Vì sao dùng residual làm bản bảo vệ?

Vì trên cùng protocol:

| Model | FIT_sim | RMSE_sim |
|---|---:|---:|
| ARX backbone | 82.496 | 0.1681 |
| Hybrid residual | 83.433 | 0.1591 |

Residual cải thiện free-run simulation, tức phần khó hơn one-step prediction. Đây là điểm có ý nghĩa vì mô hình dự đoán nhiều bước mới hữu ích cho điều khiển/MPC.

## 3. File cần đọc

Nếu bạn mới đọc:

1. [07_THUAT_NGU_CHO_NGUOI_MOI.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/ly_thuyet/07_THUAT_NGU_CHO_NGUOI_MOI.md>)
2. [01_LY_THUYET_HYBRID_ARX_RESIDUAL.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/ly_thuyet/01_LY_THUYET_HYBRID_ARX_RESIDUAL.md>)

Nếu bạn chuẩn bị bảo vệ:

1. [04_KET_QUA_TU_PHAN_BIEN.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/bao_ve/04_KET_QUA_TU_PHAN_BIEN.md>)
2. [05_CAU_HOI_BAO_VE_NHANH.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/bao_ve/05_CAU_HOI_BAO_VE_NHANH.md>)
3. [06_SO_SANH_ARX_THUAN_NARX_NNARX_MPC.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/ly_thuyet/06_SO_SANH_ARX_THUAN_NARX_NNARX_MPC.md>)
4. [08_DE_CUONG_BAO_CAO_VA_SLIDE.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/bao_ve/08_DE_CUONG_BAO_CAO_VA_SLIDE.md>)
5. [10_GIAI_THICH_BIEN_THOI_GIAN_VA_PHASE.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/ly_thuyet/10_GIAI_THICH_BIEN_THOI_GIAN_VA_PHASE.md>)

Nếu bạn chuẩn bị thu dữ liệu thật:

1. [02_QUY_TRINH_THU_THAP_DU_LIEU_THUC_TE.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/du_lieu/02_QUY_TRINH_THU_THAP_DU_LIEU_THUC_TE.md>)
2. [03_BUILD_TRAIN_VALIDATE_TEST.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/du_lieu/03_BUILD_TRAIN_VALIDATE_TEST.md>)

Nếu bạn muốn học build code từ đầu:

1. [20_BUILD_TU_DAU_MUC_LUC.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/20_BUILD_TU_DAU_MUC_LUC.md>)
2. [21_CHUAN_BI_CAU_TRUC_DU_AN_VA_LOGGER.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/21_CHUAN_BI_CAU_TRUC_DU_AN_VA_LOGGER.md>)
3. [26_CHAY_DEBUG_WALKTHROUGH_VA_DOC_LOG.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/26_CHAY_DEBUG_WALKTHROUGH_VA_DOC_LOG.md>)

Kết quả chạy:

1. [SUMMARY.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/results/SUMMARY.md>)
2. [metrics.json](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/results/metrics.json>)
3. [leaderboard.csv](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/results/leaderboard.csv>)

## 4. Lệnh chạy

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py
```

Với dữ liệu thật:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py --data-csv .\du_lieu_that.csv
```

## 5. Câu trả lời ngắn trước hội đồng

```text
Em dùng ARX làm backbone vì ARX dễ giải thích và hợp bài toán điều khiển. Sau khi bỏ biến protocol không thực tế `Phase_identification`, ARX backbone đạt FIT_sim 82.50. Em thêm residual correction được chọn bằng validation, giúp model cuối Hybrid ARX residual đạt FIT_sim 83.43 trên cùng dữ liệu và cùng split. Em không chọn theo test nên tránh leakage.
```

## 6. Ranh giới folder

Folder này là nơi làm việc chính cho residual: code, kết quả, lý thuyết, quy trình dữ liệu và câu hỏi bảo vệ đều nằm ở đây.
