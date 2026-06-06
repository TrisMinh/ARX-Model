# Build từ đầu: mục lục

## 1. Mục tiêu của bộ docs này

Bộ docs này hướng dẫn viết lại project từ đầu theo đúng code hiện tại:

```text
ARX backbone -> Hybrid ARX residual correction
```

Không đọc kiểu “hàm này xong tới hàm kia” một cách lướt qua. Mỗi phần sẽ nói:

- hàm dùng để làm gì;
- input cần có;
- output trả về;
- cách tự viết hàm;
- sau hàm nên log gì để kiểm tra;
- lỗi dễ gặp;
- vì sao hàm đó cần cho model.

## 2. Thứ tự đọc

1. [21_CHUAN_BI_CAU_TRUC_DU_AN_VA_LOGGER.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/21_CHUAN_BI_CAU_TRUC_DU_AN_VA_LOGGER.md>)
2. [22_BUILD_BACKBONE_ARX_PHAN_1_DU_LIEU.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/22_BUILD_BACKBONE_ARX_PHAN_1_DU_LIEU.md>)
3. [23_BUILD_BACKBONE_ARX_PHAN_2_MA_TRAN_MO_PHONG.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/23_BUILD_BACKBONE_ARX_PHAN_2_MA_TRAN_MO_PHONG.md>)
4. [24_BUILD_BACKBONE_ARX_PHAN_3_DANH_GIA_BAO_CAO.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/24_BUILD_BACKBONE_ARX_PHAN_3_DANH_GIA_BAO_CAO.md>)
5. [25_BUILD_HYBRID_RESIDUAL_TUNG_HAM.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/25_BUILD_HYBRID_RESIDUAL_TUNG_HAM.md>)
6. [26_CHAY_DEBUG_WALKTHROUGH_VA_DOC_LOG.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/26_CHAY_DEBUG_WALKTHROUGH_VA_DOC_LOG.md>)
7. [27_TU_PHAN_BIEN_DOC_BUILD_TU_DAU.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/docs/huong_dan_build/27_TU_PHAN_BIEN_DOC_BUILD_TU_DAU.md>)

## 3. Code chính nằm ở đâu?

```text
src/arx_backbone_pipeline.py
src/arx_residual_experiment.py
src/debug_walkthrough.py
```

Ý nghĩa:

- `arx_backbone_pipeline.py`: sinh/đọc dữ liệu, tạo feature, train ARX, mô phỏng ARX, audit và report;
- `arx_residual_experiment.py`: dùng ARX làm backbone, học residual, chọn model bằng validation;
- `debug_walkthrough.py`: chạy từng bước và in log để nhìn output sau mỗi hàm.

## 4. Lệnh nên chạy khi học code

Chạy log từng bước:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\debug_walkthrough.py
```

Chạy full residual final:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py
```

Chạy với dữ liệu thật:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py --data-csv .\data\log_that.csv
```

## 5. Sơ đồ luồng chương trình

```text
OutdoorConfig
-> load_raw_data
-> add_features
-> split_time
-> fit_scale_stats
-> apply_scale
-> run_arx_search
   -> arx_specs
   -> fit_arx
   -> evaluate_arx
   -> validation_blocks_arx
-> simulate_arx
-> residual_features
-> candidate_models
-> robust_block_score
-> chọn residual bằng validation
-> ghi metrics.json, leaderboard.csv, SUMMARY.md, test_predictions.csv
```

## 6. Câu nhớ nhanh

```text
ARX học quy luật chính.
Residual học phần sai số còn lại.
Validation chọn model.
Test chỉ báo cáo cuối.
Log từng bước để không bị mù dữ liệu.
```
