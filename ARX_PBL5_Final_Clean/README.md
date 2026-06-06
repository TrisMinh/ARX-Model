# ARX PBL5 Final Clean

Đây là bản sạch dùng để bảo vệ đồ án PBL5 cho mô hình nhà kính nhỏ `30x50x30 cm`. Sản phẩm chính hiện tại là:

```text
ARX robust chỉ cảm biến trong
```

Bản này đã bỏ:

- dòng baseline cũ `ARX(5,1,2)` khỏi báo cáo chính vì khi chạy trên bộ dữ liệu 20 giây mới nó cho `FIT_sim` âm và rất dễ gây hiểu nhầm;
- bản ARX có thêm cảm biến ngoài;
- mọi pipeline NARX/NNARX khỏi code triển khai chính.

NARX/NNARX chỉ còn trong docs so sánh lý thuyết, không phải model chạy chính.

## Kết quả hiện tại

Kết quả sinh lại ngày `2026-06-05` từ [arx_final_pipeline.py](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/src/arx_final_pipeline.py>):

| Case | Model | FIT_sim | FIT_60 20 phút | Số input |
|---|---:|---:|---:|---:|
| ARX robust chỉ cảm biến trong | `ARX_na12_nb3_nk2_alpha0.1` | 78.950 | 88.047 | 17 |

Đây là bản nên dùng để bảo vệ: gọn, dễ giải thích, không cần cảm biến ngoài, phù hợp tích hợp MPC tuyến tính.

## Cấu trúc

```text
ARX_PBL5_Final_Clean/
  src/
    arx_final_pipeline.py
  results/
    metrics.json
    comparison.csv
    FINAL_REPORT.md
    SELF_CRITIQUE.md
    arx_inside_leaderboard.csv
    test_predictions.csv
    mini_greenhouse_20s_data.csv
  docs/
    00_MUC_LUC_VA_LO_TRINH_DOC.md
    00_TONG_QUAN_CHO_NGUOI_MOI.md
    01_LY_THUYET_ARX_HE_THONG.md
    02_QUY_TRINH_THU_THAP_DU_LIEU_THUC_TE.md
    03_BUILD_TRAIN_VALIDATE_TEST.md
    04_KET_QUA_TU_PHAN_BIEN.md
    05_SO_SANH_ARX_NARX_NNARX.md
    06_CAU_HOI_BAO_VE_NHANH.md
    08_KHONG_SO_SANH_TRUC_TIEP_CAC_VERSION.md
    09_TU_DANH_GIA_NHU_GIANG_VIEN.md
    10_DE_CUONG_BAO_CAO_VA_SLIDE_BAO_VE.md
    11_THUAT_NGU_CHO_NGUOI_MOI.md
    12_NHAT_KY_RA_SOAT_TAI_LIEU.md
```

## Nên đọc theo thứ tự nào?

Người mới nên bắt đầu từ [mục lục docs](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/00_MUC_LUC_VA_LO_TRINH_DOC.md>). Nếu chuẩn bị bảo vệ gấp, đọc theo thứ tự:

1. `00_TONG_QUAN_CHO_NGUOI_MOI.md`
2. `04_KET_QUA_TU_PHAN_BIEN.md`
3. `08_KHONG_SO_SANH_TRUC_TIEP_CAC_VERSION.md`
4. `09_TU_DANH_GIA_NHU_GIANG_VIEN.md`
5. `06_CAU_HOI_BAO_VE_NHANH.md`

## Chạy nhanh

Chạy bằng dữ liệu mô phỏng có kiểm soát:

```powershell
python -B .\ARX_PBL5_Final_Clean\src\arx_final_pipeline.py
```

Chạy bằng CSV dữ liệu thật:

```powershell
python -B .\ARX_PBL5_Final_Clean\src\arx_final_pipeline.py --data-csv .\du_lieu_that.csv
```

CSV thật tối thiểu phải có các cột:

```csv
Timestamp,Soil_Moisture,Temperature_In,Humidity_In,Light_In,Drip,Fan
```

Nên có thêm:

```csv
Mist,Planned_Drip,Planned_Mist,Planned_Fan,Safety_Override,Command_Source,Soil_Low_SP,Soil_High_SP
```

## Residual đã tách riêng

Hybrid ARX residual không còn nằm trong folder này. Bản residual dùng để bảo vệ nằm ở:

```powershell
.\ARX_PBL5_Residual_Final_Clean
```

Kết quả hiện tại:

| Model | FIT_sim |
|---|---:|
| ARX thuần 17 input | 78.950 |
| Hybrid ARX residual chọn bằng validation | 81.102 |

Nếu chọn residual làm bản bảo vệ, dùng folder `ARX_PBL5_Residual_Final_Clean`. Folder `ARX_PBL5_Final_Clean` chỉ còn vai trò ARX backbone/so sánh.

## Dọn feature

Đã test ablation:

- Bỏ `VPD_Proxy_In` làm `FIT_sim` giảm khoảng `0.91` điểm, nên giữ.
- `SP_Center` và `SP_Width` không ảnh hưởng trong data hiện tại vì setpoint là hằng số, nên đã xóa khỏi input model.

## Câu nói an toàn nhất

```text
Bản ARX thuần final đạt FIT_sim 78.95 trên dữ liệu mô phỏng vật lý 20 giây/mẫu với 17 input. Các thử nghiệm mở rộng đã được tách ra folder riêng, không nằm trong bản clean chính. Kết quả này chưa được gọi là kết quả phần cứng thật; khi có CSV thật phải train/test lại bằng cùng pipeline.
```
