# Mục lục và lộ trình đọc

## 1. Folder này dùng để làm gì?

`ARX_PBL5_Final_Clean` là bản tài liệu và mã nguồn sạch cho đồ án PBL5 dự đoán độ ẩm đất của mô hình nhà kính nhỏ `30x50x30 cm`.

Mục tiêu của folder này không phải là gom mọi thử nghiệm vào một chỗ, mà là tách rõ:

- đâu là sản phẩm chính có thể bảo vệ;
- đâu là thử nghiệm mở rộng;
- đâu là dữ liệu mô phỏng;
- đâu là quy trình sẽ dùng lại khi có dữ liệu phần cứng thật.

## 2. Kết luận hiện tại

| Vai trò | Model | FIT_sim test | Ghi chú |
|---|---:|---:|---|
| Sản phẩm chính | ARX thuần `ARX_na12_nb3_nk2_alpha0.1` | 78.950 | Gọn, dễ giải thích, hợp MPC tuyến tính |
| Thử nghiệm mở rộng | Hybrid ARX residual | 81.102 | Tăng fit nhưng không còn là ARX thuần |

Không gọi các kết quả này là dữ liệu phần cứng thật. Đây là kết quả trên dữ liệu mô phỏng vật lý có kiểm soát, dùng để kiểm tra pipeline và thiết kế quy trình thu dữ liệu.

## 3. Đọc theo mục tiêu

Nếu bạn là người mới:

1. [00_TONG_QUAN_CHO_NGUOI_MOI.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/00_TONG_QUAN_CHO_NGUOI_MOI.md>)
2. [11_THUAT_NGU_CHO_NGUOI_MOI.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/11_THUAT_NGU_CHO_NGUOI_MOI.md>)
3. [01_LY_THUYET_ARX_HE_THONG.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/01_LY_THUYET_ARX_HE_THONG.md>)

Nếu bạn chuẩn bị thu dữ liệu thật:

1. [02_QUY_TRINH_THU_THAP_DU_LIEU_THUC_TE.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/02_QUY_TRINH_THU_THAP_DU_LIEU_THUC_TE.md>)
2. [03_BUILD_TRAIN_VALIDATE_TEST.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/03_BUILD_TRAIN_VALIDATE_TEST.md>)

Nếu bạn chuẩn bị bảo vệ:

1. [04_KET_QUA_TU_PHAN_BIEN.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/04_KET_QUA_TU_PHAN_BIEN.md>)
2. [08_KHONG_SO_SANH_TRUC_TIEP_CAC_VERSION.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/08_KHONG_SO_SANH_TRUC_TIEP_CAC_VERSION.md>)
3. [09_TU_DANH_GIA_NHU_GIANG_VIEN.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/09_TU_DANH_GIA_NHU_GIANG_VIEN.md>)
4. [06_CAU_HOI_BAO_VE_NHANH.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/06_CAU_HOI_BAO_VE_NHANH.md>)
5. [10_DE_CUONG_BAO_CAO_VA_SLIDE_BAO_VE.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/10_DE_CUONG_BAO_CAO_VA_SLIDE_BAO_VE.md>)

Nếu thầy hỏi vì sao không dùng model khác:

1. [05_SO_SANH_ARX_NARX_NNARX.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/05_SO_SANH_ARX_NARX_NNARX.md>)
2. [Hybrid residual final](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/README.md>)

## 4. Ranh giới học thuật

Ba câu phải giữ nhất quán:

```text
ARX thuần final là sản phẩm chính.
Hybrid ARX residual là thử nghiệm mở rộng, đã tách riêng khỏi folder clean.
Dữ liệu hiện tại là mô phỏng vật lý có kiểm soát, chưa phải dữ liệu phần cứng thật.
```

Nếu nói sai một trong ba câu này, phần bảo vệ có thể bị bắt lỗi.

## 5. Lệnh chạy chính

ARX thuần:

```powershell
python -B .\ARX_PBL5_Final_Clean\src\arx_final_pipeline.py
```

Hybrid ARX residual:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py
```

Chạy với dữ liệu thật:

```powershell
python -B .\ARX_PBL5_Final_Clean\src\arx_final_pipeline.py --data-csv .\data\log_that.csv
```
