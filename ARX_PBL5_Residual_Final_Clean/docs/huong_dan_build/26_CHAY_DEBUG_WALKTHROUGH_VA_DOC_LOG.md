# Chạy debug walkthrough và đọc log

## 1. Mục tiêu

File `src/debug_walkthrough.py` được thêm để người mới nhìn thấy output sau từng bước.

Chạy:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\debug_walkthrough.py
```

Nếu muốn chạy full ARX search trong walkthrough:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\debug_walkthrough.py --full-search
```

Nếu dùng CSV thật:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\debug_walkthrough.py --data-csv .\data\log_that.csv
```

## 2. Hàm `log_step`

`log_step` nhận:

```python
log_step(title, value, note="", max_rows=3)
```

Nếu `value` là DataFrame, log:

- shape;
- danh sách cột;
- vài dòng đầu.

Nếu `value` là numpy array, log:

- shape;
- dtype;
- vài giá trị đầu.

Nếu `value` là dict, log JSON có thụt dòng.

## 3. Các bước log chính

| Step | Hàm/bước | Cần nhìn gì |
|---|---|---|
| 01 | `OutdoorConfig` | tham số đúng chưa |
| 02 | `load_raw_data` | data có dòng/cột chưa |
| 03 | `add_features` | feature mới có xuất hiện chưa |
| 04 | `split_time` | train/val/test đúng tỷ lệ chưa |
| 05 | `fit_scale_stats` | scaler chỉ từ train |
| 06 | `apply_scale` | dữ liệu scale hợp lý |
| 07 | clip | giới hạn dự đoán |
| 08 | `fit_arx` hoặc `run_arx_search` | spec và theta |
| 09 | `build_arx_matrix` X | shape ma trận |
| 10 | target y | shape vector y |
| 11 | one-step | metrics dễ nhất |
| 12 | free-run | metrics khó nhất |
| 13 | 60-step | horizon điều khiển |
| 14 | `ResidualConfig` | lag và shrink |
| 15 | `residual_features` | shape feature residual |
| 17 | `robust_block_score` | score validation |
| 18 | Hybrid metrics | kết quả cuối của residual |

## 4. Ví dụ cách đọc shape

Nếu log:

```text
type=ndarray shape=(48372, 109)
```

Nghĩa là:

- có 48.372 dòng train sau khi bỏ lag;
- mỗi dòng có 109 cột feature ARX.

Với ARX final:

```text
109 = 12 output lag + 16 input * 6 lag + 1 intercept
```

Nếu log residual:

```text
shape=(48372, 86)
```

Nghĩa là:

```text
86 = 1 y_arx hiện tại + 5 y lag + 16 input * 5 lag
```

## 5. Khi log thấy sai thì kiểm tra gì?

### Thiếu cột

Nếu lỗi kiểu:

```text
KeyError: Temperature_In
```

Kiểm tra CSV thật có cột này không.

### Shape quá nhỏ

Nếu số dòng sau lag quá ít, nghĩa là dữ liệu quá ngắn so với `na/nb/nk`.

### FIT one-step cao nhưng FIT_sim thấp

Model dự đoán một bước được nhưng free-run trôi. Lúc này residual correction có ý nghĩa.

### Validation tốt nhưng test tệ

Không được đổi model theo test ngay. Phải xem:

- validation có đại diện chưa;
- split có bị lệch phân phối không;
- dữ liệu test có phase khác không;
- feature có bị phụ thuộc mô phỏng không.

## 6. Lệnh kiểm tra cuối

Sau khi đọc walkthrough, chạy bản chính:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py
```

Kỳ vọng hiện tại:

```text
ARX backbone:    FIT_sim=82.496
Hybrid selected: FIT_sim=83.433
Selected: hgb_leaf15_l2_0.01|0.25
```
