# Flow xử lý dữ liệu và huấn luyện mô hình ARX 5 giây

File này mô tả luồng tổng quát của project từ lúc có raw data đến khi có model ARX hoàn chỉnh.

## 1. Thu thập raw data theo block trong ngày

Dữ liệu raw được tổ chức theo từng ngày. Mỗi ngày có 4 block đại diện:

```text
01_morning_raw.csv
02_noon_raw.csv
03_afternoon_raw.csv
04_night_raw.csv
```

Ví dụ:

```text
data/_01_data/2026-04-08/01_morning_raw.csv
data/_01_data/2026-04-08/02_noon_raw.csv
data/_01_data/2026-04-08/03_afternoon_raw.csv
data/_01_data/2026-04-08/04_night_raw.csv
```

Mỗi file raw có 8 cột chính:

```text
Timestamp
Temperature
Humidity
Light
Soil_Moisture
Drip
Mist
Fan
```

Ý nghĩa:

| Nhóm | Cột |
|---|---|
| Thời gian | `Timestamp` |
| Sensor môi trường | `Temperature`, `Humidity`, `Light` |
| Target cần dự đoán | `Soil_Moisture` |
| Thiết bị điều khiển | `Drip`, `Mist`, `Fan` |

Code liên quan:

```text
src/data/collection/step_00_data_io.py
```

Các hàm chính:

| Hàm | Vai trò |
|---|---|
| `input_csv_paths()` | Lấy danh sách file CSV raw |
| `normalize_model_columns()` | Kiểm tra đủ 8 cột bắt buộc |
| `format_model_data()` | Chuẩn hóa format sensor và actuator |

## 2. Gộp raw data

Pipeline đọc tất cả file raw trong `_01_data`, sau đó gộp lại thành file tổng:

```text
data/_01_data/00_raw_tong_hop.csv
```

Mục đích:

```text
Kiểm tra tổng raw data trước xử lý
Biết trước clean có bao nhiêu dòng, bao nhiêu missing/lỗi
```

Code liên quan:

```text
src/data/collection/step_03_pipeline.py
```

Trong `run()`:

```python
raw_df = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True)
format_model_data(raw_df).to_csv(input_dir / "00_raw_tong_hop.csv", index=False)
```

## 3. Clean từng file raw

Mỗi file raw được xử lý riêng bằng:

```python
clean_data_file(path)
```

File code:

```text
src/data/collection/step_01_clean_data.py
```

Các bước clean:

| Bước | Hàm | Mục đích |
|---|---|---|
| 1 | `require_model_columns()` | Kiểm tra đủ 8 cột |
| 2 | `parse_sort_timestamps()` | Parse timestamp, làm tròn về lưới 5 giây, sort thời gian |
| 3 | `collapse_duplicate_timestamps()` | Gộp các dòng trùng timestamp |
| 4 | `reindex_to_5s_grid()` | Đưa dữ liệu về lưới thời gian đều 5 giây |
| 5 | `coerce_numeric()` | Ép sensor/actuator về kiểu số |
| 6 | `fill_sensor_missing()` | Xử lý missing/outlier sensor bằng nội suy và clip |
| 7 | `smooth_soil_moisture()` | Làm mềm spike bất thường của `Soil_Moisture` |
| 8 | `fill_actuator_missing()` | Xử lý missing actuator và ép về 0/1 |
| 9 | `format_model_data()` | Làm tròn sensor, chuẩn hóa output |

Output từng file:

```text
data/_02_clean_data/<ngày>/<phiên>_sau_xu_ly.csv
```

Ví dụ:

```text
data/_02_clean_data/2026-04-08/01_morning_sau_xu_ly.csv
```

## 4. Gộp clean data

Sau khi clean từng file, pipeline gộp lại thành:

```text
data/_02_clean_data/00_sau_xu_ly_tong_hop.csv
```

Code:

```python
cleaned_data = clean_all_data_files(raw_paths, clean_dir)
```

Trong `clean_all_data_files()`:

```python
cleaned_data = (
    pd.concat(cleaned_files, ignore_index=True)
    .sort_values("Timestamp")
    .reset_index(drop=True)
    .loc[:, MODEL_COLS]
)
```

Đến đây có thể so sánh nhanh:

```text
Trước clean: 00_raw_tong_hop.csv
Sau clean:   00_sau_xu_ly_tong_hop.csv
```

## 5. Sinh bộ data train hoàn chỉnh

Từ clean data, pipeline sinh ra data train cuối:

```text
data/mini_greenhouse_5s_data.csv
```

File này là dữ liệu chính để train ARX.

Code liên quan:

```text
src/data/collection/step_02_generate_data.py
src/data/collection/step_03_pipeline.py
```

Trong `run()`:

```python
final_df = build_training_data(cleaned_data, days, seed)
format_model_data(final_df).to_csv(data_path, index=False)
```

Trong `build_training_data()`:

```python
source_data = source_data.sort_values("Timestamp").reset_index(drop=True)
return _build_training_data(source_data, days, seed)
```

Các ý chính khi sinh data train:

| Thành phần | Cách làm |
|---|---|
| Nền môi trường | Phân tích median, q10, q90 từ clean data |
| Temperature/Humidity/Light | Sinh theo chu kỳ ngày đêm và profile đã phân tích |
| Actuator | Dùng mẫu bật/tắt `Drip`, `Mist`, `Fan` từ data sạch |
| Soil_Moisture | Sinh phản ứng theo môi trường và thiết bị |
| Số ngày | Mặc định `--days 12` |
| Sampling | 5 giây |

Command build data:

```bash
python scripts/01_build_data.py --days 12
```

Kết quả chính:

```text
data/mini_greenhouse_5s_data.csv
```

## 6. Tạo feature cho model ARX

Sau khi có `mini_greenhouse_5s_data.csv`, bước train đọc data và tạo thêm feature.

Code:

```text
src/preprocessing/features.py
```

Hàm:

```python
add_features(df)
```

Feature được tạo thêm:

| Feature | Ý nghĩa |
|---|---|
| `Light_log` | Giảm độ lệch scale của Light |
| `Temp_x_Humidity` | Tương tác nhiệt độ và độ ẩm |
| `Temp_x_Light` | Tương tác nhiệt độ và ánh sáng |
| `Humidity_x_Light` | Tương tác độ ẩm và ánh sáng |
| `Air_Dryness` | Độ khô không khí |
| `Temp_x_Air_Dryness` | Tác động nhiệt độ khi không khí khô |
| `Hour_sin`, `Hour_cos` | Chu kỳ giờ trong ngày |
| `Day_sin`, `Day_cos` | Chu kỳ ngày |

Trong train pipeline:

```python
df = add_features(raw)
```

## 7. Chia train, validation và test

Code:

```text
src/preprocessing/scaling.py
```

Hàm:

```python
split_time(df, cfg)
```

Chiến lược hiện tại:

```text
same_clock_by_day
```

Nghĩa là:

```text
Train:      các ngày trước
Validation: ngày kế cuối
Test:       ngày cuối
```

Chỉ lấy cùng khung giờ:

```text
6:00-22:00
```

Với chu kỳ 5 giây:

```text
1 ngày trong 6:00-22:00 = 16 giờ * 3600 / 5 = 11520 mẫu
```

Vì vậy validation và test có số dòng bằng nhau nếu mỗi tập là 1 ngày.

## 8. Fit scale trên train và chuẩn hóa dữ liệu

Code:

```text
src/preprocessing/scaling.py
```

Các hàm:

| Hàm | Vai trò |
|---|---|
| `fit_scale_stats()` | Tính mean/std trên train |
| `apply_scale()` | Chuẩn hóa train/validation/test bằng mean/std của train |
| `inverse_y()` | Đưa `Soil_Moisture` từ scale chuẩn hóa về scale thật |

Trong `pipeline.py`:

```python
stats = fit_scale_stats(train, INPUT_COLS)
train_z = apply_scale(train, stats)
val_z = apply_scale(val, stats)
test_z = apply_scale(test, stats)
```

Lưu ý:

```text
Mean/std chỉ fit trên train để tránh rò rỉ thông tin từ validation/test.
```

## 9. Tạo danh sách ứng viên ARX

Code:

```text
src/algorithm/specs.py
```

Hàm:

```python
default_specs(grid)
```

Mỗi cấu hình gồm:

| Tham số | Ý nghĩa |
|---|---|
| `na` | Số mẫu quá khứ của `Soil_Moisture` |
| `nb` | Số mẫu quá khứ của mỗi input |
| `nk` | Độ trễ input |
| `alpha` | Hệ số Ridge |

Ví dụ model được chọn:

```text
ARX_na96_nb16_nk2_alpha10
```

Nghĩa là:

```text
na = 96
nb = 16
nk = 2
alpha = 10
```

## 10. Fit từng ứng viên ARX

Code:

```text
src/pipeline.py
src/algorithm/arx.py
```

Trong `pipeline.py`:

```python
theta = fit_arx(train_z, spec, INPUT_COLS)
```

Trong `fit_arx()`:

```python
x_train, y_train = build_arx_matrix(df_train_z, spec, input_cols)
lhs = x_train.T @ x_train + spec.alpha * penalty
rhs = x_train.T @ y_train
theta = np.linalg.solve(lhs, rhs)
```

Tức là:

```text
1. Tạo ma trận X từ y quá khứ và input quá khứ.
2. Tạo vector Y là Soil_Moisture cần dự đoán.
3. Tính lhs = X^T X + alpha * penalty.
4. Tính rhs = X^T Y.
5. Giải lhs * theta = rhs để ra theta.
```

Kết quả của bước này:

```text
theta = vector hệ số ARX
```

## 11. Chọn model bằng validation free-run

Với mỗi ứng viên, pipeline đánh giá trên validation.

Code:

```python
val_1, val_true_1 = predict_one_step(...)
val_free, val_true_free = simulate_free_run(...)
```

Sau đó ghi vào leaderboard:

```text
result/leaderboard.csv
```

Tiêu chí chọn:

```text
Validation FIT free-run cao nhất
```

Vì free-run khó hơn 1-step: model phải dùng lại dự đoán của chính nó làm quá khứ.

## 12. Đánh giá model cuối

Code:

```text
src/evaluation/simulation.py
src/evaluation/metrics.py
```

Hàm:

```python
evaluate_model()
```

Các chế độ đánh giá:

| Chế độ | Ý nghĩa |
|---|---|
| `1-step` | Dự đoán từng bước, dùng y quá khứ thật |
| `5min chunked` | Mô phỏng từng đoạn 5 phút |
| `20min chunked` | Mô phỏng từng đoạn 20 phút |
| `free-run` | Mô phỏng tự do, dùng lại y dự đoán làm quá khứ |

Metric:

| Metric | Ý nghĩa |
|---|---|
| `FIT` | Độ khớp, càng cao càng tốt |
| `RMSE` | Sai số căn trung bình bình phương, càng thấp càng tốt |
| `Bias` | Sai lệch trung bình |

## 13. Ghi artifact kết quả

Sau khi chọn model, pipeline ghi các file:

| File | Nội dung |
|---|---|
| `result/leaderboard.csv` | Bảng so sánh các ứng viên |
| `result/metrics.json` | Metric train/validation/test |
| `result/arx_5s_model.json` | Model artifact để chạy lại |
| `result/test_predictions.csv` | Dự đoán trên tập test |
| `result/SUMMARY.md` | Tóm tắt kết quả |

Trong `arx_5s_model.json` có:

```text
target
sampling_seconds
input_cols
spec: na, nb, nk, alpha
theta
scale: mean/std
clip_scaled
```

Command train:

```bash
python scripts/02_train.py --grid quick
```

## 14. Flow tổng quát một dòng

```text
Raw session data
-> gộp raw
-> clean từng file
-> gộp clean
-> sinh mini_greenhouse_5s_data.csv
-> add_features
-> split train/validation/test
-> scale bằng train
-> thử nhiều ARX spec
-> fit theta tự viết bằng normal equation
-> chọn model theo validation free-run
-> đánh giá train/validation/test
-> lưu leaderboard, metrics, model artifact, predictions
```

## 15. Flow theo file code

```text
scripts/01_build_data.py
    -> data.collection.run()
        -> input_csv_paths()
        -> clean_all_data_files()
        -> build_training_data()
        -> data/mini_greenhouse_5s_data.csv

scripts/02_train.py
    -> run_pipeline()
        -> add_features()
        -> split_time()
        -> fit_scale_stats()
        -> apply_scale()
        -> default_specs()
        -> fit_arx()
            -> build_arx_matrix()
            -> solve normal equation ra theta
        -> evaluate_model()
        -> ghi result/*
```
