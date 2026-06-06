# Build ARX backbone phần 1: dữ liệu và feature

## 1. `json_ready`

Mục đích: chuyển object khó ghi JSON thành kiểu JSON ghi được.

Input có thể là:

- dict;
- list/tuple;
- numpy array;
- số numpy;
- `Path`.

Output: object Python thường như `dict`, `list`, `float`, `int`, `str`.

Cách viết:

```python
def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    ...
```

Log:

```python
log_step("json_ready", json_ready({"x": np.array([1, 2])}))
```

Kỳ vọng:

```json
{"x": [1, 2]}
```

## 2. `fit_metrics`

Mục đích: tính chất lượng dự đoán.

Input:

- `y_true`: giá trị thật;
- `y_pred`: giá trị dự đoán.

Output:

- `FIT`;
- `RMSE`;
- `MAE`;
- `Bias`.

Công thức FIT:

```text
FIT = 100 * (1 - norm(y_true - y_pred) / norm(y_true - mean(y_true)))
```

Cách hiểu:

- FIT càng cao càng tốt;
- RMSE càng thấp càng tốt;
- Bias gần 0 càng tốt.

Log:

```python
log_step("fit_metrics", fit_metrics(y_true, y_pred))
```

## 3. `safe_corr`

Mục đích: tính correlation nhưng tránh lỗi khi dữ liệu hỏng.

Hàm kiểm tra:

- bỏ `NaN`;
- nếu còn dưới 3 mẫu thì trả `None`;
- nếu một mảng gần như hằng số thì trả `None`;
- còn lại trả correlation.

Log:

```python
log_step("safe_corr", {"corr": safe_corr(a, b)})
```

## 4. `p_on`

Mục đích: tính phần trăm actuator bật trong một vùng điều kiện.

Ví dụ:

```text
Trong các dòng đất đang thấp, bơm bật bao nhiêu phần trăm?
```

Input:

- `mask`: vùng cần xét;
- `values`: actuator 0/1.

Output: phần trăm bật.

## 5. `phase_for_day`

Mục đích: chia ngày mô phỏng thành phase để sinh data:

```text
day < 2  -> commissioning_rule_based
day < 8  -> identification_safe_excitation
còn lại -> deployment_validation
```

Lưu ý: phase chỉ dùng trong data audit/sinh dữ liệu mô phỏng. Không dùng `Phase_identification` làm input model final.

Log:

```python
for day in range(10):
    print(day, phase_for_day(day))
```

## 6. `make_planned_excitation`

Mục đích: tạo lịch kích thích actuator có kiểm soát.

Output:

- `planned_drip`;
- `planned_mist`;
- `planned_fan`;
- `label`.

Cách viết từng bước:

1. Tính số mẫu mỗi ngày.
2. Tạo mảng 0 cho từng actuator.
3. Với mỗi ngày, bỏ qua phase commissioning.
4. Với các khung giờ như 7h, 10h, 13h, tạo pulse bơm.
5. Với các khung giờ khác, tạo pulse quạt/phun sương.
6. Ghi `label` để audit sau này.

Log nên in:

```python
planned = make_planned_excitation(cfg, rng, n_rows)
log_step("planned keys", planned.keys())
log_step("planned drip percent", planned["planned_drip"].mean() * 100)
```

Nếu `planned_drip` toàn 0 thì data nghèo, model khó học tác động bơm.

## 7. `generate_outdoor_data`

Mục đích: sinh dữ liệu mô phỏng vật lý cho nhà kính nhỏ.

Hàm này làm nhiều việc:

1. Tạo timestamp.
2. Tạo giờ/ngày.
3. Tạo ánh sáng ngoài/trong.
4. Tạo nhiệt độ và độ ẩm môi trường.
5. Gọi `make_planned_excitation`.
6. Tính trạng thái actuator thật sau safety.
7. Cập nhật độ ẩm đất theo tác động bơm, quạt, ánh sáng, nhiễu.
8. Xuất DataFrame.

Output tối thiểu có:

```text
Timestamp, Soil_Moisture, Temperature_In, Humidity_In, Light_In, Drip, Mist, Fan
```

Log:

```python
df = generate_outdoor_data(cfg)
log_step("generate_outdoor_data", df)
log_step("soil range", {"min": df["Soil_Moisture"].min(), "max": df["Soil_Moisture"].max()})
```

Nếu `Soil_Moisture` gần như hằng số, dữ liệu không đủ kích thích.

## 8. `load_raw_data`

Mục đích: lấy data từ hai nguồn:

- nếu có `--data-csv`: đọc CSV thật;
- nếu không: gọi `generate_outdoor_data`.

Hàm còn tự bổ sung cột thiếu hợp lý:

- nếu thiếu `Mist`, tạo 0;
- nếu thiếu `Planned_Drip`, tạo 0;
- nếu thiếu `Protocol_Phase`, gán `real_logged_data`;
- nếu thiếu `Day_Index`, tính từ timestamp.

Hàm kiểm tra cột bắt buộc. Nếu thiếu, báo lỗi ngay.

Log:

```python
raw_df = load_raw_data(cfg, data_csv)
log_step("load_raw_data", raw_df)
```

## 9. `add_features`

Mục đích: thêm feature từ dữ liệu đã có, không dùng tương lai.

Feature tạo thêm:

- `Light_log`;
- `TempIn_x_HumiIn`;
- `TempIn_x_Light`;
- `HumiIn_x_Light`;
- `Indoor_Dryness`;
- `VPD_Proxy_In`;
- `Hour_sin`, `Hour_cos`;
- `Day_sin`, `Day_cos`.

Không tạo `Phase_identification` trong bản final.

Log:

```python
df = add_features(raw_df)
log_step("add_features selected", df[list(INSIDE_INPUT_COLS) + ["Soil_Moisture"]])
```

## 10. `split_time`

Mục đích: chia train/validation/test theo thời gian.

Không shuffle.

Output:

```text
train, validation, test
```

Log:

```python
log_step("split_time", {"train": len(train), "val": len(val), "test": len(test)})
```

## 11. `fit_scale_stats`

Mục đích: tính mean/std chỉ trên train.

Vì sao chỉ train?

```text
Nếu fit scaler trên toàn bộ data thì validation/test đã rò thông tin vào train.
```

Output là dict:

```text
column -> (mean, std)
```

Log:

```python
stats = fit_scale_stats(train, INSIDE_INPUT_COLS)
log_step("fit_scale_stats", {k: stats[k] for k in list(stats)[:5]})
```

## 12. `apply_scale`

Mục đích: dùng stats của train để scale train/val/test.

Hàm scale:

```text
z = (x - mean_train) / std_train
```

Log:

```python
train_z = apply_scale(train, stats)
log_step("train_z", train_z[["Soil_Moisture", "Temperature_In"]])
```

## 13. `inverse_y`

Mục đích: đổi `Soil_Moisture` từ z-score về đơn vị thật.

Cần dùng khi báo cáo metrics vì người đọc cần hiểu sai số theo đơn vị độ ẩm đất.

Log:

```python
y_real = inverse_y(y_z, stats)
log_step("inverse_y", y_real)
```
