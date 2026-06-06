# Build Hybrid residual từng hàm

## 1. `ResidualConfig`

Mục đích: gom tham số riêng cho residual.

Các trường quan trọng:

- `residual_y_lags = (1, 2, 3, 6, 12)`;
- `residual_input_lags = (2, 3, 6, 12, 24)`;
- `shrink_grid = (0, 0.25, 0.5, 0.75, 1)`;
- `validation_blocks = 4`;
- `validation_std_penalty = 0.5`.

Log:

```python
cfg = ResidualConfig()
log_step("ResidualConfig", cfg.__dict__)
```

## 2. `json_ready`

Residual dùng lại:

```python
return base.json_ready(value)
```

Mục đích: không viết lại logic serialize JSON.

## 3. `residual_features`

Mục đích: tạo ma trận feature cho model học sai số.

Input:

- `df_z`: dữ liệu đã scale;
- `y_arx_sim_z`: quỹ đạo ARX mô phỏng;
- `spec`: cấu hình ARX;
- `input_cols`: danh sách input;
- `cfg`: ResidualConfig.

Ý tưởng:

```text
Residual không được dùng Soil_Moisture thật tương lai.
Nó được dùng y_arx_sim và input lag quá khứ.
```

Số cột feature:

```text
1 + len(residual_y_lags) + len(input_cols) * len(residual_input_lags)
```

Hiện tại:

```text
1 + 5 + 16*5 = 86 cột
```

Cách viết từng bước:

1. Tính `lag = base.arx_max_lag(spec)`.
2. Copy `Soil_Moisture` thành `y_source`.
3. Từ vị trí đủ lag trở đi, thay bằng `y_arx_sim_z`.
4. Chuẩn bị mảng input.
5. Với từng thời điểm `t`, tạo một row gồm:
   - `y_arx_sim_z[row_idx]`;
   - các lag của `y_source`;
   - các lag của từng input.
6. Gán row vào ma trận `x`.

Log:

```python
x_train = residual_features(train_z, y_arx_train_z, spec, INSIDE_INPUT_COLS, cfg)
log_step("residual_features", x_train)
```

Kỳ vọng: số dòng bằng số dòng sau khi bỏ lag, số cột bằng 86.

## 4. `candidate_models`

Mục đích: tạo danh sách model residual để thử.

Gồm:

- Ridge alpha 0.1;
- Ridge alpha 1;
- Ridge alpha 10;
- HGB leaf 15, l2 0.01;
- HGB leaf 31, l2 0.05.

Vì sao có Ridge?

- đơn giản;
- dễ kiểm tra;
- ít overfit.

Vì sao có HistGradientBoosting?

- học được phi tuyến nhẹ;
- phù hợp residual có quan hệ không tuyến tính.

Log:

```python
models = candidate_models(cfg)
log_step("candidate model names", [name for name, _ in models])
```

## 5. `inverse_metrics`

Mục đích: tính metrics trên đơn vị thật, không phải z-score.

Hàm gọi:

```python
base.inverse_y(...)
base.fit_metrics(...)
```

Log:

```python
metrics = inverse_metrics(y_true_z, y_pred_z, stats)
log_step("inverse_metrics", metrics)
```

## 6. `robust_block_score`

Mục đích: chọn model ổn định trên validation.

Các bước:

1. Chia validation thành 4 block.
2. Tính FIT từng block.
3. Tính mean và std.
4. Robust score:

```text
mean - 0.5 * std
```

Log:

```python
score = robust_block_score(y_true_val_z, y_pred_val_z, stats, cfg)
log_step("robust_block_score", score)
```

## 7. `run_experiment`

Mục đích: chạy full Hybrid residual.

Luồng chi tiết:

```text
OutdoorConfig
-> load_raw_data
-> add_features
-> split_time
-> fit/apply scaler
-> run_arx_search
-> simulate ARX train/val/test
-> residual_features train/val/test
-> y_res_train = y_true_train - y_arx_train
-> thêm baseline shrink=0
-> train từng residual model
-> thử từng shrink
-> tính validation robust score
-> sort leaderboard
-> chọn dòng đầu
-> ghi metrics, leaderboard, prediction, summary
```

Điểm quan trọng:

```text
selected = leaderboard.iloc[0]
```

Tức là chọn bằng validation robust score, không chọn bằng test.

Log nên có:

```python
results = run_experiment(cfg)
log_step("selected residual", results["selected_by_validation"])
log_step("metrics", results["metrics"])
```

## 8. `write_summary`

Mục đích: ghi `results/SUMMARY.md`.

Nội dung:

- ARX backbone metrics;
- Hybrid selected metrics;
- candidate được chọn;
- gain FIT;
- kết luận có nên dùng residual không.

Log:

```python
write_summary(results)
print((RESULTS_DIR / "SUMMARY.md").exists())
```

## 9. `parse_args`

Đọc:

- `--data-csv`;
- `--days`;
- `--sampling-seconds`;
- `--seed`.

## 10. `main`

Luồng:

```text
parse_args
-> ResidualConfig
-> run_experiment
-> print ARX backbone FIT
-> print Hybrid selected FIT
```

Đây là entrypoint chính của project bảo vệ.
