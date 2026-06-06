# Build ARX backbone phần 3: audit, report và main

## 1. `residual_diagnostics`

Mục đích: kiểm tra sai số còn tự tương quan không.

Input:

- `y_true`;
- `y_pred`;
- `max_lag`, mặc định 60.

Output:

- mean residual;
- std residual;
- ACF từ lag 1 đến 60;
- max abs ACF.

Nếu ACF còn cao, nghĩa là sai số còn có quy luật. Đây là lý do hợp lý để thử residual correction.

Log:

```python
diag = residual_diagnostics(y_true, y_pred)
log_step("residual diagnostics", diag)
```

## 2. `audit_data`

Mục đích: kiểm tra dữ liệu trước khi tin kết quả model.

Hàm kiểm tra:

- số dòng;
- thời gian bắt đầu/kết thúc;
- chu kỳ lấy mẫu;
- missing;
- timestamp trùng;
- range soil/nhiệt/ẩm;
- phần trăm actuator bật;
- safety override;
- correlation giữa planned actuator và trạng thái đất trước đó;
- summary train/validation/test.

Log:

```python
audit = audit_data(df, cfg)
log_step("audit_data", audit)
```

Nếu `missing_total > 0`, phải xử lý trước khi báo cáo.

## 3. `comparison_row`

Mục đích: biến kết quả model thành một dòng bảng gọn.

Input:

- `case`;
- `family`;
- `result`;
- `note`.

Output: dict có FIT/RMSE và tên model.

Log:

```python
row = comparison_row("ARX backbone", "ARX", arx, "nền so sánh")
log_step("comparison row", row)
```

## 4. `fmt`

Mục đích: format số cho markdown.

Nếu số là `None` hoặc `NaN`, trả `NA`.

Log:

```python
print(fmt(1.23456))
print(fmt(None))
```

Kỳ vọng:

```text
1.235
NA
```

## 5. `write_report`

Mục đích: tự sinh `FINAL_REPORT.md` cho ARX backbone.

Hàm nhận `payload`, lấy:

- audit data;
- comparison;
- model được chọn;
- metrics.

Sau đó ghép các dòng markdown và ghi file.

Log:

```python
write_report(payload)
print((RESULTS_DIR / "FINAL_REPORT.md").exists())
```

Lưu ý: trong project residual final, report ARX chỉ là report backbone. Kết quả bảo vệ chính nằm trong residual summary/metrics.

## 6. `write_self_critique`

Mục đích: tự sinh file tự phản biện cho ARX backbone.

Nội dung thường gồm:

- dữ liệu mô phỏng chưa phải phần cứng;
- không dùng trạng thái ẩn mô phỏng;
- không chọn theo test;
- cần báo FIT_sim, không chỉ FIT_1step.

Log:

```python
write_self_critique(payload)
print((RESULTS_DIR / "SELF_CRITIQUE.md").exists())
```

## 7. `run_pipeline`

Mục đích: chạy trọn pipeline ARX backbone.

Luồng trong hàm:

```text
load_raw_data
-> add_features
-> split_time
-> fit_scale_stats
-> apply_scale
-> run_arx_search
-> simulate_arx
-> residual_diagnostics
-> audit_data
-> ghi results
```

Output: `payload` chứa toàn bộ config, audit, model, metrics.

Log nên in nếu debug:

```python
payload = run_pipeline(cfg)
log_step("payload keys", payload.keys())
log_step("selected", payload["arx_inside"]["selected_by_validation"])
```

## 8. `parse_args`

Mục đích: đọc command line.

Các tham số:

- `--data-csv`;
- `--days`;
- `--sampling-seconds`;
- `--seed`.

Log:

```python
args = parse_args()
print(args)
```

## 9. `main`

Mục đích: điểm bắt đầu khi chạy file bằng Python.

Luồng:

```text
parse_args
-> tạo OutdoorConfig
-> run_pipeline
-> in kết quả ngắn
```

Dòng cuối:

```python
if __name__ == "__main__":
    main()
```

Ý nghĩa: chỉ chạy `main()` khi chạy trực tiếp file, không chạy khi import từ file khác.
