# Build, train, validate, test

## 1. Mục tiêu của pipeline

Pipeline phải trả lời được:

```text
Nếu có một CSV time-series, model ARX được train thế nào, chọn model thế nào, test thế nào, có leakage không?
```

Pipeline chính:

[arx_final_pipeline.py](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/src/arx_final_pipeline.py>)

Pipeline residual đã tách riêng:

[arx_residual_experiment.py](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Residual_Final_Clean/src/arx_residual_experiment.py>)

## 2. Chạy ARX thuần

```powershell
python -B .\ARX_PBL5_Final_Clean\src\arx_final_pipeline.py
```

Output:

```text
ARX_PBL5_Final_Clean/results
```

File quan trọng:

- `metrics.json`: toàn bộ config, audit, validation, test;
- `comparison.csv`: bảng kết quả ngắn;
- `arx_inside_leaderboard.csv`: các ARX candidate;
- `test_predictions.csv`: y thật và y dự đoán trên test;
- `FINAL_REPORT.md`: báo cáo tự sinh;
- `SELF_CRITIQUE.md`: tự phản biện tự sinh.

## 3. Chạy Hybrid ARX residual

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py
```

Output:

```text
ARX_PBL5_Residual_Final_Clean/results
```

File quan trọng:

- `metrics.json`: kết quả chi tiết;
- `leaderboard.csv`: residual candidate;
- `SUMMARY.md`: tóm tắt;
- `test_predictions.csv`: so sánh ARX và Hybrid.

## 4. Chạy với dữ liệu thật

```powershell
python -B .\ARX_PBL5_Final_Clean\src\arx_final_pipeline.py --data-csv .\data\log_that.csv
```

CSV thật tối thiểu:

```csv
Timestamp,Soil_Moisture,Temperature_In,Humidity_In,Light_In,Drip,Fan
```

Nếu thiếu cột bắt buộc, script sẽ báo lỗi.

## 5. Chia dữ liệu

Pipeline chia theo thời gian:

```text
Train:      70%
Validation: 15%
Test:       15%
```

Không shuffle vì đây là time-series.

Vai trò:

- train: học hệ số;
- validation: chọn order/model;
- test: báo cáo cuối.

## 6. Scale dữ liệu

Scaler chỉ fit trên train:

```text
mean_train, std_train
```

Sau đó dùng cùng mean/std để scale validation và test. Không fit scaler trên toàn bộ data vì như vậy test đã rò thông tin vào train.

## 7. Chọn ARX

ARX search thử nhiều cấu hình:

- `na`: memory output;
- `nb`: memory input;
- `nk`: delay input;
- `alpha`: Ridge regularization.

Tiêu chí chọn:

```text
validation robust score = mean(block FIT_sim) - 0.5 * std(block FIT_sim)
```

Lý do dùng robust score:

- không chọn model chỉ tốt ở một đoạn validation;
- phạt model dao động lớn giữa các block;
- gần với yêu cầu thực tế hơn.

## 8. Chọn residual

Residual cũng chọn bằng validation robust score. Có `shrink=0`, nghĩa là nếu residual không tốt thì pipeline chọn không sửa.

Không được lấy candidate có test cao nhất nếu validation không chọn.

## 9. Checklist không leakage

- Không shuffle.
- Không fit scaler trên toàn bộ data.
- Không dùng test để chọn model.
- Không dùng output tương lai.
- Residual validation/test chỉ dùng `y_arx_sim` và input lag quá khứ.
- Test chỉ báo cáo sau khi chọn bằng validation.
