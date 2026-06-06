# Build, train, validate và test

## 1. Luồng tổng quát

Pipeline residual chạy theo thứ tự:

```text
load data
-> audit data
-> add feature hợp lệ
-> split theo thời gian
-> fit scaler bằng train
-> chọn ARX backbone bằng validation
-> train residual trên train
-> chọn residual bằng validation
-> báo cáo test cuối
```

## 2. Chạy pipeline

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py
```

Với CSV thật:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py --data-csv .\data\log_that.csv
```

## 3. Chia dữ liệu

Chia theo thời gian:

```text
train      70%
validation 15%
test       15%
```

Không shuffle vì đây là dữ liệu chuỗi thời gian. Shuffle sẽ làm model nhìn thấy phân phối quá gần tương lai và dễ đánh giá ảo.

## 4. Chọn ARX backbone

ARX backbone hiện tại:

```text
ARX_na12_nb6_nk1_alpha10
```

Backbone này được chọn bằng validation robust score, không chọn bằng test.

## 5. Train residual

Residual target trên train:

```text
residual_train = y_true_train - y_arx_sim_train
```

Feature residual gồm:

- `y_arx_sim`;
- các lag của `y_arx_sim`: `1, 2, 3, 6, 12`;
- các lag input quá khứ: `2, 3, 6, 12, 24`.

Không dùng `Soil_Moisture` thật tương lai trong validation/test.

## 6. Chọn residual

Residual thử:

- Ridge;
- HistGradientBoostingRegressor;
- `shrink = 0, 0.25, 0.5, 0.75, 1.0`.

Tiêu chí chọn:

```text
validation robust score = mean(block FIT_sim) - 0.5 * std(block FIT_sim)
```

Có `shrink=0`, nên nếu residual không tốt thì pipeline có quyền quay về ARX backbone.

## 7. File kết quả

Sau khi chạy sẽ có:

- `results/metrics.json`: toàn bộ metrics và cấu hình;
- `results/leaderboard.csv`: các candidate residual;
- `results/SUMMARY.md`: tóm tắt kết quả;
- `results/test_predictions.csv`: y thật, y ARX và y Hybrid trên test.

## 8. Nguyên tắc bảo vệ

Không nói:

```text
Em chọn candidate có test cao nhất.
```

Nói đúng:

```text
Em chọn candidate đứng đầu validation robust score. Test chỉ dùng để báo cáo cuối.
```
