# Flow chương trình ARX model

Tài liệu này mô tả luồng chạy của chương trình ARX trong repo, tập trung vào file `arx_pipeline.py`, đồng thời ghi chú riêng phần notebook V5/V6/V7 có Ridge alpha search.

Mục tiêu của chương trình:

1. Đọc hoặc sinh dữ liệu greenhouse.
2. Chia dữ liệu thành Train, Validation và Test theo thứ tự thời gian.
3. Tạo ma trận hồi quy ARX từ dữ liệu quá khứ.
4. Ước lượng hệ số `theta_hat`.
5. Đánh giá mô hình bằng 1-step, 12-step và free-run simulation.
6. Lưu artifact model ra file `.json`.

## 1. Các file chính

| File | Vai trò |
| --- | --- |
| `arx_pipeline.py` | Pipeline ARX lõi: đọc data, split data, tạo ma trận `X/y`, fit OLS, simulate, tính metric, lưu artifact. |
| `arx_reporting.py` | Tạo dataframe phục vụ báo cáo: prediction frame, parameter frame, contribution, impulse response, grouped metrics. |
| `data_generator.py` | Sinh dữ liệu greenhouse gốc nếu không đọc CSV. |
| `data_generator_poly_arx.py` | Sinh dữ liệu có thành phần polynomial/phi tuyến. |
| `realtime_predict.py` | Load artifact `.json` và dự đoán realtime theo hệ số đã lưu. |
| `ARX_Model_Version 512/ARX_512_V7.ipynb` | Notebook thí nghiệm V7: dùng hàm lõi từ `arx_pipeline.py`, thêm feature engineering, missing handling, z-score, Ridge alpha search, lưu `arx_512_v7.json`. |
| `tools/build_512_version_notebooks.py` | Script tạo các notebook 512 version, có code Ridge và artifact cell. |

## 2. Tổng quan luồng chạy

Flow chuẩn khi chạy:

```bash
python arx_pipeline.py
```

Luồng gọi hàm:

```text
main()
  -> run_pipeline()
       -> load_or_generate_data()
            -> load_existing_csv()
            hoặc load_generator_from_script()
       -> split_time_series()
       -> build_regression_matrix()
       -> estimate_ols()
       -> build_true_theta()
       -> compute_ar_roots()
       -> summarize_parameters()
       -> evaluate_slice("Train")
       -> evaluate_slice("Validation")
       -> evaluate_slice("Test")
       -> model_selection_search()
            -> build_regression_matrix()
            -> estimate_ols()
            -> evaluate_slice("Validation")
       -> evaluate_candidate_order()
            -> build_regression_matrix()
            -> estimate_ols()
            -> evaluate_slice("Validation")
            -> evaluate_slice("Test")
  -> save_artifact()
       -> artifact_payload()
       -> _json_ready()
  -> print_cli_summary()
```

Flow tính hệ số model quan trọng nhất:

```text
Dữ liệu train
  -> build_regression_matrix(df_train, model_config)
       trả về X_train, y_train
  -> estimate_ols(X_train, y_train)
       trả về theta_hat, cov_hat, sigma2_hat
```

Câu ngắn gọn để trình bày:

> Hệ số của mô hình ARX được tính trong hàm `estimate_ols()`, sau khi `build_regression_matrix()` tạo xong ma trận hồi quy `X` và vector mục tiêu `y`.

## 3. Các dataclass cấu hình

### 3.1 `DataConfig`

Khai báo trong `arx_pipeline.py`.

Vai trò: quy định nguồn dữ liệu và cách sinh/đọc dữ liệu.

| Field | Ý nghĩa |
| --- | --- |
| `csv_path` | File CSV mặc định, hiện là `greenhouse_data.csv`. |
| `generator_script_path` | Script sinh data, mặc định `data_generator.py`. |
| `force_regenerate_from_script` | Nếu `True`, ưu tiên sinh lại data từ generator. Nếu `False`, ưu tiên đọc CSV có sẵn. |
| `auto_save_generated_csv` | Nếu sinh data mới thì có lưu ra CSV hay không. |
| `generated_days` | Số ngày dữ liệu cần sinh. |
| `generated_sampling_seconds` | Chu kỳ lấy mẫu, mặc định 300 giây. |
| `generated_seed` | Seed random để tái lập kết quả. |
| `generated_start_date` | Ngày bắt đầu dữ liệu sinh. |

### 3.2 `SplitConfig`

Vai trò: quy định tỉ lệ chia train/validation/test.

| Field | Ý nghĩa |
| --- | --- |
| `train_ratio` | Tỉ lệ train, mặc định `0.60`. |
| `val_ratio` | Tỉ lệ validation, mặc định `0.20`. |
| `test_ratio` | Property tính từ `1.0 - train_ratio - val_ratio`. |

Hàm liên quan:

| Hàm | Vai trò |
| --- | --- |
| `SplitConfig.test_ratio` | Tính tỉ lệ test còn lại. |
| `SplitConfig.validate()` | Kiểm tra các tỉ lệ hợp lệ và tổng bằng 1. |

### 3.3 `ModelConfig`

Vai trò: quy định cấu trúc ARX.

| Field | Ý nghĩa |
| --- | --- |
| `na` | Số bậc hồi quy theo output quá khứ. Ví dụ `na=5` nghĩa là dùng `y[t-1]` đến `y[t-5]`. |
| `nb` | Số bậc trễ của mỗi input. Ví dụ `nb=1` dùng một giá trị trễ của mỗi input. |
| `nk` | Độ trễ đầu vào. Ví dụ `nk=2` bắt đầu dùng `u[t-2]`. |
| `include_intercept` | Có thêm hệ số chặn/intercept hay không. |
| `input_cols` | Danh sách biến đầu vào. |
| `output_col` | Biến đầu ra, mặc định `Soil_Moisture`. |
| `simulation_clip` | Giới hạn giá trị khi free-run simulate, tránh drift quá xa. |

Property quan trọng:

| Property | Vai trò |
| --- | --- |
| `ModelConfig.param_names` | Tạo danh sách tên hệ số theo đúng thứ tự trong vector `theta`. |

Ví dụ với `ARX(5,1,2)` và 16 input, `param_names` có dạng:

```text
a1, a2, a3, a4, a5,
b_Temperature_1,
b_Humidity_1,
...
b_Season_cos_1,
intercept
```

Thứ tự này rất quan trọng, vì `theta_hat[i]` tương ứng với `param_names[i]`.

## 4. Đọc và chuẩn bị dữ liệu

### 4.1 `validate_dataframe(df_in, required_cols=None)`

Vai trò:

1. Kiểm tra dataframe có đủ các cột bắt buộc hay không.
2. Nếu có cột `Timestamp`, sắp xếp theo thời gian.
3. Reset index để dữ liệu liên tục.

Cột bắt buộc mặc định:

```text
Timestamp
Soil_Moisture
Temperature
Humidity
Light
Drip
Mist
Fan
```

Input/output:

| Thành phần | Ý nghĩa |
| --- | --- |
| `df_in` | DataFrame cần kiểm tra. |
| `required_cols` | Danh sách cột bắt buộc, nếu không truyền thì dùng mặc định. |
| Output | DataFrame đã validate và sắp xếp. |

Nếu thiếu cột, hàm raise `ValueError`.

### 4.2 `load_existing_csv(csv_path)`

Vai trò:

1. Đọc CSV bằng `pd.read_csv`.
2. Parse cột `Timestamp` thành datetime.
3. Gọi `validate_dataframe()`.

### 4.3 `load_generator_module(script_path)`

Vai trò: load file Python generator bằng `importlib.util`.

Hàm này chưa sinh data trực tiếp. Nó chỉ load module từ file, ví dụ `data_generator.py`.

### 4.4 `load_generator_from_script(script_path)`

Vai trò:

1. Gọi `load_generator_module()`.
2. Kiểm tra module có hàm `generate_greenhouse_data()` không.
3. Trả về function `generate_greenhouse_data`.

### 4.5 `extract_true_params_from_module(module)`

Vai trò: lấy tham số thật của data generator nếu có.

Thứ tự lấy:

1. Nếu module có `get_true_params()`, gọi hàm này.
2. Nếu module có biến `TRUE_PARAMS`, dùng biến này.
3. Nếu không có, trả về `None`.

`true_params` dùng để so sánh hệ số ước lượng với hệ số thật khi data là data giả lập.

### 4.6 `load_or_generate_data(config)`

Đây là hàm điều phối đọc/sinh data.

Logic:

```text
Nếu CSV tồn tại và force_regenerate_from_script = False:
    đọc CSV bằng load_existing_csv()
    nếu có generator thì đọc true_params
    trả về df, true_params, data_source="CSV:..."
Ngược lại:
    load generator từ script
    gọi generate_greenhouse_data(...)
    validate dataframe
    nếu auto_save_generated_csv=True thì lưu CSV
    trả về df, true_params, data_source="GENERATED:..."
```

Output:

| Output | Ý nghĩa |
| --- | --- |
| `df` | Dữ liệu full. |
| `true_params` | Tham số thật nếu data generator cung cấp. |
| `data_source` | Chuỗi mô tả nguồn data, ví dụ `CSV:greenhouse_data.csv`. |

## 5. Chia train, validation, test

### 5.1 `split_time_series(df, split_config)`

Vai trò: chia dữ liệu theo thứ tự thời gian, không shuffle.

Lý do không shuffle:

ARX là mô hình chuỗi thời gian. Nếu shuffle sẽ làm mất quan hệ quá khứ -> hiện tại, và có thể gây leakage.

Logic:

```text
n_total = len(df)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)
n_test_start = n_train + n_val

df_train = df[0 : n_train]
df_val   = df[n_train : n_test_start]
df_test  = df[n_test_start : end]
```

Output:

| Output | Ý nghĩa |
| --- | --- |
| `df_train` | Dữ liệu dùng để fit hệ số. |
| `df_val` | Dữ liệu dùng để chọn model/hyperparameter. |
| `df_test` | Dữ liệu dùng để đánh giá cuối. |

## 6. Tạo ma trận hồi quy ARX

### 6.1 Hàm trung tâm: `build_regression_matrix(df, model_config)`

Đây là hàm quan trọng thứ hai sau `estimate_ols()`.

Vai trò: biến chuỗi thời gian thành bài toán hồi quy tuyến tính:

```text
y = X theta
```

Với ARX:

```text
y[t] =
  a1*y[t-1] + a2*y[t-2] + ... + ana*y[t-na]
  + b1*u[t-nk] + b2*u[t-nk-1] + ...
  + intercept
```

Trong code, với mỗi thời điểm `t`, hàm tạo một row của `X` gồm:

1. Các output quá khứ: `y[t-1]`, `y[t-2]`, ..., `y[t-na]`.
2. Các input quá khứ: với từng input `u`, lấy `u[t-nk]` đến `u[t-(nk+nb-1)]`.
3. Nếu `include_intercept=True`, thêm `1.0` vào cuối row.

Code logic rút gọn:

```python
y = df[model_config.output_col].astype(float).to_numpy()
inputs = [df[col].astype(float).to_numpy() for col in model_config.input_cols]

max_lag = max(model_config.na, model_config.nb + model_config.nk - 1)
n_eff = len(y) - max_lag

for row_idx in range(n_eff):
    t = row_idx + max_lag
    row = []

    for lag in range(1, model_config.na + 1):
        row.append(y[t - lag])

    for u in inputs:
        for lag in range(model_config.nk, model_config.nk + model_config.nb):
            row.append(u[t - lag])

    if model_config.include_intercept:
        row.append(1.0)

    x_mat[row_idx] = row
    y_vec[row_idx] = y[t]
```

Output:

| Output | Ý nghĩa |
| --- | --- |
| `x_mat` | Ma trận hồi quy `X`, shape `(n_eff, n_params)`. |
| `y_vec` | Vector mục tiêu `y`, shape `(n_eff,)`. |

### 6.2 Ví dụ ARX(5,1,2)

Với cấu hình:

```text
na = 5
nb = 1
nk = 2
include_intercept = True
```

Mỗi row của `X` có dạng:

```text
[
  y[t-1],
  y[t-2],
  y[t-3],
  y[t-4],
  y[t-5],
  Temperature[t-2],
  Humidity[t-2],
  Light[t-2],
  Drip[t-2],
  Mist[t-2],
  Fan[t-2],
  ... các augmented features [t-2],
  1.0
]
```

Vector hệ số tương ứng:

```text
theta = [
  a1, a2, a3, a4, a5,
  b_Temperature_1,
  b_Humidity_1,
  ...
  intercept
]
```

Kết quả dự đoán 1-step:

```text
y_pred[t] = dot(X_row[t], theta)
```

## 7. Hàm tính hệ số model

### 7.1 Hàm chính trong `arx_pipeline.py`: `estimate_ols(x_mat, y_vec)`

Đây là hàm tính ra các hệ số `theta_hat` theo lý thuyết OLS.

Input/output:

| Thành phần | Ý nghĩa |
| --- | --- |
| `x_mat` | Ma trận hồi quy `X`. |
| `y_vec` | Vector output thực `y`. |
| `theta` | Vector hệ số ước lượng, chính là `theta_hat`. |
| `cov` | Ma trận covariance ước lượng của hệ số. |
| `sigma2` | Phương sai sai số ước lượng. |

Code cốt lõi:

```python
theta, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
```

Về lý thuyết, hàm đang giải bài toán:

```text
theta_hat = argmin_theta || y - X theta ||^2
```

Nếu `X^T X` khả nghịch, công thức tương đương:

```text
theta_hat = (X^T X)^(-1) X^T y
```

Sau khi có `theta`, hàm tính residual:

```text
resid = y - X theta
```

Tính `sigma2`:

```text
sigma2 = resid^T resid / (n_obs - n_params)
```

Tính covariance:

```text
cov = sigma2 * pinv(X^T X)
```

### 7.2 Điểm cần nói khi báo cáo

Nếu hỏi "hàm nào tính ra hệ số cuối cùng?", trả lời:

```text
estimate_ols() là hàm tính hệ số theta_hat bằng bình phương tối thiểu.
Trước đó, build_regression_matrix() tạo X và y.
```

Nếu là notebook V5/V6/V7 có Ridge:

```text
estimate_ols() tạo nghiệm OLS ban đầu.
estimate_ridge() trong notebook/tool tính nghiệm Ridge cho từng alpha.
alpha_search_df chọn alpha tốt nhất.
theta_hat = theta_by_alpha[best_alpha].
```

## 8. Tính metric đánh giá

### 8.1 `compute_metrics(y_true, y_pred, n_params)`

Vai trò: tính các chỉ số đánh giá dự đoán.

| Metric | Ý nghĩa |
| --- | --- |
| `RMSE` | Căn bậc hai trung bình bình phương sai số. Càng nhỏ càng tốt. |
| `MAE` | Trung bình trị tuyệt đối sai số. Càng nhỏ càng tốt. |
| `Bias` | Trung bình sai số `y_true - y_pred`. Gần 0 là tốt. |
| `FIT` | Độ fit theo phần trăm. Càng cao càng tốt. |
| `R2` | Hệ số xác định. Gần 1 là tốt. |
| `AIC` | Akaike Information Criterion. Dùng so sánh model, càng nhỏ càng tốt. |
| `BIC` | Bayesian Information Criterion. Dùng so sánh model, càng nhỏ càng tốt. |

Công thức chính:

```text
resid = y_true - y_pred
RMSE = sqrt(mean(resid^2))
MAE = mean(abs(resid))
Bias = mean(resid)
FIT = 100 * (1 - ||resid|| / ||y_true - mean(y_true)||)
R2 = 1 - sum(resid^2) / sum((y_true - mean(y_true))^2)
```

## 9. Mô phỏng ARX

Trong code có 3 chế độ đánh giá dự đoán:

1. 1-step prediction.
2. n-step prediction, mặc định 12-step.
3. free-run simulation.

### 9.1 1-step prediction

Nằm trong `evaluate_slice()`.

Logic:

```python
x_mat, y_vec = build_regression_matrix(df_slice, model_config)
y_pred = x_mat @ theta
```

Đặc điểm:

Mỗi thời điểm dự đoán dùng output quá khứ thật. Vì vậy metric thường cao hơn free-run.

### 9.2 `simulate_arx(df_sim, theta, model_config)`

Vai trò: mô phỏng free-run. Model dùng chính output đã dự đoán ở bước trước để dự đoán bước tiếp theo.

Khác biệt quan trọng:

```text
1-step: dùng y thực tế quá khứ
free-run: dùng y_sim đã dự đoán quá khứ
```

Logic:

```text
y_sim ban đầu copy từ y thật
for t từ max_lag đến cuối chuỗi:
    row gồm y_sim[t-1], y_sim[t-2], ...
    row gồm input thật tại các lag
    y_next = dot(row, theta)
    nếu simulation_clip != None:
        y_next = clip(y_next, low, high)
    y_sim[t] = y_next
```

Output:

| Output | Ý nghĩa |
| --- | --- |
| `y_sim[max_lag:]` | Chuỗi dự đoán free-run. |
| `y[max_lag:]` | Chuỗi giá trị thật tương ứng. |

### 9.3 `simulate_arx_n_step(df_sim, theta, n_steps, model_config)`

Vai trò: mô phỏng n-step ahead. Mặc định trong pipeline là `n_step=12`.

Ý tưởng:

Tại mỗi thời điểm `t`, model chỉ được free-run tối đa `n_steps` bước từ một mốc gần đó. Nó nằm giữa 1-step và full free-run.

So sánh:

| Chế độ | Dùng output quá khứ nào | Thường dùng để làm gì |
| --- | --- | --- |
| 1-step | Output thật | Kiểm tra khả năng fit cục bộ. |
| 12-step | Output dự đoán trong cửa sổ ngắn | Kiểm tra dự báo ngắn hạn. |
| free-run | Output dự đoán từ đầu đến cuối | Kiểm tra khả năng mô phỏng động học dài hạn. |

## 10. Đánh giá một split

### 10.1 `evaluate_slice(name, df_slice, theta, model_config, true_theta=None, n_step=12)`

Đây là hàm gom tất cả cách đánh giá cho một tập dữ liệu.

Bên trong hàm:

```text
1. Tạo X, y bằng build_regression_matrix()
2. Tính y_pred 1-step bằng X @ theta
3. Tính y_sim free-run bằng simulate_arx()
4. Tính y_n_step bằng simulate_arx_n_step()
5. Tính metrics_1step bằng compute_metrics()
6. Tính metrics_n_step bằng compute_metrics()
7. Tính metrics_sim bằng compute_metrics()
8. Tính residual_diagnostics()
9. Tính summarize_dataset_behavior()
10. Nếu có true_theta, tính theoretical_max_free_run
```

Output dict gồm:

| Key | Ý nghĩa |
| --- | --- |
| `name` | Tên split. |
| `metrics_1step` | Metric cho 1-step prediction. |
| `metrics_n_step` | Metric cho n-step prediction. |
| `metrics_sim` | Metric cho free-run simulation. |
| `arrays` | Các mảng `y_true`, `y_pred` để vẽ biểu đồ/báo cáo. |
| `residual_diagnostics` | Chẩn đoán residual. |
| `behavior` | Tóm tắt hành vi dữ liệu trong split. |
| `theoretical_max_free_run` | Nếu có true params, metric khi dùng hệ số thật. |

## 11. Chẩn đoán residual và tham số

### 11.1 `residual_diagnostics(residuals, df_slice, model_config, max_lag=20)`

Vai trò: kiểm tra residual sau khi fit model.

Nó tính:

| Nhóm | Nội dung |
| --- | --- |
| `mean`, `std` | Trung bình và độ lệch chuẩn residual. |
| `normality` | Shapiro test và D'Agostino normality test. |
| `ljung_box` | Ljung-Box test để xem residual còn tự tương quan không. |
| `input_cross_correlation` | Tương quan giữa residual và từng input. |

Ý nghĩa:

Nếu residual còn tự tương quan hoặc còn tương quan với input, model chưa bắt hết cấu trúc động học của dữ liệu.

### 11.2 `summarize_parameters(theta, cov, model_config, true_params)`

Vai trò: tạo bảng tóm tắt hệ số.

Mỗi row gồm:

| Field | Ý nghĩa |
| --- | --- |
| `name` | Tên hệ số, lấy từ `model_config.param_names`. |
| `estimate` | Giá trị ước lượng. |
| `std` | Standard error từ covariance. |
| `ci95_low`, `ci95_high` | Khoảng tin cậy 95%. |
| `true_value` | Giá trị thật nếu có. |
| `delta_vs_true` | Sai lệch so với hệ số thật. |
| `sign_ok` | Dấu của hệ số ước lượng có khớp dấu hệ số thật không. |

### 11.3 `parameter_reference_map(true_params)`

Vai trò: map tên tham số trong generator sang tên tham số trong model.

Ví dụ:

```text
b_temp_1 -> b_Temperature_1
b_humi_1 -> b_Humidity_1
```

### 11.4 `build_true_theta(true_params, model_config)`

Vai trò: tạo vector `true_theta` theo đúng thứ tự `model_config.param_names`.

Nếu không đủ true params, trả về `None`.

### 11.5 `compute_ar_roots(theta, model_config)`

Vai trò: tính nghiệm/cực AR của model để xem phần hồi quy theo output có ổn định không.

Lưu ý:

Trong `arx_pipeline.py`, hàm này hiện chỉ xử lý khi `na == 2`.

## 12. Model selection

### 12.1 `model_selection_search(df_train, df_val, base_model_config, na_list, nb_list, nk_list)`

Vai trò: thử nhiều cấu hình ARX khác nhau và sắp xếp theo hiệu năng free-run.

Logic:

```text
for na in na_list:
  for nb in nb_list:
    for nk in nk_list:
      tạo ModelConfig candidate
      X_train, y_train = build_regression_matrix(df_train, candidate)
      theta = estimate_ols(X_train, y_train)
      evaluation = evaluate_slice("Validation", df_val, theta, candidate)
      lưu RMSE/FIT/R2/AIC/BIC
```

Sau đó sắp xếp theo:

```text
RMSE_sim tăng dần
RMSE_1step tăng dần
AIC_1step tăng dần
n_params tăng dần
```

Nghĩa là ưu tiên free-run simulation tốt trước.

### 12.2 `evaluate_candidate_order(df_train, df_val, df_test, base_model_config, na, nb, nk)`

Vai trò:

Sau khi `model_selection_search()` chọn được order tốt trên validation, hàm này fit lại candidate đó trên train và đánh giá trên validation/test.

Output gồm:

| Key | Ý nghĩa |
| --- | --- |
| `model_config` | Cấu hình candidate. |
| `theta_hat` | Hệ số candidate. |
| `sigma2` | Phương sai residual. |
| `val` | Metric validation. |
| `test` | Metric test. |

## 13. Hàm điều phối tổng: `run_pipeline()`

Đây là hàm trung tâm của file `arx_pipeline.py`.

Chu kỳ chạy:

```text
1. Nhận data_config, split_config, model_config
2. Nếu user không truyền thì dùng mặc định
3. df, true_params, data_source = load_or_generate_data(data_cfg)
4. df_train, df_val, df_test = split_time_series(df, split_cfg)
5. X_train, y_train = build_regression_matrix(df_train, model_cfg)
6. theta_hat, cov_hat, sigma2_hat = estimate_ols(X_train, y_train)
7. true_theta = build_true_theta(true_params, model_cfg)
8. Tạo dataset_overview
9. Tính ar_roots
10. Tính parameter_summary
11. evaluate_slice cho Train
12. evaluate_slice cho Validation
13. evaluate_slice cho Test
14. Chạy model_selection_search
15. Nếu có best order, chạy evaluate_candidate_order
16. Trả về results dict
```

Output `results` có các key quan trọng:

| Key | Ý nghĩa |
| --- | --- |
| `data_source` | Nguồn data. |
| `data_config` | Cấu hình data. |
| `split_config` | Cấu hình split. |
| `model_config` | Cấu hình model. |
| `df_full`, `df_train`, `df_val`, `df_test` | Dữ liệu full và các split. |
| `true_params` | Hệ số thật nếu có. |
| `dataset_overview` | Tóm tắt dữ liệu và tính chất ma trận train. |
| `theta_hat` | Hệ số model đã fit. |
| `sigma2` | Phương sai residual. |
| `ar_roots` | Nghiệm AR nếu tính được. |
| `parameter_summary` | Bảng tóm tắt hệ số. |
| `train`, `val`, `test` | Kết quả đánh giá từng split. |
| `model_selection` | Bảng thử các order ARX. |
| `best_candidate` | Candidate tốt nhất theo model selection. |

## 14. Lưu artifact JSON

### 14.1 `_json_ready(value)`

Vai trò: chuyển các object Python/Numpy/Pandas khó serialize thành kiểu JSON-friendly.

| Kiểu gốc | Chuyển thành |
| --- | --- |
| `Path` | `str` |
| `np.ndarray` | `list` |
| `np.floating`, `np.integer` | scalar Python |
| `tuple` | `list` |
| `dict`, `list` | Đệ quy từng phần tử |

### 14.2 `artifact_payload(results)`

Vai trò: lấy `results` từ `run_pipeline()` và tạo object gọn hơn để lưu ra JSON.

Payload gồm:

| Key | Ý nghĩa |
| --- | --- |
| `model` | Tên model, ở đây là `ARX`. |
| `data_source` | Nguồn data. |
| `data_config` | Cấu hình data. |
| `split_config` | Cấu hình split. |
| `model_config` | Cấu hình ARX. |
| `dataset_overview` | Tóm tắt dataset. |
| `param_names` | Tên các hệ số. |
| `theta_hat` | Hệ số model. |
| `sigma2` | Phương sai residual. |
| `ar_roots` | Nghiệm AR. |
| `parameter_summary` | Tóm tắt tham số. |
| `slices` | Metric train/val/test. |
| `model_selection_top10` | Top 10 order trong model selection. |
| `true_params` | Hệ số thật nếu có. |
| `summary` | Tóm tắt nhanh. |
| `best_candidate` | Candidate tốt nhất nếu có. |

### 14.3 `save_artifact(results, output_path=Path("arx_model.json"))`

Vai trò: ghi payload ra file JSON.

Logic:

```python
payload = artifact_payload(results)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
```

Đây là lý do model trong project này được lưu bằng `.json`, không phải `.pkl`.

## 15. CLI entrypoint

### 15.1 `main()`

Hàm này chạy khi gõ:

```bash
python arx_pipeline.py
```

Logic:

```text
results = run_pipeline()
save_artifact(results)
print_cli_summary(results)
print("Saved model artifact to arx_model.json")
```

### 15.2 `print_cli_summary(results)`

Vai trò: in tóm tắt ra terminal:

1. Dataset source, số row, khoảng thời gian.
2. Rank và condition number của `X_train`.
3. FIT của Train/Validation/Test theo 1-step, 12-step, free-run.
4. Bảng hệ số và dấu hệ số so với true params.
5. Residual diagnostics trên Validation.
6. Best model trong model selection.

## 16. Flow riêng của notebook V7 và các bản 512

File `ARX_Model_Version 512/ARX_512_V7.ipynb` không chỉ dùng `run_pipeline()` nguyên bản. Notebook này dùng các hàm lõi trong `arx_pipeline.py`, sau đó thêm các bước riêng:

1. Xử lý missing data.
2. Feature engineering thêm augmented features.
3. Z-score normalization.
4. Tạo `MODEL_CONFIG`.
5. Tạo `X_train`, `y_train`.
6. Fit OLS ban đầu.
7. Ridge alpha search.
8. Chọn `best_alpha`.
9. Gán `theta_hat = theta_by_alpha[best_alpha]`.
10. Đánh giá Train/Validation/Test.
11. Lưu artifact `arx_512_v7.json`.

### 16.1 Các hàm được dùng lại từ `arx_pipeline.py`

| Hàm | Vai trò trong notebook |
| --- | --- |
| `ModelConfig` | Định nghĩa cấu hình ARX. |
| `build_regression_matrix()` | Tạo `X_train`, `y_train`. |
| `estimate_ols()` | Tính nghiệm OLS ban đầu. |
| `evaluate_slice()` | Đánh giá 1-step, 12-step, free-run trên data đã normalize. |
| `compute_metrics()` | Tính metric sau khi inverse về đơn vị thật. |
| `compute_ar_roots()` | Tính nghiệm AR để lưu artifact. |

### 16.2 Hàm Ridge trong notebook/tool

Trong notebook V7 và `tools/build_512_version_notebooks.py` có hàm:

```python
def estimate_ridge(x_mat: np.ndarray, y_vec: np.ndarray, alpha: float, penalize_intercept: bool = False) -> np.ndarray:
    ...
```

Về lý thuyết Ridge giải:

```text
theta_hat = argmin_theta || y - X theta ||^2 + alpha * ||theta||^2
```

Nếu không phạt intercept, phần intercept sẽ không bị tính vào penalty.

Flow trong notebook:

```text
X_train, y_train = build_regression_matrix(df_train_m, MODEL_CONFIG)
theta_ols, cov_ols, sigma2_ols = estimate_ols(X_train, y_train)

for alpha in alpha_grid:
    if alpha == 0:
        theta = theta_ols
    else:
        theta = estimate_ridge(X_train, y_train, alpha)
    evaluate_real("Train", df_train_m, theta)
    evaluate_real("Validation", df_val_m, theta)
    evaluate_real("Test", df_test_m, theta)
    lưu vào alpha_search_df

best_alpha = alpha_search_df.iloc[0]["alpha"]
theta_hat = theta_by_alpha[best_alpha]
```

Bảng phân biệt:

| Trường hợp | Hàm tính hệ số cuối |
| --- | --- |
| Pipeline OLS gốc `arx_pipeline.py` | `estimate_ols()` |
| Notebook V7/Ridge | `estimate_ridge()` cho alpha tốt nhất, hoặc `estimate_ols()` nếu `best_alpha = 0` |
| Artifact final ARX-only có `regularization_method = ridge` | Hệ số cuối đến từ Ridge alpha search |

## 17. Vai trò của `arx_reporting.py`

File này không fit model mới. Nó chỉ lấy `results` đã có và tạo bảng/dữ liệu để vẽ hình, báo cáo.

### 17.1 Các hàm báo cáo chính

| Hàm | Vai trò |
| --- | --- |
| `max_lag(results)` | Tính max lag của model. |
| `prediction_frame_for_split(results, split_key)` | Tạo dataframe `y_true`, `y_pred_1step`, `y_pred_12step`, `y_pred_sim` cho một split. |
| `combined_prediction_frame(results)` | Gộp prediction frame của train/val/test. |
| `parameter_frame(results)` | Chuyển `parameter_summary` thành DataFrame. |
| `standardized_parameter_frame(results, split_key="train")` | Tính beta chuẩn hóa: `theta_i * std(x_i) / std(y)`. |
| `behavior_frame(results)` | Gom behavior summary của train/val/test. |
| `monthly_signal_summary(df)` | Tóm tắt Soil/Temp/Humidity/Light theo tháng. |
| `monthly_actuator_summary(df)` | Tóm tắt tỉ lệ bật/tắt Drip/Mist/Fan theo tháng. |
| `monthly_setpoint_summary(df)` | Tóm tắt Soil so với setpoint theo tháng. |
| `rolling_rmse(y_true, y_pred, window)` | Tính RMSE trượt. |
| `contribution_frame(results, split_key="train")` | Tính đóng góp trung bình của từng feature/hệ số. |
| `impulse_response_frame(results, split_key="train", horizon=48, pulse_mode="std")` | Tính đáp ứng xung của model với từng input. |
| `grouped_metrics(prediction_df, group_cols, modes=None)` | Tính metric theo nhóm, ví dụ theo tháng/mùa. |
| `model_comparison_frame(results)` | Tạo bảng so sánh baseline và best candidate. |
| `selection_metric_grid(selection_df, metric, nk)` | Pivot bảng model selection để vẽ heatmap/grid. |

## 18. Giải thích các biến quan trọng

| Biến | Xuất hiện ở đâu | Ý nghĩa |
| --- | --- | --- |
| `df` | `load_or_generate_data()` | Dữ liệu full. |
| `df_train`, `df_val`, `df_test` | `split_time_series()` | Ba split theo thời gian. |
| `x_mat`, `X_train` | `build_regression_matrix()` | Ma trận feature hồi quy ARX. |
| `y_vec`, `y_train` | `build_regression_matrix()` | Vector target tương ứng. |
| `theta_hat` | `estimate_ols()` hoặc Ridge search | Hệ số model cuối cùng. |
| `cov_hat` | `estimate_ols()` | Covariance của hệ số. |
| `sigma2_hat` | `estimate_ols()` | Phương sai residual train. |
| `true_theta` | `build_true_theta()` | Hệ số thật nếu data giả lập. |
| `metrics_1step` | `evaluate_slice()` | Metric khi dự đoán 1-step. |
| `metrics_n_step` | `evaluate_slice()` | Metric khi dự đoán n-step, mặc định 12-step. |
| `metrics_sim` | `evaluate_slice()` | Metric free-run simulation. |
| `model_selection` | `model_selection_search()` | Bảng thử nhiều cấu hình ARX. |
| `best_candidate` | `evaluate_candidate_order()` | Cấu hình tốt nhất theo validation search. |

## 19. Nội dung artifact `.json`

Một file model JSON thường lưu:

```text
model/model_type
data_source
data_config
split_config
model_config
param_names
theta_hat
sigma2
metrics
ar_roots
alpha_search nếu có Ridge
comparison nếu có so sánh version
```

Để predict lại, cần tối thiểu:

1. `model_config`
2. `param_names`
3. `theta_hat`
4. Cách tạo đúng feature đầu vào
5. Nếu có z-score/normalization thì cần stats tương ứng
6. Nếu có `simulation_clip` thì dùng lại clip bounds

## 20. Tóm tắt ngắn để thuyết trình

Có thể trình bày luồng chương trình bằng 5 câu:

1. Chương trình đọc hoặc sinh dữ liệu greenhouse bằng `load_or_generate_data()`, sau đó validate cột và sắp xếp theo `Timestamp`.
2. Dữ liệu được chia theo thứ tự thời gian bằng `split_time_series()` thành Train, Validation và Test.
3. Hàm `build_regression_matrix()` biến bài toán chuỗi thời gian ARX thành bài toán hồi quy tuyến tính `y = X theta`.
4. Hàm `estimate_ols()` tính hệ số `theta_hat` bằng bình phương tối thiểu; trong các notebook Ridge thì `estimate_ridge()` tính hệ số cho từng `alpha` và chọn `best_alpha`.
5. Hàm `evaluate_slice()` đánh giá 1-step, 12-step và free-run simulation, sau đó `save_artifact()` lưu model ra JSON.

## 21. Câu trả lời trực tiếp cho câu hỏi "hàm nào tính ra kết quả cuối cùng?"

Nếu nói pipeline ARX gốc:

```text
build_regression_matrix() tạo X, y.
estimate_ols() tính theta_hat.
```

Nếu nói notebook/version có Ridge:

```text
build_regression_matrix() tạo X, y.
estimate_ols() tính theta_ols.
estimate_ridge() tính theta cho từng alpha.
alpha_search_df chọn best_alpha.
theta_hat = theta_by_alpha[best_alpha].
```

Vì artifact `arx_model_algo_final_arx_only.json` ghi:

```text
regularization_method = ridge
regularization_alpha = 0.001
```

nên với artifact final ARX-only, hệ số cuối cùng là `theta_hat` sau Ridge alpha search, không chỉ là OLS thuần.

