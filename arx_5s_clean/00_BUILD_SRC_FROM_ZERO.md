# Build source code từ đầu

Tài liệu này dùng khi cần dựng lại bản `arx_5s_clean` từ một thư mục trống. Mục tiêu là tạo đúng source code, đúng luồng data, rồi train ra model.

## 1. Yêu cầu môi trường

Cần Python và 2 thư viện chính:

```powershell
pip install numpy pandas
```

Nếu dùng `source legacy` thì cần repo còn lịch sử Git, vì code lấy lại data mẫu cũ bằng `git show`. Nếu chỉ dùng raw thật thì không cần quan tâm phần đó.

## 2. Tạo cấu trúc thư mục

Tạo cấu trúc tối thiểu như sau:

```text
arx_5s_clean/
  README.md
  00_BUILD_SRC_FROM_ZERO.md
  00_BUILD_FROM_RAW.md
  data/
    01_raw_sessions/
    01_raw_sessions_real/
    02_cleaned_sessions/
  results/
  scripts/
    01_build_data_from_collection.py
    02_train.py
  src/
    arx5s_clean/
      __init__.py
      config.py
      pipeline.py
      algorithm/
      data/
      evaluation/
      preprocessing/
      utils/
```

Không cần tạo `generator.py` hoặc `train.py` cũ. Bản hiện tại dùng script đánh số `01_...`, `02_...`.

## 3. Dựng package gốc

Tạo package:

```text
src/arx5s_clean/__init__.py
src/arx5s_clean/config.py
```

Trong `config.py` định nghĩa:

- `INPUT_COLS`: các input dùng cho ARX sau feature engineering.
- `ExperimentConfig`: cấu hình sampling 5 giây, số ngày, split train/validation/test, horizon đánh giá.

Raw CSV vẫn chỉ có 8 cột:

```csv
Timestamp,Temperature,Humidity,Light,Soil_Moisture,Drip,Mist,Fan
```

Các feature phụ được tạo trong code, không nhập thêm vào raw CSV.

## 4. Dựng phần data collection

Tạo thư mục:

```text
src/arx5s_clean/data/
src/arx5s_clean/data/collection/
```

Các file cần có:

```text
step_00_schema.py
step_00_environment.py
step_00_source_data.py
step_01_raw_sessions.py
step_02_timestamp_cleaning.py
step_03_missing_data.py
step_04_preprocess.py
step_05_augmentation.py
step_07_pipeline.py
__init__.py
```

Vai trò từng file:

- `step_00_schema.py`: khai báo 8 cột raw, sampling 5s, range hợp lệ, alias cột cũ như `Temperature_In -> Temperature`.
- `step_00_environment.py`: mô phỏng nền môi trường để sinh thêm ngày train, gồm nhiệt độ, độ ẩm, light và soil response.
- `step_00_source_data.py`: chọn `source legacy` hoặc `source real`, đọc CSV và chuẩn hóa cột.
- `step_01_raw_sessions.py`: tạo raw session từ source, xuất CSV vào `data/01_raw_sessions/`.
- `step_02_timestamp_cleaning.py`: parse `Timestamp`, sort, gom timestamp trùng, reindex về lưới 5 giây.
- `step_03_missing_data.py`: xử lý missing data, clip sensor theo range, ép `Drip/Mist/Fan` về `0/1`.
- `step_04_preprocess.py`: nối timestamp cleaning và missing data thành bước clean raw.
- `step_05_augmentation.py`: từ raw đã clean, sinh đủ data train 12 ngày và giả lập `Light` nếu cần.
- `step_07_pipeline.py`: nối toàn bộ data pipeline, xuất file data cuối.

Luồng chạy thực tế:

```text
01_build_data_from_collection.py
-> step_07_pipeline.py
   -> step_00_source_data.py
   -> step_01_raw_sessions.py
   -> step_04_preprocess.py
      -> step_02_timestamp_cleaning.py
      -> step_03_missing_data.py
   -> step_05_augmentation.py
```

Output của phần data:

```text
data/01_raw_sessions/00_raw_tong_hop.csv
data/02_cleaned_sessions/00_sau_xu_ly_tong_hop.csv
data/mini_greenhouse_5s_data.csv
```

## 5. Dựng preprocessing

Tạo:

```text
src/arx5s_clean/preprocessing/features.py
src/arx5s_clean/preprocessing/scaling.py
src/arx5s_clean/preprocessing/__init__.py
```

`features.py` nhận 8 cột raw và tạo thêm feature cho model:

- `Light_log`
- `Temp_x_Humidity`
- `Temp_x_Light`
- `Humidity_x_Light`
- `Air_Dryness`
- `Dry_Air_Effect`
- `Hour_sin`, `Hour_cos`
- `Day_sin`, `Day_cos`

`scaling.py` làm 3 việc:

- split data theo ngày và cùng khung giờ validation/test;
- fit mean/std trên train;
- scale train/validation/test và inverse `Soil_Moisture` khi tính metric.

Nguyên tắc: fit scaler chỉ trên train, không fit trên validation/test.

## 6. Dựng thuật toán ARX

Tạo:

```text
src/arx5s_clean/algorithm/specs.py
src/arx5s_clean/algorithm/arx.py
src/arx5s_clean/algorithm/__init__.py
```

`specs.py` chứa:

- `ArxSpec`: `na`, `nb`, `nk`, `alpha`.
- `default_specs(grid)`: danh sách cấu hình `tiny`, `quick`, `wide`.

`arx.py` chứa:

- `build_arx_matrix`: tạo ma trận hồi quy từ lag output và lag input.
- `fit_arx`: fit tham số bằng least squares hoặc ridge khi `alpha > 0`.
- `predict_one_step`: dự đoán 1 bước.
- `simulate_chunked`: mô phỏng theo đoạn 5 phút hoặc 20 phút.
- `simulate_free_run`: mô phỏng free-run.

Ý nghĩa cấu hình model:

- `na`: số lag của `Soil_Moisture`.
- `nb`: số lag của input.
- `nk`: delay input.
- `alpha`: regularization để tránh model quá nhạy.

## 7. Dựng evaluation

Tạo:

```text
src/arx5s_clean/evaluation/metrics.py
src/arx5s_clean/evaluation/simulation.py
src/arx5s_clean/evaluation/__init__.py
```

`metrics.py` tính:

- `FIT`
- `RMSE`
- `Bias`

`simulation.py` gọi model theo 4 kiểu đánh giá:

- `metrics_1step`
- `metrics_5min_chunked`
- `metrics_20min_chunked`
- `metrics_free_run`

## 8. Dựng utils

Tạo:

```text
src/arx5s_clean/utils/io.py
src/arx5s_clean/utils/__init__.py
```

Chỉ cần các hàm:

- `ensure_dir`
- `json_ready`
- `write_json`

Mục đích là ghi `metrics.json` và `arx_5s_model.json` ổn định.

## 9. Dựng training pipeline

Tạo:

```text
src/arx5s_clean/pipeline.py
```

File này là trung tâm train model:

1. Đọc `data/mini_greenhouse_5s_data.csv`.
2. Chuẩn hóa 8 cột raw.
3. Tạo feature bằng `add_features`.
4. Split train/validation/test.
5. Scale data.
6. Fit nhiều cấu hình ARX.
7. Chọn model tốt nhất trên validation.
8. Đánh giá lại trên train, validation, test.
9. Xuất kết quả.

Output chính:

```text
results/leaderboard.csv
results/metrics.json
results/arx_5s_model.json
results/test_predictions.csv
results/SUMMARY.md
```

## 10. Dựng scripts chạy ngoài terminal

Tạo:

```text
scripts/01_build_data_from_collection.py
scripts/02_train.py
```

`01_build_data_from_collection.py` nhận:

```text
--source legacy|real
--days 12
--seed 505031
--real-dir path
```

`02_train.py` nhận:

```text
--days 12
--seed 505031
--grid tiny|quick|wide
```

Hai script này thêm `src/` vào `sys.path`, nên không cần cài package bằng `pip install -e .`.

## 11. Kiểm tra source code

Sau khi dựng xong source, kiểm tra compile:

```powershell
cd C:\Users\minht\OneDrive\Desktop\ARX-Model\arx_5s_clean
python -B -m compileall .\src .\scripts
```

Chạy thử bằng data mẫu cũ:

```powershell
python -B .\scripts\01_build_data_from_collection.py --source legacy --days 12
python -B .\scripts\02_train.py --days 12 --grid quick
```

Khi có raw thật:

```powershell
python -B .\scripts\01_build_data_from_collection.py --source real --days 12
python -B .\scripts\02_train.py --days 12 --grid quick
```

## 12. Thứ tự dựng nhanh

Nếu làm lại thật sự từ đầu, đi theo thứ tự này:

```text
1. folder structure
2. config.py
3. data/collection/step_00_schema.py
4. data/collection/step_00_source_data.py
5. data/collection/step_01_raw_sessions.py
6. step_02_timestamp_cleaning.py + step_03_missing_data.py
7. step_04_preprocess.py + step_05_augmentation.py + step_07_pipeline.py
8. preprocessing/features.py + preprocessing/scaling.py
9. algorithm/specs.py + algorithm/arx.py
10. evaluation/metrics.py + evaluation/simulation.py
11. utils/io.py
12. pipeline.py
13. scripts/01_build_data_from_collection.py
14. scripts/02_train.py
15. compileall, build data, train model
```
