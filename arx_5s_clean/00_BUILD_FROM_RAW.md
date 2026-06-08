# Build dự án từ raw data

Tài liệu này dùng khi bắt đầu lại từ con số 0, giả sử đã có file raw CSV.

Nếu cần dựng lại source code trước, xem `00_BUILD_SRC_FROM_ZERO.md`.

## 1. Chuẩn bị raw data

Đặt file raw thật vào:

```text
data/01_raw_sessions_real/
```

CSV chỉ cần đúng 8 cột:

```csv
Timestamp,Temperature,Humidity,Light,Soil_Moisture,Drip,Mist,Fan
```

Không thêm cột ghi chú vào CSV. Nếu không đo được `Light`, ghi `0`; code sẽ tự giả lập `Light` khi sinh data train.

## 2. Build data train

Mở terminal tại thư mục dự án:

```powershell
cd C:\Users\minht\OneDrive\Desktop\ARX-Model\arx_5s_clean
```

Nếu dùng raw thật:

```powershell
python -B .\scripts\01_build_data_from_collection.py --source real --days 12
```

Nếu chưa có raw thật và muốn kiểm tra pipeline bằng data mẫu cũ:

```powershell
python -B .\scripts\01_build_data_from_collection.py --source legacy --days 12
```

Ý nghĩa:

- `real`: đọc CSV thật trong `data/01_raw_sessions_real/`.
- `legacy`: dùng data mẫu cũ trong repo để test lại pipeline.
- `--days 12`: sinh data train tương ứng 12 ngày.

## 3. Train model

Sau khi build data xong, chạy:

```powershell
python -B .\scripts\02_train.py --days 12 --grid quick
```

Nếu muốn tìm model kỹ hơn:

```powershell
python -B .\scripts\02_train.py --days 12 --grid wide
```

## 4. File cần xem sau khi chạy

Sau bước build data:

- `data/01_raw_sessions/00_raw_tong_hop.csv`: raw tổng hợp sau khi đọc từ source.
- `data/02_cleaned_sessions/00_sau_xu_ly_tong_hop.csv`: data sau xử lý timestamp và missing.
- `data/mini_greenhouse_5s_data.csv`: data cuối cùng dùng để train.

Sau bước train:

- `results/metrics.json`: kết quả model đầy đủ.
- `results/SUMMARY.md`: tóm tắt kết quả model.
- `results/arx_5s_model.json`: model đã lưu.

## 5. Thứ tự code chạy

Không cần chạy từng file step thủ công. Luồng đúng là:

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

Ý nghĩa từng phần:

- `step_00_source_data.py`: chọn source `real` hoặc `legacy`.
- `step_01_raw_sessions.py`: tạo raw session và file raw tổng hợp.
- `step_02_timestamp_cleaning.py`: xử lý `Timestamp`.
- `step_03_missing_data.py`: xử lý missing data.
- `step_04_preprocess.py`: clean raw.
- `step_05_augmentation.py`: sinh data train cuối cùng.

Sau đó script `02_train.py` sẽ train ARX model từ `data/mini_greenhouse_5s_data.csv`.

## 6. Tóm tắt ngắn nhất

Khi đã có raw thật, chỉ cần:

```powershell
cd C:\Users\minht\OneDrive\Desktop\ARX-Model\arx_5s_clean
python -B .\scripts\01_build_data_from_collection.py --source real --days 12
python -B .\scripts\02_train.py --days 12 --grid quick
```
