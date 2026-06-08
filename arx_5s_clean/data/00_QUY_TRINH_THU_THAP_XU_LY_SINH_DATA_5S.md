# Quy trình thu và sinh dữ liệu 5s

Tài liệu này là bản gọn để em làm đồ án: em chỉ cần gửi 1 phiên raw thật, còn code sẽ clean raw đó rồi sinh ra data train đầy đủ.

Nếu muốn build toàn bộ dự án từ raw data đến kết quả model, xem `../00_BUILD_FROM_RAW.md`.

## 1. Cột cần có

CSV raw và CSV train cuối cùng chỉ cần 8 cột:

```csv
Timestamp,Temperature,Humidity,Light,Soil_Moisture,Drip,Mist,Fan
```

Ý nghĩa:

| Cột | Ý nghĩa |
|---|---|
| `Timestamp` | Thời điểm lấy mẫu |
| `Temperature` | Nhiệt độ không khí |
| `Humidity` | Độ ẩm không khí |
| `Light` | Ánh sáng |
| `Soil_Moisture` | Độ ẩm đất |
| `Drip` | Bơm nhỏ giọt, `0/1` |
| `Mist` | Phun sương, `0/1` |
| `Fan` | Quạt, `0/1` |

Nếu em không đo được `Light`, cứ ghi `0`. Phần sinh data cuối sẽ tự giả lập light theo thời gian trong ngày.

## 2. Em cần gửi gì

Chỉ cần 1 phiên raw thật, tốt nhất dài khoảng 2 giờ.

Gợi ý tên file:

```text
01_real_session.csv
```

Đặt file vào:

```text
data/01_raw_sessions_real/
```

Nếu em có thêm nhiều phiên thì càng tốt, nhưng không bắt buộc.

## 3. Kịch bản thu

Làm theo file này:

```text
data/00_KICH_BAN_THU_DATA_THUC_TE_5S.md
```

Trong đó:

- giữ sampling đúng 5 giây;
- bật/tắt `Drip`, `Mist`, `Fan` theo mốc;
- `Light` có thể để `0` nếu không đo được;
- sensor thật đo bao nhiêu thì giữ nguyên, không tự chỉnh.

## 4. Xử lý raw

Sau khi thu xong, chạy:

```powershell
python -B .\scripts\01_build_data_from_collection.py --source real --days 12
```

Script sẽ tự làm:

1. Đọc raw gốc.
2. Sắp xếp `Timestamp`, gom timestamp trùng.
3. Reindex về lưới 5 giây.
4. Fill missing ngắn.
5. Xuất file tổng hợp raw và file tổng hợp sau xử lý.
6. Sinh đủ data train 12 ngày hoặc nhiều hơn.

## 5. File đầu ra

Sau khi chạy xong, em chỉ cần chú ý 4 file này:

- `data/01_raw_sessions/00_raw_tong_hop.csv`
- `data/02_cleaned_sessions/00_sau_xu_ly_tong_hop.csv`
- `data/mini_greenhouse_5s_data.csv`
- `results/metrics.json`

## 6. Quy tắc xử lý missing

- Thiếu ngắn vài mẫu: nội suy.
- Thiếu timestamp ngắn: script tự chèn vào lưới 5 giây.
- Mất đoạn dài: nên coi như một phiên khác.
- `Drip`, `Mist`, `Fan` chỉ nhận `0` hoặc `1`.

## 7. Chế độ chạy

Hiện có 2 nguồn:

- `legacy`: dùng data mẫu cũ trong repo để kiểm tra pipeline.
- `real`: dùng raw thật em gửi.

Ví dụ:

```powershell
python -B .\scripts\01_build_data_from_collection.py --source legacy --days 12
python -B .\scripts\01_build_data_from_collection.py --source real --days 12
```

## 8. Nhánh code chính

- `step_00_source_data.py`: đọc source data.
- `step_01_raw_sessions.py`: tạo raw session.
- `step_02_timestamp_cleaning.py`: xử lý timestamp.
- `step_03_missing_data.py`: xử lý missing.
- `step_04_preprocess.py`: clean raw.
- `step_05_augmentation.py`: sinh data train.
- `step_07_pipeline.py`: nối toàn bộ luồng.

## 9. Ghi nhớ

Em chỉ cần tập trung vào 1 phiên raw thật cho đúng. Phần sinh các ngày còn lại và giả lập `Light` để train sẽ do code lo. 
