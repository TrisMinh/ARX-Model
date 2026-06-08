# Đặt CSV thật vào thư mục này

Khi em có raw thật, chỉ cần đặt 1 file `.csv` vào thư mục này rồi chạy:

```powershell
python -B .\scripts\01_build_data_from_collection.py --source real --days 12
```

Mỗi CSV chỉ cần đúng 8 cột:

```csv
Timestamp,Temperature,Humidity,Light,Soil_Moisture,Drip,Mist,Fan
```

Ví dụ tên file:

```text
01_real_session.csv
02_real_session.csv
drip_20s_test_2026_06_08.csv
mist_20s_test_2026_06_08.csv
fan_120s_test_2026_06_08.csv
```

Không thêm cột ghi chú vào CSV. Các ghi chú như “độ ẩm ngoài trời 55%” hoặc “Mist bật 20s làm Humidity tăng lên 80%” ghi ở file note riêng.
