# Thuật Toán Build Data Full

## 1. Mục đích

Build data dùng để tạo lại bộ dữ liệu 5 giây cho model ARX từ dữ liệu hiện tại.

Flow hiện tại dùng dữ liệu 5 giây đang có làm nguồn và build lại theo các bước:

```text
mini_greenhouse_5s_data.csv hiện tại
  -> tách thành 12 folder theo ngày
  -> trong mỗi ngày tách các block phủ đủ 00:00 -> 24:00
  -> raw có missing/duplicate không đều và timestamp lệch nhẹ theo từng file
  -> clean từng file raw
  -> ghép lại thành mini_greenhouse_5s_data.csv
```

## 2. File chạy

Chạy từ thư mục `ARX_DO_BY_SELF`:

```powershell
python scripts/01_build_data.py --raw-dir data
```

File điều phối chính:

```text
src/data/collection/step_02_pipeline.py
```

Các file xử lý chính:

```text
src/data/collection/step_00_data_io.py
src/data/collection/step_01_clean_data.py
```

## 3. Tách dữ liệu theo ngày và theo buổi

Pipeline đọc:

```text
data/mini_greenhouse_5s_data.csv
```

Sau đó tách theo ngày thật trong cột `Timestamp`, rồi tiếp tục tách mỗi ngày thành các block liên tiếp:

```text
midnight:        00:00 -> 07:00
morning:         07:00 -> 09:00
late_morning:    09:00 -> 11:30
noon:            11:30 -> 13:30
early_afternoon: 13:30 -> 15:00
afternoon:       15:00 -> 17:00
evening:         17:00 -> 20:00
night:           20:00 -> 22:00
late_night:      22:00 -> 24:00
```

Kết quả ghi vào:

```text
data/_01_data/
```

Cấu trúc hiện tại:

```text
data/_01_data/
  2026-04-08/
    00_midnight_raw.csv
    01_morning_raw.csv
    02_late_morning_raw.csv
    03_noon_raw.csv
    04_early_afternoon_raw.csv
    05_afternoon_raw.csv
    06_evening_raw.csv
    07_night_raw.csv
    08_late_night_raw.csv
  ...
  2026-04-19/
    00_midnight_raw.csv
    01_morning_raw.csv
    02_late_morning_raw.csv
    03_noon_raw.csv
    04_early_afternoon_raw.csv
    05_afternoon_raw.csv
    06_evening_raw.csv
    07_night_raw.csv
    08_late_night_raw.csv
```

Các file block ghép lại phủ đủ một ngày 24 giờ, sampling 5 giây.

## 4. Raw tổng hợp

Sau khi có 12 folder ngày, pipeline ghép toàn bộ raw lại thành:

```text
data/_01_data/00_raw_tong_hop.csv
```

File này dùng để kiểm tra dữ liệu trước xử lý.

## 5. Clean dữ liệu

Pipeline chạy clean cho từng file raw.

Hàm chính:

```text
clean_all_data_files()
```

File code:

```text
src/data/collection/step_01_clean_data.py
```

Các bước xử lý:

```text
parse Timestamp
ước lượng offset timestamp của từng file
trừ offset để đưa timestamp về gần lưới 5 giây chuẩn
round Timestamp về lưới 5 giây
sort theo thời gian
gộp Timestamp trùng
tạo lại lưới thời gian 5 giây
xử lý missing sensor
xử lý missing actuator
clip sensor về khoảng hợp lý
```

Kết quả ghi vào:

```text
data/_02_clean_data/
```

Cấu trúc:

```text
data/_02_clean_data/
  2026-04-08/
    00_midnight_sau_xu_ly.csv
    01_morning_sau_xu_ly.csv
    02_late_morning_sau_xu_ly.csv
    03_noon_sau_xu_ly.csv
    04_early_afternoon_sau_xu_ly.csv
    05_afternoon_sau_xu_ly.csv
    06_evening_sau_xu_ly.csv
    07_night_sau_xu_ly.csv
    08_late_night_sau_xu_ly.csv
  ...
  2026-04-19/
    00_midnight_sau_xu_ly.csv
    01_morning_sau_xu_ly.csv
    02_late_morning_sau_xu_ly.csv
    03_noon_sau_xu_ly.csv
    04_early_afternoon_sau_xu_ly.csv
    05_afternoon_sau_xu_ly.csv
    06_evening_sau_xu_ly.csv
    07_night_sau_xu_ly.csv
    08_late_night_sau_xu_ly.csv
  00_sau_xu_ly_tong_hop.csv
```

## 6. Data cuối

Sau khi clean, pipeline ghép data sạch và ghi lại:

```text
data/mini_greenhouse_5s_data.csv
```

File cuối có 8 cột:

```text
Timestamp
Temperature
Humidity
Light
Soil_Moisture
Drip
Mist
Fan
```

Feature phụ như `Light_log`, `Air_Dryness`, `Hour_sin`, `Day_cos` chưa nằm trong file data này. Các feature đó được tạo ở bước train model trong:

```text
src/preprocessing/features.py
```

## 7. Kết quả hiện tại

Sau khi build:

```text
Số dòng: 207360
Số ngày: 12
Ngày bắt đầu: 2026-04-08 00:00:00
Ngày kết thúc: 2026-04-19 23:59:55
```

Vì sampling là 5 giây:

```text
1 ngày = 24 * 3600 / 5 = 17280 dòng
12 ngày = 17280 * 12 = 207360 dòng
```

Kiểm tra lỗi trước/sau clean:

```text
Raw tổng hợp:
- missing: 315 ô
- duplicate timestamp: 49 dòng

Clean tổng hợp:
- missing: 0
- duplicate timestamp: 0
```

## 8. Cách trình bày ngắn trong báo cáo

```text
Dữ liệu ban đầu được tách thành 12 ngày, mỗi ngày gồm các block liên tiếp phủ đủ 00:00 đến 24:00. Raw data có missing/duplicate không đều theo từng file và timestamp lệch nhẹ theo từng file. Sau đó pipeline clean từng block, đưa timestamp về lưới 5 giây và ghép lại thành file mini_greenhouse_5s_data.csv để train mô hình ARX.
```
