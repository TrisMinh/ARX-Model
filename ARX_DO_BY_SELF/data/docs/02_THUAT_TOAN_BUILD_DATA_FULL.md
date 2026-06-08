# Thuật Toán Build Data Full

File chạy từ terminal:

```text
scripts/01_build_data.py
```

File điều phối chính:

```text
src/data/collection/step_03_pipeline.py
```

Các file xử lý bên trong:

```text
src/data/collection/step_00_data_io.py
src/data/collection/step_01_clean_data.py
src/data/collection/step_02_generate_data.py
```

Mục tiêu của build data full là tạo ra file cuối cùng:

```text
data/mini_greenhouse_5s_data.csv
```

File này là data 5 giây dùng để train model ARX.

## 1. Cấu Trúc Data

Pipeline build data tạo và dùng các thư mục:

```text
data/_01_data/
data/_02_clean_data/
data/mini_greenhouse_5s_data.csv
```

Ý nghĩa:

```text
_01_data
data đầu vào theo từng phiên, còn lỗi nhỏ như missing hoặc duplicate

_02_clean_data
data sau khi clean timestamp, missing, duplicate

mini_greenhouse_5s_data.csv
data cuối cùng đủ ngày để train model
```

## 2. Lệnh Chạy

Lệnh mặc định:

```text
python scripts/01_build_data.py
```

Tham số mặc định:

```text
days = 12
seed = 505031
raw-dir = None
```

Raw data chính nằm trong:

```text
data/_01_data/
```

Raw được chia theo từng ngày:

```text
data/_01_data/
  2026-04-08/
  2026-04-09/
  2026-04-10/
  2026-04-11/
  2026-04-12/
```

Code sẽ đọc toàn bộ CSV trong các folder con của `_01_data`.

## 3. Luồng Tổng

Luồng build data đầy đủ:

```text
01_build_data.py
  -> data.collection.run()
  -> tạo _01_data
  -> clean sang _02_clean_data
  -> sinh hoặc lấy data 12 ngày
  -> ghi mini_greenhouse_5s_data.csv
```

Trong code, luồng này nằm ở:

```text
step_03_pipeline.py -> run()
```

## 4. Xóa Output Cũ

Hàm:

```text
clear_csv_outputs()
```

Trước khi build lại, pipeline xóa các file CSV cũ trong:

```text
data/_02_clean_data/
```

Mục đích:

```text
tránh file clean cũ bị lẫn với file mới
mỗi lần build là một bộ data rõ ràng
```

`data/_01_data/` là raw data gốc nên không bị xóa khi chạy build mặc định.

File cuối `mini_greenhouse_5s_data.csv` sẽ được ghi đè sau khi build xong.

## 5. Tạo Data Đầu Vào `_01_data`

Pipeline dùng `_01_data` làm raw data chính.

### Chạy Mặc Định

Điều kiện:

```text
--raw-dir không được truyền vào
data/_01_data/ có CSV raw
```

Khi đó pipeline đọc raw data trong `_01_data`.

Hàm:

```text
input_csv_paths()
```

Thuật toán:

```text
đọc toàn bộ CSV trong data/_01_data/
giữ nguyên cấu trúc folder ngày
ghép thành 00_raw_tong_hop.csv
clean sang data/_02_clean_data/
sinh data train cuối cùng
```

Raw hiện được tổ chức theo ngày:

```text
2026-04-08/
2026-04-09/
2026-04-10/
2026-04-11/
2026-04-12/
```

Mỗi ngày có các phiên:

```text
01_morning_raw.csv
02_noon_raw.csv
03_afternoon_raw.csv
04_night_raw.csv
```

Mục đích: thể hiện quá trình sinh viên thu raw data theo nhiều ngày, sau đó clean và sinh data train.

### Dùng Thư Mục Raw Khác

Điều kiện:

```text
có truyền --raw-dir
```

Hàm:

```text
load_source_data()
build_collection_session_files()
```

Thuật toán:

```text
đọc toàn bộ CSV trong thư mục raw-dir
kiểm tra đủ 8 cột bắt buộc
ghép lại để phân tích profile thực tế
sinh 4 phiên thu đại diện theo kịch bản lý thuyết
thêm lỗi raw nhỏ như lệch timestamp, missing, duplicate
ghi vào data/_01_data/
```

4 file raw được tạo:

```text
01_morning_raw.csv
02_noon_raw.csv
03_afternoon_raw.csv
04_night_raw.csv
```

Trường hợp này chỉ dùng nếu muốn nhập một thư mục raw khác để tạo lại `_01_data`.

## 6. Ghi File Tổng Hợp Raw

Sau khi có các file trong `_01_data`, pipeline ghép lại thành:

```text
data/_01_data/00_raw_tong_hop.csv
```

File này dùng để xem toàn bộ data trước xử lý.

Nó chưa phải data sạch.

## 7. Clean Data

Hàm:

```text
clean_all_data_files()
```

File code:

```text
src/data/collection/step_01_clean_data.py
```

Thuật toán clean chi tiết nằm trong docs:

```text
data/docs/01_THUAT_TOAN_CLEAN_DATA.md
```

Tóm tắt:

```text
parse timestamp
round timestamp về lưới 5 giây
sort thời gian
gộp duplicate timestamp
đưa về lưới 5 giây
xử lý missing sensor
xử lý missing thiết bị
ghi ra _02_clean_data
```

Output:

```text
data/_02_clean_data/*.csv
data/_02_clean_data/00_sau_xu_ly_tong_hop.csv
```

## 8. Tạo Data Train Cuối Cùng

Sau khi clean, pipeline tạo:

```text
data/mini_greenhouse_5s_data.csv
```

Với cấu hình hiện tại:

```text
days = 12
sampling = 5 giây
1 ngày = 24 * 3600 / 5 = 17280 dòng
12 ngày = 207360 dòng
```

Hàm:

```text
build_training_data()
```

Trước khi sinh data, pipeline phân tích data sạch bằng:

```text
analyze_collected_data()
```

Hàm nội bộ:

```text
_build_training_data()
```

Thuật toán:

```text
lấy data sạch sau clean
phân tích median, q10, q90 của Temperature, Humidity, Light, Soil_Moisture
ước lượng tác động thật của Mist, Fan, Drip từ các đoạn thiết bị bật
lấy mẫu bật/tắt Drip, Mist, Fan từ data sạch
sinh nền Temperature, Humidity, Light dựa trên profile đã phân tích
sinh Humidity theo phản ứng Mist/Fan đã đo
sinh Soil_Moisture theo phản ứng Drip/Mist/Fan đã đo
lặp lại thành nhiều ngày
ghi mini_greenhouse_5s_data.csv
```

Nhánh này dùng khi bạn đưa data thật vào. Khi đó pipeline không sinh ngẫu nhiên từ số cứng, mà dùng dữ liệu thật làm gốc để sinh đủ số ngày train.

Các thông tin rút ra từ data thu thập:

```text
Temperature median, q10, q90
Humidity median, q10, q90
Light median, q10, q90
Soil_Moisture median, q10, q90
mẫu bật/tắt Drip, Mist, Fan
Mist làm Humidity tăng bao nhiêu
Fan làm Humidity giảm bao nhiêu
Drip làm Soil_Moisture tăng bao nhiêu
ngày bắt đầu của data
soil0 = median(Soil_Moisture)
```

Ý nghĩa:

```text
median dùng làm nền
q10 và q90 dùng để ước lượng biên dao động
mẫu thiết bị dùng để chèn lại lịch bật/tắt
phản ứng thiết bị dùng để data sinh ra giống data thực tế hơn
soil0 dùng làm độ ẩm đất ban đầu
```

## 9. Dữ Liệu 0h Đến 7h Lấy Ở Đâu?

Đây là điểm dễ nhầm nhất.

Pipeline hiện dùng `_01_data` làm raw data chính.

### Trường Hợp Mặc Định Hiện Tại

Khi chạy:

```text
python scripts/01_build_data.py
```

và không truyền `--raw-dir`, pipeline dùng các phiên raw trong:

```text
data/_01_data/
```

để clean và sinh data train.

Trong trường hợp này, `_01_data` đã có raw theo nhiều ngày và nhiều phiên:

```text
2026-04-08/01_morning_raw.csv
2026-04-08/02_noon_raw.csv
...
2026-04-12/04_night_raw.csv
```

Với các khoảng không có phiên thu, ví dụ `0h -> 7h`, pipeline sinh bằng thuật toán trong `step_02_generate_data.py`.

Logic:

```text
1. Tạo trục thời gian nguyên ngày: 00:00 -> 23:59:55, bước 5 giây
2. Phân tích data sạch để lấy profile môi trường và đất
3. Sinh Temperature, Humidity, Light cho toàn bộ ngày theo profile đó
4. Đặt Drip, Mist, Fan = 0 ở các khoảng không có phiên thu
5. Chèn mẫu bật/tắt thiết bị từ data thật vào các mốc đại diện
6. Sinh Soil_Moisture liên tục từ 00:00 đến 23:59:55
```

Với 0h-7h:

```text
Temperature/Humidity/Light
được sinh theo chu kỳ ngày đêm nhưng bám median/q10/q90 của data bạn thu

Drip/Mist/Fan
mặc định bằng 0 vì chưa có phiên bật thiết bị

Soil_Moisture
được tính liên tục từ soil0 rút ra từ data sạch, chịu tác động của môi trường và không có tưới/phun/quạt
```

Giá trị ban đầu:

```text
soil0 = median(Soil_Moisture trong data sạch)
```

Nghĩa là model giả lập bắt đầu từ độ ẩm đất trung bình đo được, sau đó để đất biến thiên theo môi trường.

Ví dụ logic 0h-7h:

```text
00:00 -> 06:00
Light thấp, Temperature thấp hơn, Humidity cao hơn
Drip = 0, Mist = 0, Fan = 0
Soil_Moisture giảm rất chậm do bay hơi nhỏ

06:00 -> 07:00
Light bắt đầu tăng
Temperature tăng dần
Humidity giảm dần
Drip = 0, Mist = 0, Fan = 0
Soil_Moisture giảm nhanh hơn một chút vì bay hơi tăng
```

Đến 7h:

```text
pipeline chèn mẫu thiết bị từ phiên thu vào khung 07:00 -> 09:00
```

## 10. Công Thức Sinh Môi Trường

File:

```text
step_02_generate_data.py
```

Hàm:

```text
base_environment()
```

Hàm này nhận `profile` từ:

```text
analyze_collected_data()
```

Ý tưởng:

```text
Light tăng vào ban ngày, thấp vào ban đêm
Temperature tăng theo ánh sáng ban ngày
Humidity giảm khi Temperature và Light tăng
```

Nhưng các mức nền không lấy tùy ý. Chúng được lấy từ data thu thập:

```text
temp_base = median(Temperature)
temp_gain = q90(Temperature) - q10(Temperature)

humi_base = median(Humidity)
humi_drop = q90(Humidity) - q10(Humidity)

light_night = q10(Light)
light_gain = q90(Light) - q10(Light)
```

Nghĩa là nếu bạn đo ngoài thực tế độ ẩm không khí khoảng 55, profile sinh ra cũng sẽ bám vùng đó, không tự nhảy về một mức khác.

Sau khi có Humidity nền, code tiếp tục chỉnh Humidity theo Mist/Fan:

```text
humidity_response()
```

Ý tưởng:

```text
Mist bật  -> Humidity tăng dần về vùng ẩm cao
Fan bật   -> Humidity giảm dần về nền môi trường
Không bật -> Humidity quay chậm về nền ngày/đêm
```

Mức tăng/giảm được đo từ data sạch:

```text
mist_humidity_gain = Humidity peak sau khi Mist bật - Humidity trước khi bật
fan_humidity_drop  = Humidity trước khi Fan bật - Humidity thấp nhất sau khi Fan bật
```

Các biến có nhiễu nhỏ để data không bị quá đều:

```text
temp_bias
humi_bias
light_scale
random noise
```

Với ánh sáng ban ngày, code dùng biến trung gian:

```text
daylight = max(0, sin((hour - 6) / 12 * pi))
```

Ý nghĩa:

```text
trước 6h: daylight gần 0
6h -> trưa: daylight tăng
trưa -> chiều: daylight giảm
tối: daylight về 0
```

Từ đó:

```text
Light = nền ban đêm + cường độ theo daylight + nhiễu nhỏ
Temperature = nền + thành phần daylight + nhiễu nhỏ
Humidity = giảm khi daylight và Temperature tăng
```

## 11. Công Thức Sinh Soil Moisture

Hàm:

```text
soil_response()
```

Ý tưởng cập nhật từng bước:

```text
Soil(t) = Soil(t-1)
          + nước từ Drip/Mist
          - bay hơi do Temperature/Humidity/Light/Fan
          - thoát nước khi đất quá ẩm
          + cân bằng chậm về vùng nền
          + nhiễu nhỏ
```

Giá trị khởi tạo không chọn bừa:

```text
Soil(0) = median(Soil_Moisture trong data sạch)
```

Mẫu thiết bị cũng lấy từ data sạch:

```text
Drip template = cột Drip trong data sạch
Mist template = cột Mist trong data sạch
Fan template = cột Fan trong data sạch
```

Trước khi sinh Soil, code đo tác động tưới từ data sạch:

```text
drip_soil_gain = Soil_Moisture cao nhất sau khi Drip bật - Soil_Moisture trước khi bật
```

Nếu có phiên đất quá khô bất thường, code ưu tiên các phiên Soil_Moisture ở vùng thực tế hơn để tránh làm phản ứng bơm bị phóng đại.

Các thành phần chính:

```text
evap
bay hơi, tăng khi nhiệt độ cao, không khí khô, ánh sáng cao hoặc quạt bật

water
nước vào đất do Drip và Mist, có độ trễ vài mẫu

drainage
thoát nước khi Soil_Moisture cao hơn vùng ẩm

slow_balance
kéo Soil_Moisture nhẹ về vùng nền
```

Sau đó giá trị được giới hạn trong khoảng vật lý của sensor:

```text
Soil_Moisture: 0 -> 100
```

Sensor đo cũng có làm mượt:

```text
Soil_Measured(t) = 0.48 * Soil_Measured(t-1) + 0.52 * Soil_Raw(t)
```

## 12. Các Mốc Chèn Thiết Bị Khi Sinh Data Thật

Trong nhánh dùng data thật, code lấy mẫu bật/tắt:

```text
Drip
Mist
Fan
```

từ data sạch, rồi chèn vào các mốc:

```text
07:00
11:30
15:00
20:00
```

Code tương ứng:

```text
session_offsets = (7h, 11h30, 15h, 20h)
```

Ngoài các khoảng này:

```text
Drip = 0
Mist = 0
Fan = 0
```

Vì vậy nếu hỏi riêng:

```text
0h -> 7h lấy điều khiển ở đâu?
```

thì câu trả lời là:

```text
không lấy từ phiên thu nào
thiết bị mặc định tắt
môi trường và Soil_Moisture được sinh theo công thức
```

## 13. Output Cuối Cùng

Sau khi build xong, file dùng để train là:

```text
data/mini_greenhouse_5s_data.csv
```

File này có đúng 8 cột:

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

Các cột sensor được làm tròn 1 chữ số thập phân:

```text
Temperature
Humidity
Light
Soil_Moisture
```

Các cột thiết bị giữ dạng 0/1:

```text
Drip
Mist
Fan
```

Các feature như `Light_log`, `Air_Dryness`, `Hour_sin` chưa nằm trong file data này. Chúng được tạo sau, ở bước train model:

```text
src/preprocessing/features.py
```

## 14. Tóm Tắt Ngắn

```text
scripts/01_build_data.py
  -> step_03_pipeline.run()
  -> step_00_data_io tạo _01_data
  -> step_01_clean_data tạo _02_clean_data
  -> step_02_generate_data tạo mini_greenhouse_5s_data.csv
```

Luồng file:

```text
data/_01_data/
  -> data/_02_clean_data/
  -> data/mini_greenhouse_5s_data.csv
```
