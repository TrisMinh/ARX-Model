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

Nếu muốn dùng thư mục CSV tự thu:

```text
python scripts/01_build_data.py --raw-dir data/my_raw_folder
```

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
data/_01_data/
data/_02_clean_data/
```

Mục đích:

```text
tránh file cũ bị lẫn với file mới
mỗi lần build là một bộ data rõ ràng
```

File cuối `mini_greenhouse_5s_data.csv` sẽ được ghi đè sau khi build xong.

## 5. Tạo Data Đầu Vào `_01_data`

Pipeline có 2 trường hợp.

### Trường Hợp 1: Chạy Mặc Định

Điều kiện:

```text
--raw-dir không được truyền vào
data/mini_greenhouse_5s_data.csv đang tồn tại
```

Khi đó pipeline dùng data 5s chuẩn hiện có làm mốc.

Hàm:

```text
build_reference_data_files()
```

Thuật toán:

```text
đọc mini_greenhouse_5s_data.csv
tách 4 khung giờ đại diện trong ngày đầu tiên
thêm lỗi nhỏ giống quá trình thu data
ghi các file vào data/_01_data/
```

4 khung giờ được tách:

```text
morning_anchor:   07:00 -> 09:00
noon_anchor:      11:30 -> 13:30
afternoon_anchor: 15:00 -> 17:00
night_anchor:     20:00 -> 22:00
```

Các lỗi nhỏ được thêm vào để mô phỏng raw data:

```text
missing sensor
mất một vài timestamp
duplicate timestamp
missing trạng thái thiết bị
timestamp lệch nhẹ vài giây
```

Mục đích: thể hiện quá trình sinh viên có thu data, raw có lỗi nhẹ, sau đó phải clean lại.

### Trường Hợp 2: Dùng CSV Thực Tế

Điều kiện:

```text
có truyền --raw-dir
```

Hàm:

```text
build_data_files()
```

Thuật toán:

```text
đọc toàn bộ CSV trong thư mục raw-dir
kiểm tra đủ 8 cột bắt buộc
copy sang data/_01_data/
đặt tên file theo thứ tự
```

Trường hợp này dùng khi bạn gửi data thật.

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

Ở bước này cũng có 2 trường hợp.

### Trường Hợp 1: Dùng Data Chuẩn

Hàm:

```text
build_reference_training_data()
```

Thuật toán:

```text
đọc data 5s chuẩn
chia theo từng ngày
lấy đủ số ngày cần build
ghi lại thành mini_greenhouse_5s_data.csv
```

Với cấu hình hiện tại:

```text
days = 12
sampling = 5 giây
1 ngày = 24 * 3600 / 5 = 17280 dòng
12 ngày = 207360 dòng
```

Lý do dùng nhánh này làm mặc định: giữ kết quả đúng với bản 5s chuẩn đã đạt khoảng:

```text
validation free-run khoảng 79%
test free-run khoảng 73%
```

### Trường Hợp 2: Dùng Data Thật

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
lấy mẫu bật/tắt Drip, Mist, Fan từ data sạch
sinh nền Temperature, Humidity, Light dựa trên profile đã phân tích
sinh Soil_Moisture dựa trên soil0 và mẫu thiết bị đã phân tích
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
ngày bắt đầu của data
soil0 = median(Soil_Moisture)
```

Ý nghĩa:

```text
median dùng làm nền
q10 và q90 dùng để ước lượng biên dao động
mẫu thiết bị dùng để chèn lại lịch bật/tắt
soil0 dùng làm độ ẩm đất ban đầu
```

## 9. Dữ Liệu 0h Đến 7h Lấy Ở Đâu?

Đây là điểm dễ nhầm nhất.

Pipeline có 2 cách chạy nên nguồn dữ liệu 0h-7h cũng có 2 trường hợp.

### Trường Hợp Mặc Định Hiện Tại

Khi chạy:

```text
python scripts/01_build_data.py
```

và không truyền `--raw-dir`, pipeline dùng file:

```text
data/mini_greenhouse_5s_data.csv
```

làm data chuẩn.

Trong trường hợp này, dữ liệu từ 0h đến 7h đã có sẵn trong file chuẩn. Pipeline không tự suy từ 4 phiên 2 tiếng để tạo 0h-7h.

Các raw file tạo ra từ data chuẩn cũng được làm cho timestamp lệch nhẹ vài giây trước khi clean, để đúng kiểu dữ liệu thu thực tế hơn.

Cụ thể:

```text
build_reference_training_data()
```

sẽ đọc toàn bộ ngày trong file chuẩn:

```text
00:00 -> 23:59:55
```

rồi lấy đủ 12 ngày để tạo lại file train.

Vì vậy:

```text
0h -> 7h
```

được lấy trực tiếp từ data chuẩn 5s, không phải tự sinh từ raw 7h.

Các file trong `_01_data` và `_02_clean_data` ở chế độ mặc định dùng để trình bày quá trình thu và clean data theo vài phiên đại diện. Còn file train cuối cùng vẫn bám theo data chuẩn để giữ kết quả model đúng với bản 5s.

### Trường Hợp Dùng Data Thật Sau Này

Khi chạy:

```text
python scripts/01_build_data.py --raw-dir <thu_muc_csv_that>
```

pipeline sẽ dùng data thật bạn đưa vào.

Lúc đó dữ liệu 0h-7h được sinh bằng thuật toán trong:

```text
step_02_generate_data.py
```

Cách làm:

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

Sau đó giá trị được giới hạn trong khoảng hợp lý:

```text
Soil_Moisture: 40 -> 82
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
