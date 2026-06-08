# Thuật Toán Clean Data

File code tương ứng:

```text
src/data/collection/step_01_clean_data.py
```

Mục tiêu của bước này là chuyển dữ liệu thu được trong:

```text
data/_01_data/
```

thành dữ liệu sạch trong:

```text
data/_02_clean_data/
```

Dữ liệu sạch sau bước này chưa phải là data train cuối cùng. Nó là dữ liệu đã được sửa lỗi timestamp, missing data, duplicate và chuẩn hóa trạng thái thiết bị.

## 1. Kiểm Tra Cột Bắt Buộc

Hàm:

```text
require_model_columns()
```

Dữ liệu đầu vào phải có đúng các cột cần cho bài toán:

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

Nếu thiếu một trong các cột trên thì dừng xử lý và báo lỗi. Mục đích là tránh trường hợp file thu sai định dạng nhưng vẫn được đưa vào train.

## 2. Xử Lý Timestamp

Hàm:

```text
parse_sort_timestamps()
```

Cách xử lý:

```text
Timestamp -> datetime
bước đầu round về lưới 5 giây
bỏ dòng Timestamp không đọc được
sắp xếp lại dữ liệu theo thời gian tăng dần
```

Lý do cần làm bước này:

```text
data cảm biến là chuỗi thời gian
ARX cần đúng thứ tự quá khứ -> hiện tại
missing data cũng cần nội suy theo trục thời gian
raw timestamp có thể lệch nhẹ vài giây nên cần round lại trước khi clean
```

## 3. Gộp Timestamp Bị Trùng

Hàm:

```text
collapse_duplicate_timestamps()
```

Nếu có nhiều dòng cùng một `Timestamp`, thuật toán gộp lại thành một dòng.

Với các cột sensor:

```text
Temperature
Humidity
Light
Soil_Moisture
```

cách gộp là lấy trung bình:

```text
giá trị_sau_gộp = mean(các_giá_trị_trùng_timestamp)
```

Với các cột thiết bị:

```text
Drip
Mist
Fan
```

cách gộp là lấy max:

```text
giá_trị_sau_gộp = max(các_giá_trị_trùng_timestamp)
```

Lý do dùng `max` cho thiết bị: nếu tại cùng một thời điểm có một dòng ghi thiết bị bật và một dòng ghi thiết bị tắt, chọn bật để không làm mất tác động điều khiển.

## 4. Đưa Dữ Liệu Về Lưới 5 Giây

Hàm:

```text
reindex_to_5s_grid()
```

Tần số lấy mẫu của đề tài là:

```text
5 giây / mẫu
```

Thuật toán tạo lại một trục thời gian đều:

```text
từ Timestamp đầu tiên
đến Timestamp cuối cùng
bước nhảy 5 giây
```

Sau đó dữ liệu được đưa lên trục thời gian này.

Nếu trong raw data bị mất một vài timestamp, bước này sẽ tạo lại các dòng bị thiếu. Các giá trị tại những dòng mới tạo sẽ tạm thời là missing và được xử lý ở bước sau.

Ví dụ:

```text
raw thiếu:
07:17:50
07:17:55

clean tạo lại:
07:17:50
07:17:55
```

## 5. Ép Kiểu Dữ Liệu Về Dạng Số

Hàm:

```text
coerce_numeric()
```

Các cột sau được ép về dạng số:

```text
Temperature
Humidity
Light
Soil_Moisture
Drip
Mist
Fan
```

Nếu có giá trị không chuyển được sang số, giá trị đó sẽ thành missing để xử lý tiếp.

## 6. Xử Lý Missing Và Outlier Của Sensor

Hàm:

```text
fill_sensor_missing()
```

Các cột sensor được xử lý:

```text
Temperature
Humidity
Light
Soil_Moisture
```

Trước tiên, giá trị ngoài khoảng hợp lý sẽ bị xem là lỗi và chuyển thành missing:

```text
Temperature: 15 -> 45
Humidity: 30 -> 100
Light: 0 -> 1200
Soil_Moisture: 0 -> 100
```

Sau đó xử lý missing theo thứ tự:

```text
1. Nội suy theo thời gian
2. Fill xuôi nếu còn thiếu
3. Fill ngược nếu còn thiếu
4. Clip lại trong khoảng hợp lý
```

Công thức nội suy ý tưởng:

```text
x(t) = x(t1) + (x(t2) - x(t1)) * (t - t1) / (t2 - t1)
```

Trong đó:

```text
t1 là thời điểm có dữ liệu trước khoảng missing
t2 là thời điểm có dữ liệu sau khoảng missing
t là thời điểm cần nội suy
```

Giới hạn nội suy hiện tại:

```text
limit = 12 mẫu
```

Vì mỗi mẫu cách nhau 5 giây:

```text
12 mẫu = 60 giây
```

Nghĩa là thuật toán chỉ nội suy trực tiếp các đoạn thiếu ngắn, phù hợp với lỗi mất mẫu nhỏ khi thu dữ liệu.

Sau khi xử lý xong, các cột sensor được làm tròn 1 chữ số thập phân:

```text
Temperature
Humidity
Light
Soil_Moisture
```

Mục đích là để dữ liệu nhìn giống dữ liệu cảm biến thực tế, không giữ quá nhiều chữ số do tính toán float.

## 7. Xử Lý Missing Của Thiết Bị

## 7. Làm Mềm Soil Moisture

Hàm:

```text
smooth_soil_moisture()
```

Soil sensor thực tế có thể bị nhiễu kiểu:

```text
30 -> 50 -> 40 -> 34
```

Những bước nhảy lớn trong vài giây thường không phải phản ứng thật của đất, mà là nhiễu cảm biến hoặc tiếp xúc đầu dò.

Cách xử lý:

```text
1. Tính median cục bộ với cửa sổ 5 mẫu
2. Nếu Soil_Moisture lệch khỏi median cục bộ hơn 5% thì xem là spike
3. Chuyển spike thành missing
4. Nội suy lại theo thời gian
5. Làm mềm nhẹ bằng rolling median 3 mẫu
```

Nguyên tắc là chỉ lọc các cú nhảy phi thực tế, không xóa phản ứng tăng thật do bơm.

## 8. Xử Lý Missing Của Thiết Bị

Hàm:

```text
fill_actuator_missing()
```

Các cột thiết bị:

```text
Drip
Mist
Fan
```

Các thiết bị chỉ nhận trạng thái:

```text
0 = tắt
1 = bật
```

Cách xử lý:

```text
1. Giá trị khác 0/1 được xem là lỗi và chuyển thành missing
2. Fill xuôi tối đa 12 mẫu
3. Fill ngược tối đa 2 mẫu
4. Nếu vẫn missing thì gán 0
5. Ép lần cuối về 0/1
```

Lý do dùng fill xuôi: trạng thái thiết bị thường giữ nguyên trong một khoảng thời gian ngắn. Nếu một dòng bị mất trạng thái, lấy trạng thái trước đó là hợp lý hơn nội suy ra số lẻ.

Sau bước này, thiết bị chỉ còn:

```text
0.0 hoặc 1.0
```

Ví dụ:

```text
Fan trước clean: NaN
Fan sau clean: 0
```

## 9. Clean Một File Data

Hàm:

```text
clean_data_file()
```

Luồng xử lý đầy đủ cho một file:

```text
đọc CSV
kiểm tra cột
parse và sort Timestamp
gộp duplicate Timestamp
đưa về lưới 5 giây
ép kiểu số
xử lý missing sensor
xử lý spike Soil_Moisture
xử lý missing thiết bị
trả về đúng 8 cột model cần
```

## 10. Clean Toàn Bộ Data

Hàm:

```text
clean_all_data_files()
```

Hàm này chạy `clean_data_file()` cho từng file trong `_01_data`, sau đó ghi kết quả vào `_02_clean_data`.

Mỗi file sẽ có một file sau xử lý tương ứng:

```text
morning_anchor_raw.csv
-> morning_anchor_sau_xu_ly.csv
```

Ngoài ra còn có file tổng hợp:

```text
00_sau_xu_ly_tong_hop.csv
```

File tổng hợp này ghép toàn bộ data sạch theo thứ tự thời gian.

## 11. Tóm Tắt Luồng Clean Data

```text
data/_01_data/*.csv
  -> kiểm tra cột
  -> parse Timestamp
  -> sort theo thời gian
  -> gộp timestamp trùng
  -> reindex về 5 giây
  -> xử lý missing sensor
  -> lọc spike Soil_Moisture
  -> xử lý missing thiết bị
  -> data/_02_clean_data/*.csv
```

Kết quả sau clean:

```text
không còn missing
không còn timestamp trùng
mỗi phiên 2 tiếng có 1440 dòng
dữ liệu đúng lưới 5 giây
thiết bị chỉ còn 0 hoặc 1
```
