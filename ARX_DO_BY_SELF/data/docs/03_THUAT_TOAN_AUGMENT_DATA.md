# Thuật Toán Augment Data

## 1. Mục đích

Augment data dùng để tạo thêm dữ liệu train từ các ngày đã có, thay vì tự sinh lại toàn bộ hệ thống từ công thức mô phỏng.

Ý tưởng chính:

```text
lấy data ngày cũ
-> đổi Timestamp sang ngày mới
-> thêm sai lệch nhỏ cho cảm biến
-> dịch nhẹ thời điểm actuator
-> ghép thành data train nhiều ngày
```

Cách này giúp dữ liệu mới vẫn bám theo mẫu dữ liệu đã thu, dễ giải thích hơn so với việc tự sinh lại Temperature, Humidity, Light và Soil_Moisture từ đầu.

## 2. Input

Input của thuật toán là data đã xử lý sạch:

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

Data nên là dữ liệu liên tục theo ngày. Nếu chỉ có các phiên rời rạc như morning/noon/afternoon/night thì cần ghép hoặc nội suy thành ngày đầy đủ trước khi augment.

## 3. Tách data theo ngày

Đầu tiên, sắp xếp data theo `Timestamp`, sau đó tách thành từng ngày:

```text
Ngày 1: 2026-04-08
Ngày 2: 2026-04-09
Ngày 3: 2026-04-10
...
```

Mỗi ngày được xem là một template.

## 4. Tạo ngày mới từ ngày cũ

Với mỗi ngày cần tạo thêm:

```text
1. Chọn một ngày cũ làm template.
2. Giữ nguyên hình dạng biến thiên trong ngày.
3. Đổi Timestamp sang ngày mới.
```

Ví dụ:

```text
Template: 2026-04-08 00:00:00 -> 23:59:55
Ngày mới: 2026-04-13 00:00:00 -> 23:59:55
```

Các giá trị cảm biến ban đầu được copy từ ngày cũ.

## 5. Thêm sai lệch nhỏ cho cảm biến

Để ngày mới không giống y hệt ngày cũ, thêm nhiễu nhỏ và bias nhỏ:

```text
Temperature_new = Temperature_old + temp_bias + noise
Humidity_new    = Humidity_old + humi_bias + noise
Light_new       = Light_old * light_scale + noise
Soil_new        = Soil_old + soil_bias + trend nhỏ + noise
```

Giá trị đề xuất:

| Biến | Cách augment |
|---|---|
| `Temperature` | cộng bias khoảng `-0.4` đến `+0.4` độ C |
| `Humidity` | cộng bias khoảng `-2` đến `+2` % |
| `Light` | nhân hệ số khoảng `0.9` đến `1.1` |
| `Soil_Moisture` | cộng bias nhỏ khoảng `-0.3` đến `+0.3` % |

Nhiễu chỉ nên nhỏ để không phá mẫu dữ liệu gốc.

## 6. Augment actuator

Các actuator là biến bật/tắt nên không cộng nhiễu số thực.

Với:

```text
Drip
Mist
Fan
```

Cách làm:

```text
1. Giữ giá trị 0/1.
2. Dịch nhẹ thời điểm bật/tắt khoảng vài bước mẫu.
3. Không biến actuator thành giá trị lẻ như 0.2 hoặc 0.7.
```

Ví dụ với sampling `5s`, có thể shift:

```text
-6 bước đến +6 bước
= -30s đến +30s
```

Việc này mô phỏng chuyện thiết bị bật sớm hoặc trễ nhẹ giữa các ngày.

## 7. Clip lại giá trị

Sau khi augment, cần giới hạn giá trị trong khoảng hợp lý:

```text
Temperature: 15 -> 45
Humidity: 30 -> 100
Light: 0 -> 1200
Soil_Moisture: 0 -> 100
Drip/Mist/Fan: 0 hoặc 1
```

Bước này tránh sinh ra giá trị không thực tế.

## 8. Ghép thành data train

Sau khi tạo đủ số ngày:

```text
ngày augment 1
ngày augment 2
ngày augment 3
...
```

Ghép tất cả lại, sắp xếp theo `Timestamp`, rồi ghi ra:

```text
data/mini_greenhouse_5s_data.csv
```

## 9. Flow tổng quát

```text
raw data
  -> clean timestamp, missing, duplicate
  -> data sạch
  -> tách theo ngày
  -> chọn ngày cũ làm template
  -> đổi timestamp sang ngày mới
  -> thêm bias/noise nhỏ cho sensor
  -> shift nhẹ actuator
  -> clip giá trị
  -> ghép thành data train cuối
```

## 10. Lưu ý

Không nên augment bằng cách tự sinh lại toàn bộ hệ thống nếu mục tiêu là báo cáo dữ liệu dựa trên quá trình thu thập.

Cách nên trình bày:

```text
Dữ liệu train được mở rộng từ các ngày đã thu và xử lý sạch. Mỗi ngày mới được tạo bằng cách lấy lại mẫu biến thiên của một ngày cũ, thay đổi Timestamp, thêm sai lệch nhỏ cho cảm biến và dịch nhẹ thời điểm actuator. Nhờ vậy data augment vẫn giữ đặc trưng của dữ liệu gốc nhưng có thêm khác biệt giữa các ngày.
```

