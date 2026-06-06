# Hướng Dẫn Thu Thập Dữ Liệu Thực Tế Cho ARX

Tài liệu này dùng cho mô hình nhà kính nhỏ kích thước khoảng `30x50x30 cm`, lấy mẫu mỗi `20 giây`. Mục tiêu là để một người mới bắt đầu vẫn có thể thu được dữ liệu đủ đúng về học thuật, đủ an toàn khi chạy mô hình thật, và đủ rõ ràng để huấn luyện ARX/NARX mà không bị bắt lỗi do rò rỉ dữ liệu.

Điểm quan trọng nhất:

- Không để AI/MPC tự điều khiển ngay từ đầu.
- Ban đầu phải có luật an toàn đơn giản để bảo vệ cây, đất, bơm và cảm biến.
- Dữ liệu phải có kích thích điều khiển nhỏ, có kiểm soát, để mô hình học được tác động của bơm/quạt/phun sương.
- Test thật phải được giữ riêng, không dùng để chọn mô hình.
- Dữ liệu làm giàu chỉ được dùng cho train/pretrain, không được dùng để thay thế kết quả kiểm tra thật.

---

## 1. Mục Tiêu Của Việc Thu Dữ Liệu

ARX cần học quan hệ động học:

```text
Soil_Moisture hiện tại
= hàm của Soil_Moisture quá khứ
+ Temperature/Humidity/Light/Drip/Mist/Fan quá khứ
```

Muốn mô hình học đúng, dữ liệu phải có đủ các trạng thái:

- đất đang khô dần khi không tưới;
- đất tăng ẩm sau khi bật bơm;
- khoảng trễ sau khi bật bơm;
- nhiệt độ/độ ẩm không khí thay đổi khi bật quạt/phun sương;
- điều kiện sáng, trưa, chiều/tối;
- vùng đất hơi khô, vùng an toàn, và vùng gần ẩm.

Nếu chỉ để hệ chạy kiểu “đất khô thì bơm”, dữ liệu sẽ bị closed-loop bias. Khi đó mô hình khó phân biệt:

```text
đất khô -> bơm bật
```

với:

```text
bơm bật -> đất tăng ẩm
```

Vì vậy cần thêm các xung kích thích nhỏ, an toàn, được lên lịch trước. Trong nhận dạng hệ thống, đây gọi là `planned excitation` hoặc `persistent excitation`.

---

## 2. Cấu Hình Lấy Mẫu Khuyến Nghị

Với mô hình nhỏ `30x50x30 cm`, nên dùng:

```text
Sampling time = 20 giây/mẫu
```

Lý do:

- Thể tích không khí nhỏ nên nhiệt độ và độ ẩm không khí có thể đổi nhanh khi bật quạt/phun sương.
- Độ ẩm đất chậm hơn không khí, nhưng nếu cảm biến gần vùng tưới thì đáp ứng vẫn có thể xuất hiện sau khoảng `40-120 giây`.
- `20 giây/mẫu` đủ nhanh để bắt đáp ứng quá độ, nhưng chưa quá nặng cho ESP32, máy tính hoặc file CSV.

Khi báo cáo, không được chỉ dùng `FIT_1step` vì 1 bước ở đây chỉ là 20 giây. Cần báo cáo thêm:

| Chỉ số | Ý nghĩa |
| --- | --- |
| `FIT_1step` | dự báo sau 20 giây |
| `FIT_12` | dự báo sau 4 phút |
| `FIT_60` | dự báo sau 20 phút |
| `FIT_sim` | mô phỏng free-run, khó hơn và sát ứng dụng điều khiển hơn |

---

## 3. Chuẩn Bị Phần Cứng

### 3.1. Cảm biến tối thiểu

Cần có:

- cảm biến độ ẩm đất;
- cảm biến nhiệt độ không khí;
- cảm biến độ ẩm không khí;
- cảm biến ánh sáng hoặc ít nhất là trạng thái ngày/đêm;
- log trạng thái bơm tưới nhỏ giọt;
- log trạng thái quạt;
- log trạng thái phun sương nếu có.

### 3.2. Actuator tối thiểu

Nên có:

- `Drip`: bơm tưới nhỏ giọt, dạng `0/1` hoặc PWM;
- `Fan`: quạt, dạng `0/1`;
- `Mist`: phun sương, dạng `0/1`.

Nếu đồ án chỉ có bơm, vẫn làm được ARX cho độ ẩm đất, nhưng phần nhiệt độ/độ ẩm không khí sẽ nghèo hơn. Khi bảo vệ cần nói rõ giới hạn này.

### 3.3. Kiểm tra trước khi thu

Trước khi chạy thu dữ liệu chính:

1. Bật hệ thống, chưa bật actuator.
2. Log liên tục ít nhất 10 phút.
3. Kiểm tra thời gian giữa hai mẫu có gần `20 giây` không.
4. Bật thử từng actuator trong `20-40 giây`.
5. Xem cảm biến có nhảy vô lý, mất kết nối, hoặc đứng yên bất thường không.
6. Kiểm tra nước có chảy đúng vùng cảm biến đất không.

Nếu cảm biến độ ẩm đất nhiễu mạnh, không xóa dữ liệu tùy tiện. Hãy giữ raw data, ghi chú lỗi, rồi xử lý ở file cleaned.

---

## 4. Schema Dữ Liệu Cần Log

File CSV tối thiểu nên có các cột sau:

| Tên cột | Ý nghĩa | Ghi chú |
| --- | --- | --- |
| `Timestamp` | thời điểm đo | dạng ISO hoặc `YYYY-MM-DD HH:MM:SS` |
| `Soil_Moisture` | độ ẩm đất | phần trăm hoặc đơn vị đã hiệu chuẩn |
| `Temperature` | nhiệt độ không khí | độ C |
| `Humidity` | độ ẩm không khí | phần trăm |
| `Light` | ánh sáng | lux hoặc giá trị ADC đã quy đổi |
| `Drip` | bơm tưới nhỏ giọt | `0/1` |
| `Mist` | phun sương | `0/1` |
| `Fan` | quạt | `0/1` |
| `Soil_Low_SP` | ngưỡng dưới an toàn | ví dụ `55` |
| `Soil_High_SP` | ngưỡng trên an toàn | ví dụ `65` |
| `Command_Source` | nguồn lệnh actuator | `none`, `manual_test`, `safety_rescue`, `planned_drip_excitation`, ... |
| `Planned_Drip` | bơm được lên lịch trước | `0/1` |
| `Planned_Mist` | phun sương được lên lịch trước | `0/1` |
| `Planned_Fan` | quạt được lên lịch trước | `0/1` |
| `Safety_Override` | luật an toàn có can thiệp không | `0/1` |

Nên log thêm nếu làm được:

| Tên cột | Ý nghĩa |
| --- | --- |
| `Mode` | `manual`, `safety`, `identification`, `validation` |
| `Pump_PWM` | mức PWM nếu bơm không chỉ có `0/1` |
| `Sensor_Raw` | giá trị ADC gốc của cảm biến đất |
| `Water_Level` | mực nước bình chứa |
| `Note` | ghi chú sự kiện bất thường |

Ví dụ một dòng CSV:

```csv
Timestamp,Soil_Moisture,Temperature,Humidity,Light,Drip,Mist,Fan,Soil_Low_SP,Soil_High_SP,Command_Source,Planned_Drip,Planned_Mist,Planned_Fan,Safety_Override
2026-06-05 08:00:00,57.2,29.1,72.5,530,0,0,0,55,65,none,0,0,0,0
```

---

## 5. Luật An Toàn Ban Đầu

Quy trình đúng là:

```text
Rule-based safety -> thu dữ liệu -> train ARX -> validate -> MPC/AI dùng ARX
```

Không nên nói “AI tự học để tưới” ngay từ đầu, vì khi chưa có dữ liệu và chưa có mô hình thì AI/MPC chưa có cơ sở điều khiển.

Luật an toàn gợi ý:

```text
Nếu Soil_Moisture < 54.2%:
    bật bơm 40-60 giây

Nếu Soil_Moisture > 65.8%:
    chặn mọi planned drip pulse

Nếu Temperature > 31 độ C hoặc Humidity > 88%:
    bật quạt

Nếu Temperature > 31 độ C và Humidity < 68%:
    cho phép phun sương ngắn
```

Yêu cầu khi chạy:

- Luật an toàn chỉ dùng cảm biến hiện tại/quá khứ.
- Không dùng dữ liệu tương lai.
- Khi luật an toàn can thiệp, ghi `Safety_Override = 1`.
- Khi planned pulse bị chặn vì đất quá ẩm, ghi `Command_Source = planned_drip_blocked_wet`.

---

## 6. Lịch Thu Dữ Liệu Nếu Chỉ Có 1 Ngày

Nếu chỉ có một ngày, vẫn có thể thu được dữ liệu dùng được nếu chia đúng giai đoạn.

### Giai đoạn A: kiểm tra hệ thống, 30 phút

Mục tiêu: đảm bảo cảm biến và actuator hoạt động.

Thao tác:

1. Chạy hệ thống, chưa bật actuator.
2. Log liên tục 30 phút.
3. Kiểm tra `Timestamp` có đều `20 giây` không.
4. Kiểm tra sensor có NaN, mất tín hiệu, hoặc nhảy bất thường không.
5. Bật thử từng actuator thủ công:
   - bơm `20-40 giây`;
   - quạt `20-40 giây`;
   - phun sương `20-40 giây`.
6. Ghi `Command_Source = manual_test`.

### Giai đoạn B: chạy luật an toàn, 1-2 giờ

Mục tiêu: quan sát hệ vận hành bình thường.

Thao tác:

1. Chạy rule-based safety.
2. Không thêm planned pulse.
3. Log liên tục.
4. Ghi nguồn lệnh rõ ràng:
   - `none` nếu không actuator;
   - `safety_rescue` nếu luật an toàn bật bơm;
   - `safety_fan` nếu luật an toàn bật quạt;
   - `safety_mist` nếu luật an toàn bật phun sương.

### Giai đoạn C: nhận dạng hệ thống, 4-6 giờ

Mục tiêu: tạo dữ liệu để ARX học tác động của actuator.

Nguyên tắc:

- Xung phải được lên lịch trước theo thời gian.
- Xung không được quyết định từ `Soil_Moisture` tương lai.
- Xung phải nhỏ và vẫn chịu giám sát bởi luật an toàn.
- Nếu đất quá ẩm, chặn xung tưới.
- Nếu đất quá khô, luật an toàn được quyền bật bơm.

Lịch gợi ý cho bơm nhỏ giọt:

| Thời điểm tính từ lúc bắt đầu giai đoạn C | Lệnh | Thời lượng |
| --- | --- | ---: |
| 30 phút | Drip pulse | 2 phút |
| 90 phút | Drip pulse | 3 phút |
| 130 phút | Drip pulse | 4 phút |
| 180 phút | Drip pulse | 2 phút |
| 240 phút | Drip pulse | 5 phút |
| 300 phút | Drip pulse | 3 phút |

Nếu đất tăng quá nhanh:

- giảm thời lượng pulse xuống `40-90 giây`;
- không tăng số pulse;
- tăng thời gian nghỉ giữa hai pulse.

Nếu đất gần như không đổi:

- kiểm tra bơm có tưới đúng vùng cảm biến không;
- tăng pulse thêm `1-2 phút`;
- không tăng quá mạnh trong một lần.

Lịch gợi ý cho quạt/phun sương:

| Thời điểm | Lệnh | Thời lượng |
| --- | --- | ---: |
| sáng hoặc trưa | Fan | 1-2 phút |
| trưa nóng | Mist | 40-90 giây |
| chiều | Fan | 1-2 phút |

Mục tiêu của quạt/phun sương là tạo biến động `Temperature` và `Humidity`; không nhất thiết phải làm tăng `Soil_Moisture`.

### Giai đoạn D: validation holdout, 1-2 giờ cuối

Mục tiêu: giữ dữ liệu thật để đánh giá mô hình.

Thao tác:

1. Vẫn lấy mẫu `20 giây/mẫu`.
2. Không dùng đoạn này để chọn mô hình.
3. Có thể thêm 1-2 pulse nhỏ, nhưng phải log rõ `Planned_*`.
4. Sau khi train, đoạn này dùng làm validation/test thật.

---

## 7. Lịch Thu Dữ Liệu Nếu Có 2 Ngày

Nếu có 2 ngày, kết quả đáng tin hơn.

### Ngày 1

| Khoảng thời gian | Mục tiêu |
| --- | --- |
| 30 phút đầu | kiểm tra sensor/actuator |
| 1-2 giờ tiếp | chạy rule-based safety |
| 4-6 giờ tiếp | planned excitation chính |
| 1 giờ cuối | validation nhanh |

### Ngày 2

| Khoảng thời gian | Mục tiêu |
| --- | --- |
| 1-2 giờ đầu | chạy rule-based trong điều kiện khác |
| 3-5 giờ tiếp | planned excitation nhẹ hơn ngày 1 |
| 2 giờ cuối | test holdout |

Chia dữ liệu theo thời gian:

```text
Train: phần đầu và phần giữa
Validation: cuối ngày 1 hoặc đầu ngày 2
Test: 2 giờ cuối ngày 2
```

Nếu chỉ có 1 ngày:

```text
Train: 70% đầu
Validation: 15% tiếp theo
Test: 15% cuối
```

Không shuffle chuỗi thời gian.

---

## 8. Cách Đặt Planned Pulse Cho Đúng Học Thuật

Planned pulse hợp lệ khi:

- được lên lịch trước;
- không dùng `Soil_Moisture` tương lai;
- không dùng test label;
- có safety override;
- được log riêng bằng `Planned_Drip`, `Planned_Mist`, `Planned_Fan`;
- nếu bị luật an toàn chặn thì vẫn phải ghi rõ lý do.

Ví dụ logic:

```python
if current_time in planned_drip_schedule:
    Planned_Drip = 1
else:
    Planned_Drip = 0

if Soil_Moisture > 65.8:
    Drip = 0
    Safety_Override = 1
    Command_Source = "planned_drip_blocked_wet"
elif Soil_Moisture < 54.2:
    Drip = 1
    Safety_Override = 1
    Command_Source = "safety_rescue"
elif Planned_Drip == 1:
    Drip = 1
    Safety_Override = 0
    Command_Source = "planned_drip_excitation"
else:
    Drip = 0
    Safety_Override = 0
    Command_Source = "none"
```

Câu trả lời khi bị hỏi:

> Planned pulse không phải là gian lận. Đây là kỹ thuật tạo persistent excitation trong nhận dạng hệ thống. Xung được lên lịch trước, không dùng dữ liệu tương lai, và vẫn chịu giám sát bởi luật an toàn.

---

## 9. Kiểm Tra Dữ Liệu Ngay Sau Khi Thu

### 9.1. Kiểm tra missing

Cần kiểm tra:

- `Timestamp` có bị thiếu không;
- sensor có NaN không;
- actuator có log đủ `0/1` không;
- file có bị trùng dòng không.

Nếu thiếu ít mẫu:

- sensor có thể nội suy nhẹ;
- actuator không nên nội suy bừa, chỉ dùng trạng thái lệnh gần nhất nếu biết relay thực sự đang giữ trạng thái đó.

### 9.2. Kiểm tra sampling

Với `20 giây/mẫu`, cần:

```text
Timestamp[i] - Timestamp[i-1] xấp xỉ 20 giây
```

Nếu có gap lớn do mất mạng hoặc mất nguồn:

- ghi chú;
- tách thành nhiều segment;
- không train xuyên qua gap như thể dữ liệu liên tục.

### 9.3. Kiểm tra độ dao động của đất

Vùng hợp lý:

```text
Soil_Moisture dao động khoảng 54-65%
```

Nếu chỉ dao động rất hẹp, ví dụ `56-57%`:

- dữ liệu quá phẳng;
- `FIT_1step` có thể cao ảo;
- `FIT_sim` dễ drift;
- cần thêm planned pulse an toàn.

### 9.4. Kiểm tra tỷ lệ actuator ON

Gợi ý:

```text
Drip ON khoảng 1-5% tổng thời gian
Planned_Drip khoảng 0.5-2% tổng thời gian
```

Với 1 ngày:

```text
Planned_Drip = 1% tương đương khoảng 14.4 phút/ngày
```

Tỷ lệ này hợp lý nếu dùng bơm nhỏ, chia thành nhiều xung ngắn, và vẫn có chặn khi đất quá ẩm.

### 9.5. Kiểm tra planned pulse có độc lập tương đối với đất không

Sau khi thu, kiểm tra:

```text
corr(Planned_Drip, Soil_Moisture_prev - SP_Center)
```

Mong muốn:

```text
gần 0
```

Nếu tương quan âm rất mạnh, ví dụ `-0.6`, nghĩa là planned pulse thực ra vẫn bị quyết định bởi đất khô. Khi đó không còn là planned excitation độc lập nữa.

---

## 10. Làm Giàu Dữ Liệu Khi Chỉ Có 1-2 Ngày

Nguyên tắc bắt buộc:

```text
Raw real test luôn giữ riêng.
Synthetic/augmented data chỉ dùng cho train hoặc pretrain.
Kết quả cuối phải báo cáo trên real validation/test.
```

### 10.1. Cách 1: residual noise augmentation

Quy trình:

1. Fit một ARX sơ bộ trên train thật.
2. Tính residual:

```text
residual = y_true - y_pred
```

3. Sinh thêm chuỗi train bằng cách cộng nhiễu có phân phối giống residual.
4. Không dùng cách này để tạo test.

Mục tiêu:

- làm mô hình bớt nhạy với nhiễu cảm biến;
- kiểm tra robustness;
- không làm đẹp test giả.

### 10.2. Cách 2: parameter perturbation

Fit một plant đơn giản từ dữ liệu thật, rồi thay đổi nhẹ tham số:

```text
water_gain +/- 10-20%
evaporation_gain +/- 10-20%
sensor_noise +/- 20%
delay +/- 1 sample
```

Điều kiện:

- output vẫn nằm trong vùng thực tế;
- tham số không được phi lý;
- dữ liệu sinh thêm phải đánh dấu `Data_Source = synthetic_augmented`;
- kết quả cuối vẫn phải đo trên test thật.

### 10.3. Cách 3: replay thời tiết/ánh sáng thật

Dùng `Temperature`, `Humidity`, `Light` thật đã thu, nhưng tạo thêm vài lịch planned pulse khác nhau rồi chạy plant đã hiệu chỉnh để sinh soil response.

Cách này hợp lý vì:

- nhiễu môi trường vẫn bám dữ liệu thật;
- actuator plan đa dạng hơn;
- mô hình học tốt hơn đáp ứng của bơm.

### 10.4. Cách 4: ghép segment có điều kiện

Chỉ ghép hai đoạn dữ liệu nếu trạng thái cuối đoạn A gần trạng thái đầu đoạn B:

```text
abs(Soil_end_A - Soil_start_B) < 0.5%
abs(Temp_end_A - Temp_start_B) < 1.0 độ C
abs(Humi_end_A - Humi_start_B) < 3%
```

Không ghép bừa vì sẽ tạo bước nhảy giả khiến mô hình học sai động học.

### 10.5. Những việc không được làm

Không làm:

- copy test vào train;
- shuffle time-series rồi chia train/test;
- tạo synthetic data rồi báo như data thật;
- dùng `Soil_Moisture` tương lai để quyết định actuator;
- fit scaler trên toàn bộ dữ liệu trước khi split;
- chọn mô hình theo test;
- xóa đoạn test khó để tăng FIT.

---

## 11. Pipeline Huấn Luyện Sau Khi Có Dữ Liệu

### Bước 1: lưu raw data

Lưu file gốc:

```text
data/raw/mini_greenhouse_real_raw.csv
```

Không chỉnh trực tiếp file raw.

### Bước 2: tạo cleaned data

Tạo:

```text
data/processed/mini_greenhouse_real_cleaned.csv
```

Chỉ xử lý:

- parse timestamp;
- sort theo thời gian;
- xử lý missing nhỏ;
- chuẩn hóa tên cột;
- thêm đặc trưng thời gian như `Hour_sin`, `Hour_cos`;
- giữ lại cột `Command_Source` và `Safety_Override`.

### Bước 3: chia dữ liệu theo thời gian

Ví dụ:

```text
Train: 70%
Validation: 15%
Test: 15%
```

Không shuffle.

### Bước 4: scale bằng train only

Mean/std chỉ fit trên train.

Không fit scaler trên toàn bộ dữ liệu.

### Bước 5: tìm ARX theo ý nghĩa vật lý

Với sampling `20 giây`, nên search theo thời gian nhớ vật lý:

| Tham số | Giá trị gợi ý | Ý nghĩa |
| --- | --- | --- |
| `na` | `6, 12, 18, 24` | nhớ output khoảng `2-8 phút` |
| `nb` | `3, 6, 9` | nhớ input khoảng `1-3 phút` |
| `nk` | `1, 2, 3` | trễ input khoảng `20-60 giây` |
| `alpha` | `0.01, 0.1, 1.0` | Ridge nhẹ để tránh overfit |

Không so trực tiếp `na=18` ở sampling `20 giây` với `na=18` ở sampling `5 phút`. Cần đổi về thời gian:

```text
physical_memory = na * sampling_time
```

Ví dụ:

```text
na = 18, sampling = 20 giây -> nhớ output 360 giây = 6 phút
```

Vì vậy `na=18` không phải quá lớn nếu lấy mẫu nhanh.

### Bước 6: chọn mô hình bằng validation robust

Chia validation thành nhiều block:

```text
robust_score = mean(block_FIT_sim) - 0.5 * std(block_FIT_sim)
```

Ý nghĩa:

- mô hình phải fit đều nhiều đoạn;
- không chỉ ăn may ở một đoạn validation;
- giảm nguy cơ chọn mô hình quá nhạy với một đoạn dữ liệu.

### Bước 7: báo cáo trên test

Bắt buộc báo cáo:

- `FIT_1step`;
- `FIT_12`;
- `FIT_60`;
- `FIT_sim`;
- RMSE;
- kiểm tra residual;
- mô tả rõ đoạn test không dùng để chọn mô hình.

---

## 12. Cách Trả Lời Khi Bị Hỏi

### Câu hỏi: dữ liệu này có phải dữ liệu thật không?

Trả lời:

> Dữ liệu mô phỏng chỉ dùng để kiểm thử pipeline và thiết kế quy trình. Khi triển khai trên mô hình thật, nhóm thu dữ liệu theo cùng schema, cùng luật an toàn và cùng nguyên tắc planned excitation. Mô hình cuối phải huấn luyện và kiểm tra lại bằng dữ liệu thật.

### Câu hỏi: làm giàu dữ liệu có phải fake không?

Trả lời:

> Không, nếu dùng đúng. Dữ liệu làm giàu chỉ dùng cho train hoặc pretrain và được đánh dấu synthetic. Đoạn test thật luôn giữ riêng, không làm giàu, không dùng để chọn mô hình.

### Câu hỏi: tại sao không chỉ thu bình thường?

Trả lời:

> Nếu chỉ thu bình thường, actuator thường bật khi đất đã khô, tạo closed-loop bias. Planned excitation giúp mô hình học được đáp ứng nhân-quả của actuator, nhưng vẫn có safety override để không làm hại hệ.

### Câu hỏi: tại sao lấy mẫu 20 giây?

Trả lời:

> Vì mô hình `30x50x30 cm` có thể tích nhỏ, nhiệt độ và độ ẩm không khí thay đổi nhanh khi bật quạt/phun sương. Tuy nhiên độ ẩm đất chậm hơn nên nhóm không chỉ báo cáo 1-step 20 giây, mà đánh giá thêm horizon 4 phút, 20 phút và free-run simulation.

### Câu hỏi: AI dự đoán thì tại sao vẫn cần logic bơm?

Trả lời:

> Ở giai đoạn thu dữ liệu, logic bơm là safety supervisor và excitation scheduler, không phải MPC cuối cùng. Sau khi có dữ liệu và mô hình ARX đủ tin cậy, MPC mới dùng mô hình đó để tối ưu lệnh điều khiển. Dù dùng MPC, hệ thực vẫn cần lớp safety supervisor để chặn các lệnh nguy hiểm.

---

## 13. Checklist Trước Khi Nộp Báo Cáo

- [ ] Có raw CSV.
- [ ] Có cleaned CSV.
- [ ] Có mô tả sampling time.
- [ ] Có schema cột.
- [ ] Có luật an toàn ban đầu.
- [ ] Có planned excitation.
- [ ] Có `Command_Source`.
- [ ] Có `Safety_Override`.
- [ ] Có split theo thời gian.
- [ ] Có train/validation/test riêng.
- [ ] Có ARX baseline đơn giản, ví dụ `ARX(5,1,2)`.
- [ ] Có ARX robust search.
- [ ] Có NARX comparison nếu cần.
- [ ] Có `FIT_1step`, `FIT_12`, `FIT_60`, `FIT_sim`.
- [ ] Có tự phản biện về dữ liệu mô phỏng, dữ liệu thật, augmentation, leakage và closed-loop bias.

---

## 14. Mẫu Kết Luận Đưa Vào Báo Cáo

```text
Trong giai đoạn đầu, hệ thống được vận hành bằng rule-based safety để đảm bảo độ ẩm đất không vượt vùng an toàn. Sau đó, nhóm bổ sung các xung kích thích nhỏ đã lên lịch trước cho bơm/quạt/phun sương nhằm tạo persistent excitation phục vụ nhận dạng hệ thống. Các xung này không được sinh từ Soil_Moisture tương lai mà dựa trên lịch thời gian, đồng thời vẫn chịu giám sát bởi safety supervisor. Dữ liệu thu được được chia theo thời gian thành train/validation/test, không shuffle. Model ARX được chọn bằng validation robust score và đánh giá cuối trên test set thông qua các chỉ số 1-step, multi-step và free-run simulation.
```
