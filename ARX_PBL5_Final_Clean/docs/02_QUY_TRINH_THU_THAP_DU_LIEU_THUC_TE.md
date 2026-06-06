# Quy trình thu thập dữ liệu thực tế trong 1 đến 2 ngày

## 1. Mục tiêu thu dữ liệu

Thu dữ liệu không chỉ để có nhiều dòng CSV. Dữ liệu phải giúp model học được:

- đất khô tự nhiên như thế nào;
- bơm làm độ ẩm tăng ra sao;
- độ trễ từ lúc bật bơm đến lúc cảm biến thấy thay đổi;
- quạt và ánh sáng làm tốc độ khô thay đổi thế nào;
- hệ phản ứng khác nhau giữa sáng, trưa, chiều, tối.

Nếu dữ liệu chỉ toàn trạng thái ổn định, ARX sẽ không học được động học actuator.

## 2. Không dùng AI điều khiển ngay từ đầu

Trong giai đoạn thu dữ liệu nhận dạng, không nên để AI tự quyết định bơm. Lý do:

- model chưa học nên quyết định chưa đáng tin;
- AI có thể giữ hệ quanh một vùng quá hẹp;
- dữ liệu thiếu kích thích;
- model sau đó học kém tác động actuator.

Cách đúng:

```text
Lịch kích thích có kiểm soát + safety supervisor.
```

Safety supervisor là lớp bảo vệ:

- đất quá khô thì bật bơm cứu;
- đất quá ẩm thì chặn tưới;
- nhiệt độ/độ ẩm không khí nguy hiểm thì bật quạt/phun sương theo luật an toàn.

## 3. Cột CSV bắt buộc

CSV tối thiểu cho ARX final:

```csv
Timestamp,Soil_Moisture,Temperature_In,Humidity_In,Light_In,Drip,Fan
```

Nên có thêm:

```csv
Mist,Planned_Drip,Planned_Mist,Planned_Fan,Safety_Override,Command_Source,Soil_Low_SP,Soil_High_SP
```

Ý nghĩa:

| Cột | Ý nghĩa |
|---|---|
| `Timestamp` | thời điểm lấy mẫu |
| `Soil_Moisture` | độ ẩm đất |
| `Temperature_In` | nhiệt độ trong mô hình |
| `Humidity_In` | độ ẩm không khí trong mô hình |
| `Light_In` | ánh sáng trong mô hình |
| `Drip` | bơm nhỏ giọt, 0/1 |
| `Mist` | phun sương, 0/1 |
| `Fan` | quạt, 0/1 |
| `Planned_*` | lệnh kích thích đã lên lịch |
| `Safety_Override` | cờ luật an toàn can thiệp |
| `Command_Source` | nguồn tạo lệnh |

## 4. Kiểm tra trước khi thu chính

Trước khi log dữ liệu:

1. Kiểm tra cảm biến đất bằng đất khô và đất ẩm.
2. Kiểm tra DHT/HTU/SHT đọc nhiệt độ và độ ẩm hợp lý.
3. Kiểm tra bơm bật/tắt đúng log `Drip`.
4. Kiểm tra quạt bật/tắt đúng log `Fan`.
5. Chạy thử 20 phút và mở CSV xem có mất dòng không.
6. Đảm bảo sampling gần `20 giây/mẫu`.

Không đổi vị trí cảm biến đất giữa chừng. Nếu đổi, phải ghi chú thời điểm đổi.

## 5. Lịch thu dữ liệu 1 ngày

| Thời gian | Mục tiêu | Thao tác |
|---|---|---|
| 07:00-08:00 | nền buổi sáng | chỉ log, không kích thích mạnh |
| 08:00-09:00 | xung bơm | bật bơm 20-60 giây, lặp 2-3 lần |
| 10:30-11:30 | tác động quạt | bật quạt 1-3 phút, lặp 2-3 lần |
| 12:30-13:30 | nóng/ánh sáng mạnh | log tự nhiên, safety nếu cần |
| 14:30-15:30 | bơm và quạt lệch nhau | giúp model phân biệt tác động |
| 17:00-18:00 | ánh sáng giảm | log quá trình chậm lại |
| 20:00-21:00 | ban đêm | log ít ánh sáng |

## 6. Nếu có 2 ngày

Ngày 1:

- nhiều kích thích hơn;
- dùng chủ yếu cho train/validation.

Ngày 2:

- không lặp y hệt ngày 1;
- giữ một phần cuối làm test thật.

Không dùng đoạn test để chọn model.

## 7. Làm giàu dữ liệu đúng cách

Vì thời gian thu chỉ 1-2 ngày, có thể làm giàu dữ liệu, nhưng phải tách rõ:

- dữ liệu thật;
- dữ liệu tổng hợp;
- test thật.

Cách chấp nhận được:

1. Giữ một đoạn thật làm test cuối.
2. Ước lượng nhiễu cảm biến từ train.
3. Sinh thêm train bằng cách thay đổi nhẹ thời điểm bơm/quạt.
4. Không copy test vào train.
5. Báo cáo kết quả trên test thật riêng.

Cách không được làm:

- shuffle time-series;
- dùng output tương lai;
- cắt đoạn dự đoán kém;
- gọi dữ liệu mô phỏng là dữ liệu thật.

## 8. Sau khi thu xong chạy thế nào?

```powershell
python -B .\ARX_PBL5_Final_Clean\src\arx_final_pipeline.py --data-csv .\data\log_that.csv
```

Nếu muốn thử residual:

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py --data-csv .\data\log_that.csv
```
