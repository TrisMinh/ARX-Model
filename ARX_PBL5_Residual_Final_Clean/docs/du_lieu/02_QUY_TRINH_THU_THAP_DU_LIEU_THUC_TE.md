# Quy trình thu thập dữ liệu thực tế

## 1. Mục tiêu

Dữ liệu cần đủ để train hai tầng:

```text
ARX backbone -> residual correction
```

Vì vậy log phải có cả biến cần dự đoán, biến môi trường trong mô hình và trạng thái actuator.

## 2. Tần số lấy mẫu

Với mô hình `30x50x30 cm`, chu kỳ hợp lý là:

```text
20 giây/mẫu
```

Lý do:

- mô hình nhỏ phản ứng nhanh;
- bơm, quạt và phun sương có tác động ngắn;
- 20 giây vẫn đủ nhẹ để lưu CSV trong 1 đến 2 ngày;
- 60 bước tương đương 20 phút, đủ để đánh giá dự báo nhiều bước.

## 3. Cột dữ liệu tối thiểu

CSV tối thiểu:

```csv
Timestamp,Soil_Moisture,Temperature_In,Humidity_In,Light_In,Drip,Fan
```

Nên có thêm:

```csv
Mist,Planned_Drip,Planned_Mist,Planned_Fan,Safety_Override,Command_Source
```

Ý nghĩa:

- `Soil_Moisture`: độ ẩm đất, là output cần dự đoán;
- `Temperature_In`: nhiệt độ trong mô hình;
- `Humidity_In`: độ ẩm không khí trong mô hình;
- `Light_In`: ánh sáng trong mô hình;
- `Drip`: bơm nhỏ giọt đang bật hay tắt;
- `Mist`: phun sương đang bật hay tắt;
- `Fan`: quạt đang bật hay tắt;
- `Planned_*`: lệnh kích thích dự định;
- `Safety_Override`: hệ an toàn có can thiệp hay không;
- `Command_Source`: lý do actuator bật.

## 4. Nguyên tắc kích thích actuator

Không chỉ chạy luật tự động kiểu đất khô thì tưới. Nếu chỉ chạy luật đó, dữ liệu sẽ bị nghèo vì actuator thường bật trong một vùng rất hẹp.

Cần có các đoạn kích thích an toàn:

- bật bơm ngắn khi đất còn trong vùng an toàn;
- bật quạt ở nhiều thời điểm sáng, trưa, chiều, tối;
- bật phun sương ngắn nếu phần cứng có;
- giữ một số đoạn không bật gì để model học khô tự nhiên.

Mọi kích thích phải có safety:

```text
Nếu Soil_Moisture quá cao thì chặn bơm.
Nếu nhiệt độ hoặc độ ẩm không khí vượt ngưỡng nguy hiểm thì chặn actuator tương ứng.
```

## 5. Lịch thu trong 1 đến 2 ngày

Nếu chỉ có 1 ngày:

| Giai đoạn | Thời lượng | Mục tiêu |
|---|---:|---|
| Sáng | 2 giờ | quạt, ánh sáng tăng, đất khô dần |
| Trưa | 2 giờ | nhiệt cao, quạt/phun sương có tác động rõ |
| Chiều | 2 giờ | ánh sáng giảm, tưới ngắn |
| Tối | 2 giờ | khô chậm, ít ánh sáng |
| Qua đêm | 6 đến 8 giờ | đoạn tự nhiên dài |

Nếu có 2 ngày, ngày 1 dùng để train/validation, ngày 2 giữ nhiều hơn cho test. Không trộn ngẫu nhiên theo dòng.

## 6. Làm giàu dữ liệu đúng cách

Được làm:

- nội suy timestamp nhỏ nếu mất vài mẫu;
- thêm nhiễu đo nhỏ để kiểm tra độ bền, nhưng phải ghi rõ là augmentation;
- tạo feature từ quá khứ và hiện tại như lag, rolling, cạnh bật actuator;
- mô phỏng thêm dữ liệu bằng mô hình vật lý đã mô tả rõ.

Không được làm:

- copy test vào train;
- dùng `Soil_Moisture` tương lai làm input;
- chọn model vì test đẹp;
- xóa các đoạn model dự đoán xấu để tăng FIT.

## 7. Chạy sau khi có CSV thật

```powershell
python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py --data-csv .\data\log_that.csv
```

Kết quả thật phải lưu riêng, không trộn với kết quả mô phỏng.
