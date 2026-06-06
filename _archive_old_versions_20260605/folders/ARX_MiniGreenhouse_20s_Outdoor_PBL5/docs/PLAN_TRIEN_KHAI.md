# Kế Hoạch Triển Khai Bản Có Cảm Biến Ngoài

## 1. Giả thuyết kỹ thuật

Nhà kính nhỏ không phải là hệ kín. Khi quạt bật:

- không khí ngoài đi vào;
- không khí trong đi ra;
- nhiệt độ và độ ẩm trong hộp thay đổi nhanh;
- tốc độ bay hơi nước khỏi đất thay đổi theo chênh lệch trong-ngoài.

Vì vậy, nếu chỉ dùng `Temperature_In` và `Humidity_In`, ARX có thể bỏ sót một phần nhiễu môi trường. Nếu thêm `Temperature_Out` và `Humidity_Out`, mô hình có thêm thông tin để giải thích tác động của quạt và môi trường ngoài.

## 2. Cách kiểm chứng

Không kết luận bằng cảm tính. Pipeline train các model trên cùng một dữ liệu:

1. `ARX baseline cũ chỉ cảm biến trong`: order gần kiểu cũ `ARX(5,1,2)`.
2. `ARX robust chỉ cảm biến trong`: search order theo validation robust score.
3. `ARX robust có cảm biến ngoài`: cùng search, nhưng thêm `Temperature_Out`, `Humidity_Out`, `Outdoor_Dryness` và `VPD_Proxy_Out`.

Model được chọn bằng:

```text
robust_score = mean(block_FIT_sim) - 0.5 * std(block_FIT_sim)
```

Test set chỉ dùng để báo cáo cuối, không dùng để chọn model.

Cấu hình mặc định dùng 16 ngày mô phỏng để có đủ biến động môi trường. Khi áp dụng thực tế trong 1-2 ngày thu dữ liệu, phần còn thiếu phải được bù bằng làm giàu dữ liệu có kiểm soát, và test thật vẫn phải giữ riêng.

## 3. Các cột đo khuyến nghị khi làm phần cứng

Schema nên log:

```csv
Timestamp,
Soil_Moisture,
Temperature_In,
Humidity_In,
Temperature_Out,
Humidity_Out,
Light_In,
Light_Out,
Drip,
Mist,
Fan,
Planned_Drip,
Planned_Mist,
Planned_Fan,
Safety_Override,
Command_Source
```

Nếu thiếu cảm biến ánh sáng ngoài thì vẫn làm được, nhưng tối thiểu nên có:

- `Soil_Moisture`;
- `Drip`;
- `Fan`;
- `Temperature_In`;
- `Humidity_In`;
- `Temperature_Out`;
- `Humidity_Out`;
- `Light_In`.

## 4. Nguyên tắc để không bị bắt lỗi

- Không shuffle time-series.
- Không fit scaler trên toàn bộ data.
- Không dùng test để chọn model.
- Không dùng trạng thái ẩn của mô phỏng để train.
- Không dùng `Soil_Moisture` tương lai để quyết định actuator.
- Planned pulse phải được lên lịch trước và có safety supervisor.
- Nếu kết quả thật không cải thiện khi thêm sensor ngoài, phải báo cáo trung thực.

## 5. Kết luận mong muốn

Nếu ARX có cảm biến ngoài tốt hơn rõ trên `FIT_sim` và `FIT_60`, có thể kết luận:

```text
Với mô hình nhà kính nhỏ, cảm biến ngoài giúp mô hình hóa nhiễu trao đổi khí do quạt, từ đó cải thiện dự báo free-run của độ ẩm đất. Vì ARX vẫn tuyến tính theo tham số, mô hình này phù hợp để đưa vào MPC tuyến tính, dễ giải thích và dễ kiểm chứng hơn các mô hình phi tuyến khi dữ liệu thật còn ít.
```

Nếu ARX có cảm biến ngoài chỉ ngang hoặc kém hơn nhẹ, kết luận đúng hơn là:

```text
Cảm biến ngoài vẫn nên được log để kiểm soát nhiễu môi trường và phục vụ phân tích, nhưng model triển khai cuối có thể chọn bản chỉ cảm biến trong nếu validation/test cho thấy nó ổn định hơn. Không nên thêm biến chỉ vì cảm giác là hợp lý; phải để dữ liệu quyết định.
```
