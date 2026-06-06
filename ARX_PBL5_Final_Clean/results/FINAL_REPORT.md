# Báo Cáo Cuối: ARX Chỉ Dùng Cảm Biến Trong

## Kết luận ngắn

- Bản triển khai cuối chỉ dùng các tín hiệu đo trong mô hình và actuator.
- Đã bỏ dòng baseline cũ `ARX(5,1,2)` khỏi báo cáo chính vì nó chỉ là phép thử lịch sử và gây hiểu nhầm khi chạy trên bộ dữ liệu 20 giây mới.
- Đã bỏ bản có cảm biến ngoài khỏi pipeline triển khai chính để giữ mô hình gọn, dễ bảo vệ và đúng yêu cầu hiện tại.
- Kết quả dưới đây là kết quả chạy thật của pipeline trên dữ liệu mô phỏng vật lý có log rõ, không phải dữ liệu phần cứng.
- Bản chính ưu tiên cấu trúc ARX gọn 17 input, không trộn các script thử nghiệm mở rộng vào pipeline bảo vệ.
- ARX robust chỉ cảm biến trong đạt `FIT_sim = 78.950` và `FIT_60 = 88.047`.

## Kiểm tra dữ liệu

| Hạng mục | Giá trị |
| --- | ---: |
| Thể tích nhà kính | 0.0450 m3 |
| Thời gian lấy mẫu | 20 giây |
| Số dòng dữ liệu | 69120 |
| Missing | 0 |
| Timestamp trùng | 0 |
| Sai chu kỳ lấy mẫu | 0 |
| Soil min-max | 54.007 - 61.642 |
| Soil std | 1.306 |
| Temperature_In range | 22.206 - 36.479 |
| Humidity_In range | 52.334 - 88.530 |
| Drip ON % | 2.418 |
| Fan ON % | 42.124 |
| Planned Drip % | 1.105 |
| Planned Fan % | 0.318 |
| Safety override % | 0.440 |
| corr(Planned Drip, prev soil-center) | 0.006 |
| corr(Planned Fan, prev soil-center) | 0.034 |

Các planned pulse được tạo theo lịch/seed và vẫn bị safety supervisor chặn nếu nguy hiểm. Vì vậy đây là nhận dạng hệ thống có kiểm soát, không phải dùng tương lai để làm đẹp kết quả.

## So sánh mô hình

| Trường hợp | Họ mô hình | Input | Số cột input | Model chọn | FIT_1step 20s | FIT_12 4 phút | FIT_60 20 phút | FIT_sim | RMSE_sim | Ghi chú |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ARX robust chỉ cảm biến trong | ARX tuyến tính | inside_only | 17 | `ARX_na12_nb3_nk2_alpha0.1` | 95.373 | 90.839 | 88.047 | 78.950 | 0.2021 | Sản phẩm chính: không dùng cảm biến ngoài, chọn bằng validation robust score |

## Chi tiết ARX

- Model chọn `ARX_na12_nb3_nk2_alpha0.1`, robust validation score `75.270`.
- Input delay `40` giây, output memory `240` giây, input memory `60` giây.
- Residual free-run max abs ACF lag 1..60: `0.960`.

## Kết luận cho đồ án

Bản final giữ ARX chỉ cảm biến trong vì đây là mô hình gọn, dễ giải thích, đủ tốt trên test mô phỏng và thuận lợi để đưa vào MPC tuyến tính. Nếu sau này có dữ liệu thật dài hơn, cảm biến ngoài có thể được thử lại như một nghiên cứu mở rộng, không phải sản phẩm chính của bản này.

## Giới hạn

1. Dữ liệu này vẫn là mô phỏng vật lý để kiểm thử quy trình, không được gọi là dữ liệu phần cứng.
2. Khi có dữ liệu thật, phải retrain và báo cáo lại trên test thật; không được lấy chỉ số mô phỏng làm kết quả thực nghiệm cuối.
3. Các biến `Temperature_Air_True`, `Humidity_Air_True`, `Air_Exchange_Rate` chỉ là trạng thái ẩn trong mô phỏng để audit; model ARX final không dùng các cột này.

## Chạy lại

```powershell
python -B .\ARX_PBL5_Final_Clean\src\arx_final_pipeline.py
```
