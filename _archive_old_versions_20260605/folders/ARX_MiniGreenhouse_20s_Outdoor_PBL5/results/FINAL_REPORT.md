# Báo Cáo Cuối: ARX Có Cảm Biến Ngoài

## Kết luận ngắn

- Với mô hình nhỏ `30x50x30 cm`, quạt làm không khí trong hộp trao đổi với môi trường ngoài rất nhanh.
- Vì vậy `Temperature_Out` và `Humidity_Out` là biến nhiễu đo được, không nên bỏ qua nếu có thể lắp cảm biến.
- Kết quả dưới đây là kết quả chạy thật của pipeline trên dữ liệu mô phỏng vật lý có log rõ, không phải dữ liệu phần cứng.
- ARX chỉ dùng cảm biến trong đạt `FIT_sim = 78.950`.
- ARX có thêm cảm biến ngoài đạt `FIT_sim = 78.953`.
- Mức chênh free-run khi thêm cảm biến ngoài: `0.003` điểm FIT, tức là gần như ngang free-run với bản chỉ dùng cảm biến trong.
- Mức chênh horizon 20 phút khi thêm cảm biến ngoài: `0.790` điểm FIT.
- Nếu thu dữ liệu thật, vẫn nên log cả biến trong và biến ngoài. Sau khi có dữ liệu thật, chọn dùng hay bỏ sensor ngoài phải dựa trên validation/test, không dựa trên cảm tính.

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
| Temperature_Out range | 20.354 - 34.690 |
| Humidity_In range | 52.334 - 88.530 |
| Humidity_Out range | 52.758 - 89.615 |
| Mean abs temp delta out-in | 2.098 |
| Mean abs humi delta out-in | 3.300 |
| Air exchange fan OFF mean | 0.012 |
| Air exchange fan ON mean | 0.356 |
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
| ARX baseline cũ chỉ cảm biến trong | ARX tuyến tính | inside_only | 19 | `ARX_na5_nb1_nk2_alpha0` | 95.031 | 86.465 | 76.201 | -8.543 | 1.0420 | Order ARX(5,1,2) để so với cách cũ |
| ARX robust chỉ cảm biến trong | ARX tuyến tính | inside_only | 19 | `ARX_na12_nb3_nk2_alpha0.1` | 95.373 | 90.839 | 88.047 | 78.950 | 0.2021 | Không dùng Temperature_Out/Humidity_Out |
| ARX robust có cảm biến ngoài | ARX tuyến tính | outdoor | 23 | `ARX_na24_nb6_nk1_alpha1` | 95.556 | 92.140 | 88.837 | 78.953 | 0.2021 | Có thêm Temperature_Out, Humidity_Out, Outdoor_Dryness và VPD_Proxy_Out |

## Chi tiết ARX

- ARX chỉ cảm biến trong chọn `ARX_na12_nb3_nk2_alpha0.1`, robust validation score `75.270`.
- ARX có cảm biến ngoài chọn `ARX_na24_nb6_nk1_alpha1`, robust validation score `70.435`.
- ARX ngoài có input delay `20` giây, output memory `480` giây, input memory `120` giây.
- Residual free-run ARX ngoài max abs ACF lag 1..60: `0.968`.

## Kết luận cho đồ án

Nếu có điều kiện phần cứng, nên thêm một cảm biến nhiệt độ/độ ẩm ngoài hộp. Khi quạt bật, biến ngoài quyết định chiều và tốc độ trao đổi không khí, nên nó là nhiễu đo được của plant. Tuy nhiên, kết quả cuối phải được chọn theo validation/test: nếu sensor ngoài không cải thiện rõ thì vẫn có thể dùng ARX chỉ cảm biến trong để giữ mô hình gọn hơn.

## Giới hạn

1. Dữ liệu này vẫn là mô phỏng vật lý để kiểm thử quy trình, không được gọi là dữ liệu phần cứng.
2. Khi có dữ liệu thật, phải retrain và báo cáo lại trên test thật; không được lấy chỉ số mô phỏng làm kết quả thực nghiệm cuối.
3. Các biến `Temperature_Air_True`, `Humidity_Air_True`, `Air_Exchange_Rate` chỉ là trạng thái ẩn trong mô phỏng để audit; khi train model chỉ dùng các cột input đã khai báo.
4. Nếu thêm cảm biến ngoài mà dữ liệu thật cho thấy không cải thiện validation/test, vẫn phải báo cáo trung thực và giữ mô hình đơn giản hơn.

## Chạy lại

```powershell
python -B .\ARX_MiniGreenhouse_20s_Outdoor_PBL5\src\outdoor_pipeline.py
```
