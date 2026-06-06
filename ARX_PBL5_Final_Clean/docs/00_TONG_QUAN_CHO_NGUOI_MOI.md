# Tổng quan cho người mới

## 1. Bài toán đang giải quyết

Mô hình nhà kính nhỏ có kích thước:

```text
30 cm x 50 cm x 30 cm
```

Thể tích xấp xỉ:

```text
0.30 * 0.50 * 0.30 = 0.045 m3
```

Mục tiêu của project là xây dựng mô hình dự đoán độ ẩm đất để phục vụ điều khiển tưới. Đầu ra cần dự đoán là:

```text
Soil_Moisture
```

Các tín hiệu đầu vào chính:

- `Temperature_In`: nhiệt độ trong mô hình;
- `Humidity_In`: độ ẩm không khí trong mô hình;
- `Light_In`: ánh sáng trong mô hình;
- `Drip`: trạng thái bơm nhỏ giọt;
- `Mist`: trạng thái phun sương;
- `Fan`: trạng thái quạt;
- các đặc trưng phụ được tính từ những tín hiệu trên.

## 2. Vì sao không điều khiển bằng luật đơn giản là đủ?

Luật đơn giản kiểu:

```text
Nếu đất khô thì bật bơm.
Nếu đất đủ ẩm thì tắt bơm.
```

có thể chạy được, nhưng có nhược điểm:

- không dự đoán trước đất sẽ ẩm lên hay khô đi;
- dễ tưới quá tay vì nước cần thời gian thấm đến cảm biến;
- khó tối ưu lượng nước;
- khó phối hợp bơm, quạt, phun sương.

Mô hình ARX giúp trả lời câu hỏi:

```text
Với trạng thái hiện tại và lịch actuator vừa qua, độ ẩm đất sắp tới sẽ đi theo hướng nào?
```

## 3. Vì sao lấy mẫu 20 giây?

Mô hình nhỏ phản ứng nhanh hơn nhà kính thật. Nếu lấy mẫu 1 phút, nhiều tác động ngắn của bơm và quạt có thể bị bỏ qua.

Chu kỳ `20 giây/mẫu` có lợi:

- thấy được tác động ngắn của actuator;
- vẫn đủ nhẹ cho vi điều khiển hoặc máy tính lưu CSV;
- dễ quy đổi horizon điều khiển:

```text
12 bước = 4 phút
60 bước = 20 phút
```

## 4. Sản phẩm chính là gì?

Sản phẩm chính hiện tại:

```text
ARX thuần: ARX_na12_nb3_nk2_alpha0.1
```

Kết quả trên dữ liệu mô phỏng vật lý có kiểm soát:

| Chỉ số | Giá trị |
|---|---:|
| FIT_1step | 95.373 |
| FIT_12 | 90.839 |
| FIT_60 | 88.047 |
| FIT_sim | 78.950 |
| RMSE_sim | 0.2021 |

Ý nghĩa:

- `FIT_1step` cao vì dự đoán từng bước ngắn dễ hơn;
- `FIT_60` là horizon 20 phút, gần bài toán MPC;
- `FIT_sim` là free-run dài, khó hơn và trung thực hơn về khả năng mô phỏng.

## 5. Thử nghiệm mở rộng là gì?

Ngoài ARX thuần, có thử lại:

```text
Hybrid ARX residual
```

Công thức:

```text
y_hybrid = y_arx + shrink * residual_model(features)
```

Kết quả:

| Model | FIT_sim |
|---|---:|
| ARX thuần 17 input | 78.950 |
| Hybrid ARX residual | 81.102 |

Điểm quan trọng: Hybrid residual có tăng fit, nhưng không còn là ARX thuần 100%. Vì vậy nên trình bày nó như thử nghiệm mở rộng.

## 6. Dữ liệu hiện tại có phải dữ liệu thật không?

Không. Dữ liệu hiện tại là:

```text
dữ liệu mô phỏng vật lý có kiểm soát
```

Nó dùng để:

- kiểm tra pipeline;
- thiết kế quy trình thu dữ liệu thật;
- kiểm tra logic không leakage;
- chuẩn bị báo cáo PBL5.

Khi có dữ liệu phần cứng, phải chạy lại:

```powershell
python -B .\ARX_PBL5_Final_Clean\src\arx_final_pipeline.py --data-csv .\data\log_that.csv
```

## 7. Câu nói cốt lõi để bảo vệ

```text
Nhóm xây dựng pipeline nhận dạng ARX cho mô hình nhà kính nhỏ. Bản ARX thuần 17 input đạt FIT_sim 78.95 trên dữ liệu mô phỏng vật lý 20 giây/mẫu và phù hợp để tích hợp MPC tuyến tính. Nhóm cũng thử Hybrid ARX residual, đạt 81.10 nhưng xem là hướng mở rộng vì không còn là ARX thuần. Khi có dữ liệu phần cứng thật, nhóm sẽ train/test lại bằng cùng pipeline không leakage.
```
