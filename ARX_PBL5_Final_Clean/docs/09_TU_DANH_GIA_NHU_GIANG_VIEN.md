# Tự đánh giá như giảng viên

## 1. Vai trò đánh giá

Phần này tự soi đồ án như một giảng viên khó tính. Mục tiêu không phải tự khen, mà là tìm chỗ dễ bị hỏi để sửa trước.

## 2. Rubric đề xuất

| Tiêu chí | Điểm tối đa | Đánh giá hiện tại | Nhận xét |
|---|---:|---:|---|
| Phát biểu bài toán | 10 | 8 | Mục tiêu rõ, cần nhấn mạnh dữ liệu hiện tại là mô phỏng |
| Cơ sở lý thuyết ARX | 15 | 12 | Có công thức, cần giải thích thêm bằng lời khi trình bày |
| Quy trình dữ liệu | 20 | 15 | Pipeline sạch, nhưng dữ liệu thật chưa có |
| Tránh leakage | 15 | 13 | Split theo thời gian, scaler train-only, feature/residual chọn bằng validation |
| Kết quả và phân tích | 20 | 16 | ARX gọn làm chính, residual là mở rộng |
| Khả năng áp dụng thực tế | 10 | 7 | Có `--data-csv`, cần thu dữ liệu phần cứng để hoàn thiện |
| Trình bày bảo vệ | 10 | 8 | Docs đã có, cần luyện trả lời ngắn |

Tổng tự đánh giá: khoảng `79/100` nếu chỉ có dữ liệu mô phỏng. Nếu thu được dữ liệu thật và chạy lại pipeline, điểm có thể tăng đáng kể.

## 3. Điểm mạnh

- Có pipeline chạy lại được, không chỉ là notebook rời.
- Có chia train/validation/test theo thời gian.
- Có ARX thuần và Hybrid ARX residual tách riêng.
- Có `shrink=0` trong residual, nghĩa là residual không bị ép phải sửa.
- Không báo candidate test cao hơn làm model final nếu validation không chọn.
- Có hỗ trợ CSV thật qua `--data-csv`.

## 4. Điểm yếu dễ bị hỏi

### 4.1. Dữ liệu hiện tại chưa phải dữ liệu thật

Câu hỏi có thể gặp:

```text
Em nói áp dụng thực tế nhưng dữ liệu lấy từ đâu?
```

Câu trả lời:

```text
Hiện tại nhóm dùng dữ liệu mô phỏng vật lý có kiểm soát để kiểm tra pipeline và thiết kế thí nghiệm. Pipeline đã hỗ trợ CSV thật. Khi triển khai phần cứng, nhóm sẽ thu dữ liệu theo lịch kích thích an toàn, giữ test riêng và train lại bằng cùng quy trình.
```

### 4.2. ARX final và Hybrid residual khác nhau về bản chất

Câu hỏi:

```text
Nếu thêm residual thì còn là ARX không?
```

Câu trả lời:

```text
Backbone vẫn là ARX, nhưng model cuối có thêm tầng sửa sai số nên gọi chính xác là Hybrid ARX residual. Vì vậy nhóm tách ARX thuần 17 input làm sản phẩm chính và residual làm hướng mở rộng.
```

### 4.3. Vì sao không báo FIT cao hơn trong residual leaderboard?

Câu trả lời:

```text
Vì những candidate đó không đứng đầu theo validation robust score. Nếu chọn theo test thì là leakage lựa chọn model. Nhóm chỉ báo model residual được validation chọn, FIT_sim test 81.10.
```

### 4.4. Vì sao không so 66 với 78.95?

Câu trả lời:

```text
Vì đó là hai dataset/protocol khác nhau. Trên data cũ, residual cải thiện 66 lên khoảng 69-71. Trên protocol mới, ARX gọn đạt 78.95 và Hybrid đạt 81.10.
```

## 5. Việc cần làm nếu muốn đạt chuẩn cao hơn

1. Thu dữ liệu phần cứng thật tối thiểu 1 đến 2 ngày.
2. Chạy `arx_final_pipeline.py --data-csv`.
3. Lưu riêng kết quả thật vào một folder mới, ví dụ `results_real_hw`.
4. So sánh ARX thuần và Hybrid residual trên cùng CSV thật.
5. Nếu Hybrid residual không cải thiện trên dữ liệu thật, giữ ARX thuần.
6. Bổ sung hình plot `y_true` và `y_pred` trong báo cáo.

## 6. Kết luận giảng viên giả định

Nếu chỉ nộp hiện tại, đánh giá hợp lý là:

```text
Tốt về phương pháp và tính trung thực, nhưng cần dữ liệu phần cứng thật để kết luận thực nghiệm.
```

Nếu có thêm dữ liệu thật:

```text
Đủ chuẩn PBL5 tốt: có nhận dạng hệ thống, đánh giá không leakage, có so sánh model và có hướng tích hợp điều khiển.
```
