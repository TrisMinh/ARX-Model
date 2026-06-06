# Đề cương báo cáo và slide bảo vệ

## 1. Cấu trúc báo cáo đề xuất

### Chương 1. Giới thiệu

Nội dung cần có:

- lý do cần dự đoán độ ẩm đất;
- mô hình nhà kính nhỏ `30x50x30 cm`;
- mục tiêu: xây dựng model ARX dự đoán độ ẩm đất để hỗ trợ điều khiển;
- phạm vi: dữ liệu hiện tại là mô phỏng vật lý có kiểm soát, pipeline sẵn sàng nhận dữ liệu thật.

### Chương 2. Cơ sở lý thuyết

Nội dung:

- khái niệm time-series và hệ động học rời rạc;
- mô hình ARX;
- ý nghĩa `na`, `nb`, `nk`;
- chỉ số `FIT`;
- khác nhau giữa one-step, multi-step và free-run simulation.

### Chương 3. Dữ liệu và quy trình nhận dạng

Nội dung:

- schema dữ liệu;
- chu kỳ lấy mẫu 20 giây;
- chia train/validation/test theo thời gian;
- scaler fit trên train;
- validation robust score;
- nguyên tắc không leakage.

### Chương 4. Kết quả ARX

Nội dung:

- ARX final `ARX_na12_nb3_nk2_alpha0.1`;
- `FIT_sim = 78.950`;
- `FIT_60 = 88.047`;
- giải thích vì sao `na=12`, `nb=3`, `nk=2` hợp lý theo thời gian vật lý;
- tự phản biện về dữ liệu mô phỏng.

### Chương 5. Thử nghiệm mở rộng

Nội dung:

- Hybrid ARX residual;
- công thức `y_hybrid = y_arx + shrink * residual`;
- chọn bằng validation, không chọn bằng test;
- kết quả `81.102`;
- vì sao không gọi là ARX thuần.

### Chương 6. Kết luận và hướng phát triển

Nội dung:

- ARX thuần là sản phẩm chính;
- Hybrid residual là hướng mở rộng;
- cần thu dữ liệu phần cứng thật;
- có thể tích hợp ARX vào MPC tuyến tính.

## 2. Cấu trúc slide 10-12 trang

1. Tên đề tài và mục tiêu.
2. Mô hình nhà kính nhỏ và tín hiệu đo.
3. Vì sao cần model dự đoán.
4. Lý thuyết ARX ngắn gọn.
5. Quy trình dữ liệu và split.
6. Cấu trúc pipeline không leakage.
7. Kết quả ARX final.
8. Thử nghiệm Hybrid ARX residual.
9. So sánh ARX, Hybrid, NARX/NNARX.
10. Tự phản biện và giới hạn.
11. Hướng triển khai dữ liệu thật.
12. Kết luận.

## 3. Slide kết quả nên viết thế nào?

Nên viết:

```text
ARX final:
- Model: ARX_na12_nb3_nk2_alpha0.1
- FIT_sim: 78.95%
- FIT_60: 88.05%
- Chọn bằng validation robust score
- Test không dùng để chọn model
```

Không nên viết:

```text
Đã tăng từ 66 lên 78.95
```

Nếu muốn nhắc lịch sử:

```text
Trên dữ liệu cũ, residual correction cải thiện ARX từ khoảng 66 lên 69-71. Trên protocol dữ liệu mới, ARX gọn đạt 78.95 và Hybrid residual đạt 81.10. Không so trực tiếp 66 với các số protocol mới vì khác dataset/protocol.
```

## 4. Câu kết luận cuối slide

```text
Trong phạm vi PBL5, ARX thuần 17 input là lựa chọn hợp lý vì dễ giải thích, dễ kiểm tra và phù hợp MPC tuyến tính. Hybrid residual cho thấy còn hướng nâng cao, nhưng bản gọn vẫn là sản phẩm chính. Bước tiếp theo là thu dữ liệu phần cứng thật và chạy lại cùng pipeline không leakage.
```
