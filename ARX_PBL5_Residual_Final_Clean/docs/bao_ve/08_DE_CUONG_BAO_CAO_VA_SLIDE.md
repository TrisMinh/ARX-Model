# Đề cương báo cáo và slide bảo vệ

## 1. Cấu trúc báo cáo

### Chương 1. Giới thiệu

- mô hình nhà kính nhỏ `30x50x30 cm`;
- mục tiêu dự đoán độ ẩm đất;
- lý do cần mô hình dự báo nhiều bước;
- phạm vi: dữ liệu hiện tại là mô phỏng vật lý có kiểm soát.

### Chương 2. Cơ sở lý thuyết

- ARX;
- residual;
- Hybrid ARX residual correction;
- chỉ số FIT;
- one-step, multi-step và free-run simulation.

### Chương 3. Dữ liệu và quy trình

- schema dữ liệu;
- sampling 20 giây;
- chia train/validation/test theo thời gian;
- nguyên tắc không leakage;
- validation robust score.

### Chương 4. Kết quả

- ARX backbone `82.496`;
- Hybrid residual `83.433`;
- model chọn `hgb_leaf15_l2_0.01 | shrink=0.25`;
- giải thích vì sao không chọn candidate test cao hơn.

### Chương 5. Tự phản biện và hướng triển khai

- dữ liệu chưa phải phần cứng thật;
- model không còn ARX thuần;
- ảnh hưởng đến MPC;
- kế hoạch thu dữ liệu thật và chạy lại pipeline.

## 2. Slide 10 trang

1. Tên đề tài và mô hình nhà kính.
2. Bài toán dự đoán độ ẩm đất.
3. Vì sao cần dự báo nhiều bước.
4. ARX backbone.
5. Residual correction.
6. Quy trình train/validation/test không leakage.
7. Kết quả ARX backbone.
8. Kết quả Hybrid residual.
9. Tự phản biện: dữ liệu mô phỏng, MPC, không chọn theo test.
10. Kết luận và hướng dữ liệu thật.

## 3. Câu ghi trên slide kết quả

```text
ARX backbone FIT_sim = 82.50.
Hybrid ARX residual FIT_sim = 83.43.
Residual được chọn bằng validation robust score, không chọn bằng test.
```

## 4. Câu kết luận cuối

```text
Hybrid ARX residual correction là lựa chọn chính vì vẫn giữ ARX làm backbone nhưng cải thiện free-run simulation. Kết quả hiện tại là trên dữ liệu mô phỏng vật lý; khi có dữ liệu phần cứng thật, nhóm sẽ train/test lại bằng cùng pipeline.
```
