# Lý thuyết Hybrid ARX residual

## 1. ARX backbone

ARX dự đoán đầu ra hiện tại bằng đầu ra quá khứ và input quá khứ:

```text
y(k) = a1*y(k-1) + ... + ana*y(k-na)
     + b1*u(k-nk) + ... + e(k)
```

Trong project này, ARX backbone là:

```text
ARX_na12_nb6_nk1_alpha10
```

Với sampling `20 giây/mẫu`:

- `na=12`: nhớ độ ẩm đất quá khứ 4 phút;
- `nb=6`: nhớ input 2 phút;
- `nk=1`: input trễ 20 giây.

## 2. Residual là gì?

Residual là sai số còn lại sau khi ARX dự đoán:

```text
residual(k) = y_true(k) - y_arx_sim(k)
```

Nếu residual còn có quy luật theo quá khứ, ta có thể học phần sai số đó bằng một model phụ.

## 3. Công thức model cuối

Model cuối:

```text
y_hybrid(k) = y_arx_sim(k) + shrink * residual_model(x_residual(k))
```

Trong đó `x_residual(k)` chỉ gồm thông tin hợp lệ tại thời điểm dự đoán:

- dự đoán ARX tại hiện tại;
- các lag của quỹ đạo ARX mô phỏng;
- các lag input quá khứ;
- không dùng `Soil_Moisture` thật tương lai.

## 4. Vì sao không leakage?

Pipeline giữ 4 nguyên tắc:

- chia train/validation/test theo thời gian `70/15/15`;
- scaler fit trên train;
- residual train trên train;
- chọn residual model và `shrink` bằng validation robust score;
- test chỉ dùng sau khi đã chọn xong.

Trong grid có `shrink=0`. Nghĩa là nếu residual không giúp, pipeline được phép quay về ARX backbone.

## 5. Vì sao model này vẫn hợp lý cho đồ án ARX?

Vì phần nền vẫn là ARX. Residual không thay thế ARX, mà chỉ sửa sai số còn lại:

```text
ARX backbone trước, residual correction sau.
```

Cách trình bày trung thực:

```text
Đề tài xuất phát từ ARX. Bản tốt nhất dùng ARX làm backbone và bổ sung residual correction để cải thiện mô phỏng nhiều bước.
```

Không trình bày thành:

```text
ARX thuần đạt 83.43.
```

Vì câu đó sai bản chất.
