# Ví dụ tính toán từng bước: từ input đến kết quả cuối

File này dùng một ví dụ nhỏ để nhìn rõ thuật toán đang làm gì. Ví dụ cố ý rất nhỏ để có thể viết ma trận ra bằng tay. Code thật dùng nhiều input hơn, nhiều dòng hơn và có chuẩn hóa dữ liệu, nhưng logic cốt lõi giống hệt.

## 1. Bài toán ví dụ

Giả sử chỉ dùng một input là `Drip`.

```text
y(k) = Soil_Moisture tại mẫu k
u(k) = Drip tại mẫu k
T_s  = 20 giây
```

Chọn mô hình:

```text
ARX(na=2, nb=2, nk=1)
```

Nghĩa là:

- dùng 2 giá trị độ ẩm đất quá khứ: `y(k-1)`, `y(k-2)`;
- dùng 2 giá trị input quá khứ: `u(k-1)`, `u(k-2)`;
- input có trễ tối thiểu 1 mẫu, tức 20 giây.

Phương trình:

```text
y_hat(k) =
    a1*y(k-1)
  + a2*y(k-2)
  + b1*u(k-1)
  + b2*u(k-2)
  + c
```

Vector hệ số cần học:

```text
theta = [a1, a2, b1, b2, c]^T
```

## 2. Bảng dữ liệu thô

Ví dụ có 12 mẫu:

| k | thời gian | Soil_Moisture y(k) | Drip u(k) |
| ---: | --- | ---: | ---: |
| 0 | 00:00:00 | 55.0000 | 0 |
| 1 | 00:00:20 | 55.0500 | 0 |
| 2 | 00:00:40 | 55.0400 | 1 |
| 3 | 00:01:00 | 55.3395 | 1 |
| 4 | 00:01:20 | 55.6776 | 0 |
| 5 | 00:01:40 | 55.6930 | 0 |
| 6 | 00:02:00 | 55.6560 | 0 |
| 7 | 00:02:20 | 55.6288 | 1 |
| 8 | 00:02:40 | 55.9014 | 1 |
| 9 | 00:03:00 | 56.2155 | 0 |
| 10 | 00:03:20 | 56.2076 | 0 |
| 11 | 00:03:40 | 56.1484 | 0 |

Mẫu `k=0` và `k=1` chưa đủ quá khứ để train vì mô hình cần `y(k-1)` và `y(k-2)`. Vì vậy dòng train đầu tiên bắt đầu từ `k=2`.

## 3. Tạo một dòng hồi quy

Tại `k=4`, ta muốn dự đoán:

```text
y(4) = 55.6776
```

Các giá trị quá khứ:

```text
y(k-1) = y(3) = 55.3395
y(k-2) = y(2) = 55.0400
u(k-1) = u(3) = 1
u(k-2) = u(2) = 1
bias   = 1
```

Vậy vector hồi quy tại `k=4` là:

```text
phi(4) = [55.3395, 55.0400, 1, 1, 1]
```

Target tương ứng:

```text
Y(4) = 55.6776
```

## 4. Lập ma trận hồi quy

Với tất cả dòng từ `k=2` đến `k=11`, ta có ma trận:

```text
Phi =
[
  [55.0500, 55.0000, 0, 0, 1],
  [55.0400, 55.0500, 1, 0, 1],
  [55.3395, 55.0400, 1, 1, 1],
  [55.6776, 55.3395, 0, 1, 1],
  [55.6930, 55.6776, 0, 0, 1],
  [55.6560, 55.6930, 0, 0, 1],
  [55.6288, 55.6560, 1, 0, 1],
  [55.9014, 55.6288, 1, 1, 1],
  [56.2155, 55.9014, 0, 1, 1],
  [56.2076, 56.2155, 0, 0, 1]
]
```

Vector target:

```text
Y =
[
  55.0400,
  55.3395,
  55.6776,
  55.6930,
  55.6560,
  55.6288,
  55.9014,
  56.2155,
  56.2076,
  56.1484
]^T
```

Mục tiêu train là tìm `theta` sao cho:

```text
Phi * theta ≈ Y
```

## 5. Tính hệ số theta

Nếu dùng least squares:

```text
theta = (Phi^T * Phi)^(-1) * Phi^T * Y
```

Với ví dụ này, nghiệm là:

```text
theta =
[
  0.8000,
  0.1500,
  0.3000,
  0.1000,
  2.7500
]^T
```

Tức là:

```text
a1 = 0.8000
a2 = 0.1500
b1 = 0.3000
b2 = 0.1000
c  = 2.7500
```

Phương trình sau khi train:

```text
y_hat(k) =
    0.8000*y(k-1)
  + 0.1500*y(k-2)
  + 0.3000*u(k-1)
  + 0.1000*u(k-2)
  + 2.7500
```

Trong code thật, nếu `alpha > 0`, project dùng Ridge:

```text
theta = (Phi^T * Phi + alpha * I)^(-1) * Phi^T * Y
```

Ridge giúp hệ số ổn định hơn khi có nhiều input, nhiều feature và nhiễu cảm biến.

## 6. Tính thử một dự đoán

Tại `k=4`:

```text
y_hat(4)
= 0.8000*y(3)
 + 0.1500*y(2)
 + 0.3000*u(3)
 + 0.1000*u(2)
 + 2.7500
```

Thay số:

```text
y_hat(4)
= 0.8000*55.3395
 + 0.1500*55.0400
 + 0.3000*1
 + 0.1000*1
 + 2.7500
```

Tính:

```text
0.8000*55.3395 = 44.2716
0.1500*55.0400 = 8.2560
0.3000*1       = 0.3000
0.1000*1       = 0.1000
bias           = 2.7500
```

Cộng lại:

```text
y_hat(4) = 55.6776
```

Khớp với giá trị thật trong ví dụ:

```text
y(4) = 55.6776
```

## 7. Dự đoán mẫu tiếp theo

Giả sử muốn dự đoán `k=12`.

Ta có:

```text
y(11) = 56.1484
y(10) = 56.2076
u(11) = 0
u(10) = 0
```

Tính:

```text
y_hat(12)
= 0.8000*56.1484
 + 0.1500*56.2076
 + 0.3000*0
 + 0.1000*0
 + 2.7500
```

Kết quả:

```text
y_hat(12) ≈ 56.0999
```

Đây là cách mô hình đi từ input quá khứ đến dự đoán output.

## 8. One-step khác free-run như thế nào?

### One-step

Khi dự đoán `y(k)`, mô hình dùng `y` thật trong quá khứ:

```text
y_hat(k) = f(y_true(k-1), y_true(k-2), u quá khứ)
```

### Free-run simulation

Sau vài mẫu khởi tạo, mô hình dùng chính dự đoán của nó:

```text
y_sim(k) = f(y_sim(k-1), y_sim(k-2), u quá khứ)
```

Free-run khó hơn vì nếu dự đoán lệch một chút, sai số có thể tích lũy theo thời gian.

Trong code thật:

- `predict_arx_one_step` dùng one-step;
- `simulate_arx` dùng free-run simulation;
- `simulate_arx_n_step` dùng multi-step theo horizon.

## 9. Tính metric bằng ví dụ nhỏ

Giả sử trên một đoạn test có 5 điểm:

```text
y_true = [55.00, 55.40, 55.90, 56.10, 56.00]
y_arx  = [55.08, 55.35, 55.84, 56.03, 56.04]
```

Sai số:

```text
e = y_true - y_arx
  = [-0.08, 0.05, 0.06, 0.07, -0.04]
```

### RMSE

```text
RMSE = sqrt(mean(e^2))
     = sqrt((0.0064 + 0.0025 + 0.0036 + 0.0049 + 0.0016) / 5)
     = 0.0616
```

### Bias

```text
Bias = mean(e)
     = (-0.08 + 0.05 + 0.06 + 0.07 - 0.04) / 5
     = 0.0120
```

Bias dương nghĩa là trung bình mô hình dự đoán hơi thấp hơn thực tế.

### FIT

Trung bình của `y_true`:

```text
mean(y_true) = 55.68
```

Norm sai số:

```text
||y_true - y_arx|| = sqrt(0.0190) = 0.1378
```

Norm độ biến thiên của tín hiệu thật:

```text
||y_true - mean(y_true)|| = sqrt(0.8680) = 0.9317
```

FIT:

```text
FIT = 100 * (1 - 0.1378 / 0.9317)
    = 85.205%
```

## 10. Residual correction hoạt động như thế nào?

Sau ARX, ta còn sai số:

```text
residual = y_true - y_arx
```

Hybrid residual không bỏ ARX. Nó giữ ARX làm backbone, rồi học thêm phần sai số còn lại.

Công thức:

```text
y_hybrid = y_arx + shrink * r_hat
```

Trong đó:

- `y_arx`: dự đoán từ ARX;
- `r_hat`: residual model dự đoán phần sai số còn lại;
- `shrink`: hệ số giảm cường độ hiệu chỉnh để tránh sửa quá tay.

Giả sử residual model dự đoán:

```text
r_hat = [-0.10, 0.04, 0.05, 0.08, -0.02]
```

Chọn:

```text
shrink = 0.5
```

Phần hiệu chỉnh thật sự được cộng vào:

```text
shrink * r_hat = [-0.05, 0.02, 0.025, 0.04, -0.01]
```

Dự đoán hybrid:

```text
y_hybrid
= y_arx + shrink*r_hat
= [55.03, 55.37, 55.865, 56.07, 56.03]
```

Sai số mới:

```text
e_hybrid = y_true - y_hybrid
         = [-0.03, 0.03, 0.035, 0.03, -0.03]
```

Metric mới:

```text
RMSE_hybrid = 0.0311
FIT_hybrid  = 92.544%
```

So với ARX:

```text
RMSE_arx = 0.0616
FIT_arx  = 85.205%
```

Vậy residual correction giúp giảm sai số, nhưng cần nhớ: trong thực tế `r_hat` phải được dự đoán từ feature hợp lệ, không được lấy trực tiếp `y_true - y_arx` của test để cộng vào. Nếu lấy residual thật của test thì đó là leakage.

## 11. Liên hệ với project thật

Trong project thật, ARX không chỉ có một input `Drip`. Bản final dùng 16 input:

```text
Temperature_In
Humidity_In
Light_In
Drip
Mist
Fan
Light_log
TempIn_x_HumiIn
TempIn_x_Light
HumiIn_x_Light
Indoor_Dryness
VPD_Proxy_In
Hour_sin
Hour_cos
Day_sin
Day_cos
```

Nếu:

```text
na = 12
nb = 6
nk = 1
số input = 16
```

thì số hệ số ARX là:

```text
n_params = na + số_input*nb + 1
         = 12 + 16*6 + 1
         = 109
```

Con số này đúng với `metrics.json` của bản residual final:

```text
ARX_na12_nb6_nk1_alpha10
n_params = 109
```

Ma trận `Phi` trong project thật có dạng:

```text
Phi =
[
  y_lag_1, ..., y_lag_12,
  Temperature_In_lag_1, ..., Temperature_In_lag_6,
  Humidity_In_lag_1, ..., Humidity_In_lag_6,
  ...
  Day_cos_lag_1, ..., Day_cos_lag_6,
  bias
]
```

Vậy khác biệt giữa ví dụ nhỏ và project thật chỉ là:

- ví dụ nhỏ có 1 input, project thật có 16 input;
- ví dụ nhỏ không scale, project thật scale bằng mean/std của train;
- ví dụ nhỏ ít dòng, project thật có hàng chục nghìn dòng;
- ví dụ nhỏ tính để minh họa, project thật chọn model bằng validation robust score.

## 12. Tóm tắt luồng tính toán

Toàn bộ thuật toán đi theo luồng:

```text
CSV dữ liệu
-> kiểm tra timestamp, missing, duplicate
-> tạo feature
-> chia train/validation/test theo thời gian
-> scale bằng thống kê train
-> tạo ma trận Phi từ y quá khứ và u quá khứ
-> học theta bằng least squares/ridge
-> mô phỏng one-step, multi-step, free-run
-> tính FIT, RMSE, Bias
-> train residual model trên sai số ARX của train
-> chọn residual/shrink bằng validation
-> báo cáo test cuối cùng
```

Đây là đường đi từ input thô đến kết quả cuối cùng.
