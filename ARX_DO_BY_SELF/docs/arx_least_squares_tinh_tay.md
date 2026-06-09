# Ví dụ tính tay thuật toán ARX Least Squares/Ridge

File code chính: `src/algorithm/arx.py`

Mục tiêu là đi từ dữ liệu nhỏ đến vector hệ số:

```text
theta = [a1, b1, bias]
```

Ví dụ dùng mô hình rất nhỏ:

```text
na = 1
nb = 1
nk = 1
alpha = 0
input_cols = ["u"]
```

Vì `alpha = 0`, bài toán là Least Squares thường:

```text
theta = (X^T X)^(-1) X^T y
```

Trong code, công thức này được giải bằng:

```python
np.linalg.solve(lhs, rhs)
```

tức là giải hệ:

```text
lhs * theta = rhs
```

## 1. Dữ liệu ví dụ

Giả sử có 5 mẫu dữ liệu:

| k | Soil_Moisture y(k) | input u(k) |
|---|---:|---:|
| 0 | 1 | 0 |
| 1 | 2 | 1 |
| 2 | 4 | 0 |
| 3 | 7 | 2 |
| 4 | 11 | 1 |

Tương ứng trong code:

```python
y = df_z["Soil_Moisture"].to_numpy(dtype=float)
```

Ở `arx.py`, dòng 25.

Với ví dụ này:

```text
y = [1, 2, 4, 7, 11]
u = [0, 1, 0, 2, 1]
```

## 2. Tính lag bắt đầu

Code:

```python
lag = max_lag(spec)
```

Ở `arx.py`, dòng 26.

Với:

```text
na = 1
nb = 1
nk = 1
```

thì:

```text
lag = max(na, nb + nk - 1)
    = max(1, 1 + 1 - 1)
    = 1
```

Nghĩa là bắt đầu tạo mẫu train từ `k = 1`, vì tại `k = 0` chưa có `y(k-1)` và `u(k-1)`.

## 3. Dựng cột y quá khứ

Code:

```python
for y_lag in range(1, spec.na + 1):
    cols.append(y[lag - y_lag : len(y) - y_lag])
```

Ở `arx.py`, dòng 30-31.

Vì `na = 1`, chỉ lấy:

```text
y(k-1)
```

Tính ra:

| k dự đoán | y(k-1) |
|---:|---:|
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 4 | 7 |

Cột đầu tiên của `X` là:

```text
[1, 2, 4, 7]
```

## 4. Dựng cột input quá khứ

Code:

```python
for col in input_cols:
    values = df_z[col].to_numpy(dtype=float)
    for u_lag in range(spec.nk, spec.nk + spec.nb):
        cols.append(values[lag - u_lag : len(values) - u_lag])
```

Ở `arx.py`, dòng 34-37.

Vì `nk = 1`, `nb = 1`, chỉ lấy:

```text
u(k-1)
```

Tính ra:

| k dự đoán | u(k-1) |
|---:|---:|
| 1 | 0 |
| 2 | 1 |
| 3 | 0 |
| 4 | 2 |

Cột input của `X` là:

```text
[0, 1, 0, 2]
```

## 5. Thêm bias

Code:

```python
cols.append(np.ones(len(y) - lag))
```

Ở `arx.py`, dòng 40.

Bias là cột toàn số 1:

```text
[1, 1, 1, 1]
```

## 6. Ghép thành ma trận X và vector Y

Code:

```python
return np.vstack(cols).T, y[lag:]
```

Ở `arx.py`, dòng 43.

Ma trận `X` gồm 3 cột:

```text
X = [ y(k-1), u(k-1), bias ]
```

Cụ thể:

```text
X =
[
  [1, 0, 1],
  [2, 1, 1],
  [4, 0, 1],
  [7, 2, 1],
]
```

Vector output thật:

```text
Y = y[lag:] = [2, 4, 7, 11]
```

Đến đây bài toán ARX đã thành:

```text
Y = X * theta
```

tức:

```text
[
  2,
  4,
  7,
  11
]
=
[
  [1, 0, 1],
  [2, 1, 1],
  [4, 0, 1],
  [7, 2, 1],
]
*
[
  a1,
  b1,
  bias
]
```

## 7. Tính X^T X

Code:

```python
lhs = x_train.T @ x_train + spec.alpha * penalty
```

Ở `arx.py`, dòng 58.

Vì `alpha = 0`, phần `spec.alpha * penalty = 0`.

Nên:

```text
lhs = X^T X
```

Tính ra:

```text
X^T X =
[
  [70, 16, 14],
  [16,  5,  3],
  [14,  3,  4],
]
```

## 8. Tính X^T Y

Code:

```python
rhs = x_train.T @ y_train
```

Ở `arx.py`, dòng 59.

Tính ra:

```text
X^T Y = [115, 26, 24]
```

## 9. Giải hệ để ra theta

Code:

```python
return np.linalg.solve(lhs, rhs)
```

Ở `arx.py`, dòng 62.

Tức giải hệ:

```text
[
  [70, 16, 14],
  [16,  5,  3],
  [14,  3,  4],
]
*
[
  a1,
  b1,
  bias
]
=
[
  115,
  26,
  24
]
```

Kết quả:

```text
theta = [1.5, -0.09090909, 0.81818182]
```

Nghĩa là:

```text
a1   = 1.5
b1   = -0.09090909
bias = 0.81818182
```

Vậy công thức ARX fit được là:

```text
y_hat(k) = 1.5*y(k-1) - 0.09090909*u(k-1) + 0.81818182
```

## 10. Kiểm tra dự đoán bằng theta

Lấy từng dòng của `X` nhân với `theta`:

```text
y_hat = X * theta
```

Tính ra:

| k | y thật | y dự đoán |
|---:|---:|---:|
| 1 | 2 | 2.31818 |
| 2 | 4 | 3.72727 |
| 3 | 7 | 6.81818 |
| 4 | 11 | 11.13636 |

Mô hình không khớp tuyệt đối vì có nhiều điểm hơn số hệ số, nên Least Squares chọn `theta` sao cho tổng bình phương sai số nhỏ nhất.

## 11. Nếu dùng alpha = 10 như model hiện tại

Model hiện tại của project là:

```text
ARX_na96_nb16_nk2_alpha10
```

nên:

```text
alpha = 10
```

Trong code, `alpha` nằm ở dòng:

```python
lhs = x_train.T @ x_train + spec.alpha * penalty
```

Ở `arx.py`, dòng 58.

Với ví dụ nhỏ ở trên, ma trận `penalty` là:

```text
penalty =
[
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 0],
]
```

`penalty` là ma trận dùng để chỉ định hệ số nào bị phạt trong Ridge.

Ban đầu code tạo ma trận đơn vị:

```python
penalty = np.eye(x_train.shape[1], dtype=float)
```

Ở `arx.py`, dòng 53.

Nếu `theta = [a1, b1, bias]`, ma trận đơn vị ban đầu là:

```text
[
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
]
```

Sau đó bias không bị phạt nên phần tử cuối được đổi thành `0`.

Dòng cuối là `0` vì bias không bị phạt:

```python
penalty[-1, -1] = 0.0
```

Ở `arx.py`, dòng 56.

Khi `alpha = 10`:

```text
lhs = X^T X + 10 * penalty
```

Từ:

```text
X^T X =
[
  [70, 16, 14],
  [16,  5,  3],
  [14,  3,  4],
]
```

suy ra:

```text
lhs =
[
  [80, 16, 14],
  [16, 15,  3],
  [14,  3,  4],
]
```

Nói ngắn gọn:

```text
alpha = mức phạt
penalty = ma trận cho biết hệ số nào bị phạt
alpha * penalty = phần cộng thêm vào X^T X
```

Với ví dụ này:

```text
10 * penalty =
[
  [10, 0, 0],
  [0, 10, 0],
  [0, 0, 0],
]
```

Nên chỉ `a1` và `b1` bị phạt, còn `bias` không bị phạt.

Vế phải vẫn là:

```text
rhs = X^T Y = [115, 26, 24]
```

Giải hệ:

```text
lhs * theta = rhs
```

Kết quả:

```text
theta = [0.96232877, 0.21232877, 2.47260274]
```

Nghĩa là với `alpha = 10`:

```text
a1   = 0.96232877
b1   = 0.21232877
bias = 2.47260274
```

So với `alpha = 0`, hệ số `a1`, `b1` bị kéo nhỏ lại do Ridge regularization. Bias không bị phạt nên vẫn được học tự do.

## 12. Tóm tắt dòng code tương ứng

| Bước | Ý nghĩa | Dòng code trong `arx.py` |
|---|---|---|
| 1 | Lấy chuỗi `y` | dòng 25 |
| 2 | Tính `lag` | dòng 26 |
| 3 | Tạo cột `y(k-1)...y(k-na)` | dòng 30-31 |
| 4 | Tạo cột `u(k-nk)...u(k-nk-nb+1)` | dòng 34-37 |
| 5 | Thêm bias | dòng 40 |
| 6 | Ghép thành `X`, lấy `Y` | dòng 43 |
| 7 | Gọi build matrix trong fit | dòng 48 |
| 8 | Tạo ma trận đơn vị Ridge | dòng 53 |
| 9 | Tính `X^T X + alpha I` | dòng 58 |
| 10 | Tính `X^T Y` | dòng 59 |
| 11 | Giải ra `theta` | dòng 62 |

Trong ví dụ Least Squares, `alpha = 0`, nên dòng 58 chính là `X^T X`. Với model hiện tại, `alpha = 10`, nên dòng 58 là `X^T X + 10 * penalty`.
