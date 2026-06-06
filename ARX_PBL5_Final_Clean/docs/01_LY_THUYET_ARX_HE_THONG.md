# Lý thuyết ARX và nhận dạng hệ thống

## 1. Hệ động học rời rạc

Độ ẩm đất không thay đổi tức thời. Nó phụ thuộc vào:

- độ ẩm đất trước đó;
- nước vừa tưới;
- quạt/phun sương;
- nhiệt độ, độ ẩm không khí;
- ánh sáng và bay hơi;
- độ trễ cảm biến.

Vì dữ liệu được lấy mẫu theo thời gian, ta dùng chỉ số rời rạc:

```text
k = 0, 1, 2, ...
Ts = 20 giây
```

`y(k)` là độ ẩm đất tại thời điểm `k`.

`u(k)` là vector đầu vào tại thời điểm `k`.

## 2. Công thức ARX

ARX có dạng:

```text
y(k) = c
     + a1*y(k-1) + ... + a_na*y(k-na)
     + b11*u1(k-nk) + ... + b1nb*u1(k-nk-nb+1)
     + ...
     + e(k)
```

Trong đó:

- `y(k)`: output cần dự đoán;
- `u(k)`: input đo được;
- `na`: số mẫu quá khứ của output;
- `nb`: số mẫu quá khứ của mỗi input;
- `nk`: độ trễ input;
- `c`: hệ số chặn;
- `e(k)`: sai số còn lại.

## 3. Ý nghĩa model final

Model final:

```text
ARX_na12_nb3_nk2_alpha0.1
```

Quy đổi theo `Ts = 20 giây`:

| Tham số | Giá trị | Ý nghĩa vật lý |
|---|---:|---|
| `na` | 12 | nhớ độ ẩm đất 4 phút |
| `nb` | 3 | nhớ input 1 phút |
| `nk` | 2 | input ảnh hưởng sau 40 giây |
| `alpha` | 0.1 | regularization Ridge nhẹ |

Đây là cấu hình hợp lý cho mô hình nhỏ: không quá ngắn như `ARX(5,1,2)` cũ, nhưng cũng không quá dài đến mức khó giải thích.

## 4. ARX thuần và feature mở rộng

Pipeline có thêm feature vật lý:

- `Light_log`;
- `TempIn_x_HumiIn`;
- `TempIn_x_Light`;
- `HumiIn_x_Light`;
- `Indoor_Dryness`;
- `VPD_Proxy_In`;
- chu kỳ ngày: `Hour_sin`, `Hour_cos`, `Day_sin`, `Day_cos`.

`SP_Center` và `SP_Width` từng được thử nhưng đã xóa khỏi input model vì trong dữ liệu hiện tại setpoint là hằng số, không làm thay đổi kết quả.

Các feature này không làm model thành neural network. Model vẫn tuyến tính theo tham số:

```text
y = theta^T * phi
```

Trong đó `phi` là vector feature đã xây dựng trước.

Cách nói đúng:

```text
Đây là ARX tuyến tính theo tham số, có mở rộng input bằng feature vật lý tính từ cảm biến trong mô hình.
```

## 5. Regularization là gì?

Nếu dùng hồi quy tuyến tính thường, hệ số có thể quá lớn khi feature tương quan nhau. Ridge regularization thêm một mức phạt:

```text
min ||X theta - y||^2 + alpha * ||theta||^2
```

Lợi ích:

- giảm hệ số quá nhạy;
- giúp mô phỏng free-run ổn định hơn;
- phù hợp khi có nhiều feature lag.

## 6. Các kiểu đánh giá

### 6.1. One-step prediction

Dự đoán từng bước ngắn, thường cao:

```text
FIT_1step = 95.373
```

Không nên chỉ báo chỉ số này vì nó dễ làm kết quả nhìn quá đẹp.

### 6.2. Multi-step prediction

Dự đoán nhiều bước:

```text
FIT_12 = 90.839
FIT_60 = 88.047
```

Với `20 giây/mẫu`, `FIT_60` tương ứng horizon 20 phút.

### 6.3. Free-run simulation

Model tự chạy liên tục:

```text
FIT_sim = 78.950
```

Đây là chỉ số khó hơn vì lỗi có thể tích lũy.

## 7. Công thức FIT

```text
FIT = 100 * (1 - norm(y_true - y_pred) / norm(y_true - mean(y_true)))
```

Nếu `FIT < 0`, dự đoán còn tệ hơn lấy giá trị trung bình làm dự đoán. Điều này có thể xảy ra khi free-run bị trôi.

## 8. Tài liệu tham khảo

- MathWorks ARX model: https://www.mathworks.com/help/ident/ref/arx.html
- MathWorks Nonlinear ARX model: https://www.mathworks.com/help/ident/ref/nlarx.html
- Narendra và Parthasarathy, neural networks cho nhận dạng và điều khiển hệ động học: https://pubmed.ncbi.nlm.nih.gov/18282820/
