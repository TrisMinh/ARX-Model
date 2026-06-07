# Công thức đánh giá: FIT, RMSE, Bias, R2

File này giải thích các chỉ số đánh giá mô hình được giữ lại trong project. Đây là phần rất nên nắm chắc khi bảo vệ, vì thầy có thể hỏi: "FIT là gì?", "Vì sao FIT âm?", "RMSE khác FIT ở đâu?".

## 1. Sai số dự đoán

Gọi:

```text
y(k)     = giá trị thật
y_hat(k) = giá trị mô hình dự đoán
e(k)     = sai số
```

Sai số:

```text
e(k) = y(k) - y_hat(k)
```

Dạng công thức đẹp:

$$
e(k)=y(k)-\hat{y}(k)
$$

Nếu `e(k) > 0`, mô hình dự đoán thấp hơn thực tế.

Nếu `e(k) < 0`, mô hình dự đoán cao hơn thực tế.

## 2. RMSE

RMSE là Root Mean Squared Error:

```text
RMSE = sqrt( mean( e(k)^2 ) )
```

Viết đầy đủ:

```text
RMSE = sqrt( (1/N) * sum từ k=1 đến N của (y(k) - y_hat(k))^2 )
```

Dạng công thức đẹp:

$$
\mathrm{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{k=1}^{N}
\left(y(k)-\hat{y}(k)\right)^2
}
$$

Ý nghĩa:

- cùng đơn vị với độ ẩm đất;
- phạt sai số lớn mạnh hơn vì có bình phương;
- càng nhỏ càng tốt.

Ví dụ:

```text
RMSE = 0.159
```

nghĩa là sai số trung bình theo kiểu bình phương khoảng `0.159% độ ẩm đất`.

## 3. Bias

Bias là sai số trung bình có dấu:

```text
Bias = mean( e(k) )
```

Dạng công thức đẹp:

$$
\mathrm{Bias}
=
\frac{1}{N}
\sum_{k=1}^{N}
\left(y(k)-\hat{y}(k)\right)
$$

Nếu:

```text
Bias > 0
```

mô hình có xu hướng dự đoán thấp hơn thực tế.

Nếu:

```text
Bias < 0
```

mô hình có xu hướng dự đoán cao hơn thực tế.

Bias gần 0 là tốt, nhưng bias gần 0 không có nghĩa là mô hình tốt hoàn toàn. Sai số dương và âm có thể triệt tiêu nhau.

## 4. FIT

FIT là chỉ số phần trăm mô hình bám theo dữ liệu thật tốt hơn so với việc chỉ đoán bằng giá trị trung bình.

Trong code hiện tại, FIT được tính theo chuẩn vector:

```text
FIT = 100 * (1 - ||y - y_hat|| / ||y - mean(y)||)
```

Dạng công thức đẹp:

$$
\mathrm{FIT}
=
100
\left(
1-
\frac{\left\|y-\hat{y}\right\|_2}
{\left\|y-\bar{y}\right\|_2}
\right)
$$

Trong đó:

```text
||y - y_hat||      = độ lớn vector sai số
||y - mean(y)||    = độ biến thiên của tín hiệu thật quanh trung bình
```

Với:

$$
\bar{y}=\frac{1}{N}\sum_{k=1}^{N}y(k)
$$

Vì:

```text
||y - y_hat|| / ||y - mean(y)||
```

là sai số đã được chuẩn hóa theo độ biến thiên của dữ liệu, nên FIT cho biết mô hình tốt hơn baseline trung bình bao nhiêu.

## 5. Cách hiểu FIT

Nếu:

```text
FIT = 100%
```

mô hình khớp hoàn hảo.

Nếu:

```text
FIT = 0%
```

mô hình chỉ tốt ngang việc đoán hằng số bằng trung bình của dữ liệu thật.

Nếu:

```text
FIT < 0%
```

mô hình còn tệ hơn đoán trung bình. Điều này có thể xảy ra khi mô phỏng free-run bị trôi mạnh.

Nếu:

```text
FIT = 83%
```

có thể hiểu gần đúng: sai số chuẩn hóa còn khoảng 17% so với độ biến thiên của tín hiệu thật.

## 6. Vì sao cùng RMSE nhưng FIT có thể khác?

FIT phụ thuộc vào độ biến thiên của `y`.

Nếu dữ liệu test biến thiên mạnh, cùng một RMSE sẽ cho FIT cao hơn. Nếu dữ liệu test gần như phẳng, cùng một RMSE có thể cho FIT thấp hơn.

Ví dụ:

```text
Data A: RMSE = 0.2, std(y) = 2.0
RMSE/std = 0.1 -> FIT khoảng 90%

Data B: RMSE = 0.2, std(y) = 0.5
RMSE/std = 0.4 -> FIT khoảng 60%
```

Vì vậy không nên so sánh FIT giữa hai bộ dữ liệu khác nhau rồi kết luận mô hình tốt hơn tuyệt đối. Muốn so sánh mô hình, phải chạy trên cùng data, cùng split.

## 7. R2

R2 thường được định nghĩa:

```text
R2 = 1 - sum(e(k)^2) / sum((y(k) - mean(y))^2)
```

Dạng công thức đẹp:

$$
R^2
=
1-
\frac{
\sum_{k=1}^{N}\left(y(k)-\hat{y}(k)\right)^2
}{
\sum_{k=1}^{N}\left(y(k)-\bar{y}\right)^2
}
$$

Ý nghĩa:

- càng gần 1 càng tốt;
- bằng 0 là ngang đoán trung bình;
- âm là tệ hơn đoán trung bình.

FIT và R2 đều so với baseline trung bình, nhưng cách lấy căn khác nhau. Với cùng tập dữ liệu:

```text
FIT = 100 * (1 - sqrt( sum(e^2) / sum((y - mean(y))^2) ))
```

Trong khi:

```text
R2 = 1 - sum(e^2) / sum((y - mean(y))^2)
```

Dạng công thức đẹp:

$$
\mathrm{FIT}
=
100
\left(
1-
\sqrt{
\frac{\sum e(k)^2}{\sum\left(y(k)-\bar{y}\right)^2}
}
\right)
$$

$$
R^2
=
1-
\frac{\sum e(k)^2}{\sum\left(y(k)-\bar{y}\right)^2}
$$

Do đó FIT và R2 không giống hệt nhau, nhưng có liên hệ chặt.

## 8. Ví dụ từ project hiện tại

Trên dữ liệu mới, ARX backbone:

```text
FIT_sim = 82.496%
RMSE_sim = 0.1681
```

Hybrid residual:

```text
FIT_sim = 83.433%
RMSE_sim = 0.1591
```

Mức cải thiện:

```text
FIT tăng khoảng 0.937 điểm
RMSE giảm khoảng 0.0090
```

Kết luận đúng:

```text
Residual có cải thiện, nhưng cải thiện nhỏ. Thành phần chính vẫn là ARX backbone và protocol dữ liệu phù hợp.
```

## 9. Câu trả lời nhanh khi thầy hỏi FIT là gì

Có thể trả lời:

```text
FIT là chỉ số phần trăm đánh giá mô hình bám dữ liệu thật tốt hơn bao nhiêu so với baseline đoán bằng giá trị trung bình. Công thức em dùng là 100 nhân với 1 trừ tỉ số giữa norm sai số dự đoán và norm độ lệch của dữ liệu thật so với trung bình. FIT 100% là khớp hoàn hảo, FIT 0% là ngang đoán trung bình, FIT âm là tệ hơn đoán trung bình.
```
