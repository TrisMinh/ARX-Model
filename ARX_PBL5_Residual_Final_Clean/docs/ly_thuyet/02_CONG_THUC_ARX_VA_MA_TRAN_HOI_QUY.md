# Công thức ARX và ma trận hồi quy

File này giải thích ARX từ công thức lý thuyết đến đúng cách code hiện tại đang xây dựng ma trận để train.

Ghi chú: file giữ cả công thức dạng `text` và công thức LaTeX trong `$$...$$`. Dạng `text` dễ đọc ở mọi nơi; dạng LaTeX hiển thị đẹp trong VS Code Markdown Preview, GitHub hoặc trình đọc hỗ trợ MathJax/KaTeX.

## 1. ARX là gì?

ARX là viết tắt của:

```text
AutoRegressive with eXogenous input
```

Dịch gần nghĩa:

```text
Tự hồi quy theo đầu ra quá khứ, có thêm đầu vào bên ngoài.
```

Trong bài này:

- "AutoRegressive" nghĩa là dùng `Soil_Moisture` quá khứ để dự đoán `Soil_Moisture` hiện tại;
- "eXogenous input" nghĩa là dùng các biến bên ngoài đầu ra, ví dụ `Temperature_In`, `Humidity_In`, `Light_In`, `Drip`, `Mist`, `Fan`.

## 2. Công thức ARX truyền thống

Công thức ARX một đầu vào có thể viết:

```text
y(k) + a1*y(k-1) + ... + a_na*y(k-na)
= b1*u(k-nk) + b2*u(k-nk-1) + ... + b_nb*u(k-nk-nb+1) + e(k)
```

Dạng công thức đẹp:

$$
y(k) + a_1y(k-1) + a_2y(k-2) + \cdots + a_{n_a}y(k-n_a)
= b_1u(k-n_k) + b_2u(k-n_k-1) + \cdots + b_{n_b}u(k-n_k-n_b+1) + e(k)
$$

Viết gọn bằng ký hiệu tổng:

$$
y(k)
= -\sum_{i=1}^{n_a} a_i y(k-i)
+ \sum_{j=1}^{n_b} b_j u(k-n_k-j+1)
+ e(k)
$$

Trong đó:

- `y(k)`: đầu ra tại thời điểm k;
- `u(k)`: đầu vào tại thời điểm k;
- `na`: số bậc trễ của đầu ra;
- `nb`: số bậc trễ của đầu vào;
- `nk`: độ trễ từ đầu vào đến đầu ra;
- `e(k)`: nhiễu hoặc sai số mô hình.

Với nhiều đầu vào, công thức mở rộng thành:

```text
y(k) = phần từ y quá khứ
     + phần từ u1 quá khứ
     + phần từ u2 quá khứ
     + ...
     + sai số
```

Dạng nhiều input:

$$
y(k)
= -\sum_{i=1}^{n_a} a_i y(k-i)
+ \sum_{m=1}^{M}\sum_{j=1}^{n_b} b_{m,j}u_m(k-n_k-j+1)
+ e(k)
$$

Trong đó `M` là số input.

## 3. Cấu tạo từng phần của ARX

ARX có thể hiểu là ghép từ 4 khối chính:

```text
ARX = phần tự hồi quy + phần input ngoài + phần trễ + phần sai số
```

### Phần 1: AutoRegressive, tức tự hồi quy

Đây là phần dùng đầu ra quá khứ để dự đoán đầu ra hiện tại:

```text
y(k-1), y(k-2), ..., y(k-na)
```

Trong bài này:

```text
y(k) = Soil_Moisture(k)
```

Nên phần tự hồi quy là:

```text
Soil_Moisture(k-1)
Soil_Moisture(k-2)
...
Soil_Moisture(k-na)
```

Ý nghĩa vật lý: độ ẩm đất có quán tính. Độ ẩm hiện tại thường gần với độ ẩm vài mẫu trước đó, chứ không nhảy ngẫu nhiên. Vì vậy ARX cho phép mô hình "nhớ" trạng thái trước của đất.

Nếu `na` lớn hơn, mô hình nhớ quá khứ dài hơn.

Ví dụ với `T_s = 20 giây`:

```text
na = 12 -> nhớ 12 mẫu quá khứ = 240 giây = 4 phút
```

### Phần 2: eXogenous input, tức đầu vào bên ngoài

Đây là phần dùng các biến tác động từ bên ngoài đầu ra:

```text
u(k)
```

Trong bài này, input có thể gồm:

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

Ý nghĩa:

- `Drip`: bơm/tưới làm đất ẩm lên;
- `Mist`: phun sương có thể ảnh hưởng gián tiếp đến độ ẩm đất và vi khí hậu;
- `Fan`: quạt làm thay đổi trao đổi không khí, ảnh hưởng bay hơi;
- `Temperature_In`: nhiệt độ trong cao thường làm khô nhanh hơn;
- `Humidity_In`: không khí ẩm làm bay hơi chậm hơn;
- `Light_In`: ánh sáng mạnh thường làm nóng và tăng bay hơi;
- các feature phụ giúp ARX tuyến tính biểu diễn một phần quan hệ phức tạp hơn.

### Phần 3: Độ trễ input `nk`

Không phải bật bơm/quạt là độ ẩm đất đổi ngay trong cùng mẫu. Thường phải có độ trễ.

`nk` là số mẫu trễ tối thiểu từ input đến output.

Nếu:

```text
nk = 1
T_s = 20 giây
```

thì input gần nhất được dùng là:

```text
u(k-1)
```

nghĩa là tác động sớm nhất sau 20 giây.

Nếu:

```text
nk = 2
```

thì input gần nhất được dùng là:

```text
u(k-2)
```

nghĩa là tác động sớm nhất sau 40 giây.

### Phần 4: Bộ nhớ input `nb`

`nb` là số mẫu quá khứ của mỗi input được dùng.

Nếu:

```text
nb = 6
nk = 1
```

thì với mỗi input, ARX dùng:

```text
u(k-1), u(k-2), u(k-3), u(k-4), u(k-5), u(k-6)
```

Với `T_s = 20 giây`, phần input nhớ:

```text
6 mẫu * 20 giây = 120 giây = 2 phút
```

Ý nghĩa vật lý: một lần bật bơm hoặc quạt có thể còn ảnh hưởng trong nhiều mẫu sau đó, không chỉ một mẫu duy nhất.

### Phần 5: Hệ số mô hình

Mỗi thành phần quá khứ có một hệ số:

```text
y_hat(k) =
    c1*y(k-1)
  + c2*y(k-2)
  + ...
  + d1*u(k-1)
  + d2*u(k-2)
  + ...
  + bias
```

Dạng công thức đẹp:

$$
\hat{y}(k)
= \sum_{i=1}^{n_a} c_i y(k-i)
+ \sum_{m=1}^{M}\sum_{j=1}^{n_b} d_{m,j}u_m(k-n_k-j+1)
+ c_0
$$

Hệ số cho biết mức ảnh hưởng tuyến tính của từng thành phần.

Ví dụ:

```text
d_Drip_lag2 > 0
```

có thể hiểu là bật bơm trước đó 2 mẫu có xu hướng làm tăng độ ẩm đất hiện tại.

Nếu:

```text
d_Fan_lag3 < 0
```

có thể hiểu là bật quạt trước đó 3 mẫu có xu hướng làm độ ẩm đất giảm, do tăng bay hơi hoặc trao đổi khí.

Không nên diễn giải từng hệ số quá tuyệt đối nếu các input tương quan mạnh với nhau. Khi nhiều biến đi cùng nhau, hệ số là ảnh hưởng trong điều kiện các biến khác đã có mặt trong mô hình.

### Phần 6: Bias

Bias là hằng số nền:

```text
bias
```

Nó giúp mô hình có mức nền phù hợp, thay vì bắt mọi dự đoán phải đi qua gốc tọa độ.

Trong code, bias là cột toàn số `1` trong ma trận hồi quy.

### Phần 7: Sai số `e(k)`

Sau khi mô hình dự đoán, phần còn lại là sai số:

```text
e_arx(k) = y_true(k) - y_arx(k)
```

Dạng công thức đẹp:

$$
e_{\mathrm{arx}}(k)=y_{\mathrm{true}}(k)-y_{\mathrm{arx}}(k)
$$

Sai số này gồm:

- nhiễu cảm biến;
- nhiễu môi trường;
- động học chưa được input mô tả hết;
- sai số do mô hình tuyến tính chưa đủ;
- sai số do chọn `na`, `nb`, `nk` chưa hoàn hảo.

Nếu dùng mô hình cuối Hybrid thì sai số cuối là:

```text
e_hybrid(k) = y_true(k) - y_hybrid(k)
```

Dạng công thức đẹp:

$$
e_{\mathrm{hybrid}}(k)=y_{\mathrm{true}}(k)-y_{\mathrm{hybrid}}(k)
$$

Vậy `e(k)` luôn là sai số của mô hình đang xét, không phải output của residual model.

## 4. Công thức đúng với code hiện tại

Trong code hiện tại, phương trình được viết theo dạng hồi quy trực tiếp:

```text
y_hat(k) =
    c1*y(k-1) + c2*y(k-2) + ... + c_na*y(k-na)
  + d11*u1(k-nk) + d12*u1(k-nk-1) + ...
  + d21*u2(k-nk) + d22*u2(k-nk-1) + ...
  + bias
```

Dạng công thức đẹp:

$$
\hat{y}(k)
= \sum_{i=1}^{n_a} c_i y(k-i)
+ \sum_{m=1}^{M}\sum_{j=1}^{n_b} d_{m,j}u_m(k-n_k-j+1)
+ c_0
$$

Điểm cần hiểu: dạng này tương đương ARX truyền thống, chỉ khác cách đặt dấu hệ số. Công thức truyền thống hay viết `y(k) + a1*y(k-1)`, còn code viết thẳng `y_hat(k) = c1*y(k-1)`. Hai cách chỉ khác quy ước hệ số.

## 5. Ý nghĩa `na`, `nb`, `nk`

Với chu kỳ lấy mẫu:

```text
T_s = 20 giây
```

Nếu model chọn:

```text
na = 12
nb = 6
nk = 1
```

thì ý nghĩa vật lý là:

```text
na = 12  -> nhớ đầu ra quá khứ 12 mẫu = 240 giây = 4 phút
nb = 6   -> nhớ mỗi đầu vào 6 mẫu = 120 giây = 2 phút
nk = 1   -> đầu vào trễ tối thiểu 1 mẫu = 20 giây
```

Vậy mô hình không chỉ nhìn một điểm hiện tại. Nó nhìn lịch sử ngắn của hệ, phù hợp với nhà kính nhỏ.

## 6. Vector hồi quy

Tại mỗi thời điểm `k`, ta gom các giá trị quá khứ thành một vector:

```text
phi(k) = [
  y(k-1), y(k-2), ..., y(k-na),
  u1(k-nk), u1(k-nk-1), ..., u1(k-nk-nb+1),
  u2(k-nk), u2(k-nk-1), ..., u2(k-nk-nb+1),
  ...
  1
]
```

Số `1` cuối cùng là bias/intercept.

Dự đoán:

```text
y_hat(k) = phi(k) * theta
```

Dạng công thức đẹp:

$$
\hat{y}(k)=\varphi(k)^T\theta
$$

Trong đó:

```text
theta = vector hệ số cần học
```

Với:

$$
\varphi(k)=
\begin{bmatrix}
y(k-1) \\
\vdots \\
y(k-n_a) \\
u_1(k-n_k) \\
\vdots \\
u_M(k-n_k-n_b+1) \\
1
\end{bmatrix},
\quad
\theta=
\begin{bmatrix}
c_1 \\
\vdots \\
c_{n_a} \\
d_{1,1} \\
\vdots \\
d_{M,n_b} \\
c_0
\end{bmatrix}
$$

## 7. Ma trận hồi quy

Nếu có nhiều thời điểm k, ta xếp các vector `phi(k)` thành ma trận:

```text
Phi = [
  phi(k1)
  phi(k2)
  phi(k3)
  ...
]
```

Vector đầu ra thật:

```text
Y = [
  y(k1)
  y(k2)
  y(k3)
  ...
]
```

Bài toán train ARX trở thành:

```text
Y ≈ Phi * theta
```

Dạng công thức đẹp:

$$
Y \approx \Phi\theta
$$

Mục tiêu là tìm `theta` sao cho sai số nhỏ nhất.

## 8. Least Squares

Nếu không regularization, nghiệm bình phương tối thiểu là:

```text
theta = (Phi^T * Phi)^(-1) * Phi^T * Y
```

Dạng công thức đẹp:

$$
\theta=(\Phi^T\Phi)^{-1}\Phi^TY
$$

Ý nghĩa: tìm bộ hệ số làm tổng bình phương sai số nhỏ nhất:

```text
min sum( y(k) - y_hat(k) )^2
```

Dạng công thức đẹp:

$$
\min_{\theta}\sum_{k=1}^{N}\left(y(k)-\hat{y}(k)\right)^2
$$

Trong code, nếu `alpha <= 0`, hàm `np.linalg.lstsq` được dùng.

## 9. Ridge regularization và alpha

Nếu có `alpha > 0`, code dùng Ridge:

```text
theta = (Phi^T * Phi + alpha * I)^(-1) * Phi^T * Y
```

Dạng công thức đẹp:

$$
\theta=(\Phi^T\Phi+\alpha I)^{-1}\Phi^TY
$$

`alpha` là hệ số phạt regularization.

Nói dễ hiểu:

```text
alpha = mức độ "kìm" hệ số theta lại
```

Nếu:

```text
alpha = 0
```

thì mô hình là least squares thường. Nó cố fit train tốt nhất có thể, nhưng nếu dữ liệu nhiễu hoặc feature tương quan mạnh, hệ số có thể lớn bất thường.

Nếu:

```text
alpha > 0
```

thì mô hình không chỉ cố giảm sai số, mà còn bị phạt nếu hệ số quá lớn.

Bài toán Ridge có thể hiểu là:

```text
minimize  sum( y(k) - y_hat(k) )^2  +  alpha * sum(theta_i^2)
```

Dạng công thức đẹp:

$$
\min_{\theta}
\left[
\sum_{k=1}^{N}\left(y(k)-\hat{y}(k)\right)^2
+\alpha\sum_{i=1}^{p}\theta_i^2
\right]
$$

Trong đó:

```text
sum( y(k) - y_hat(k) )^2 = phần muốn fit dữ liệu
alpha * sum(theta_i^2)   = phần phạt hệ số quá lớn
```

Ý nghĩa của `alpha`:

```text
alpha nhỏ -> ít phạt, model bám train mạnh hơn
alpha lớn -> phạt mạnh, hệ số nhỏ hơn, model ổn định hơn nhưng có thể underfit
```

Ví dụ trực giác:

```text
alpha = 0      -> fit train rất mạnh, dễ nhạy với nhiễu
alpha = 0.1    -> phạt nhẹ
alpha = 1      -> phạt vừa
alpha = 10     -> phạt mạnh hơn, thường ổn định hơn khi free-run
```

Với ARX, `alpha` quan trọng vì mô hình không chỉ dự đoán một bước. Khi free-run, mô hình dùng lại chính dự đoán quá khứ của nó. Nếu hệ số quá lớn hoặc quá nhạy, sai số nhỏ có thể bị khuếch đại theo thời gian.

Ridge giúp mô hình ổn định hơn khi:

- số feature nhiều;
- các feature tương quan với nhau;
- dữ liệu có nhiễu;
- mô phỏng free-run dễ bị trôi.

Trong code hiện tại, bias không bị phạt:

```text
penalty[-1, -1] = 0
```

Điều này hợp lý vì bias chỉ là mức lệch nền, không nên ép về 0 như các hệ số động học.

Trong project hiện tại, `alpha` cũng là một phần của cấu trúc candidate. Pipeline thử nhiều giá trị `alpha`, train trên train, đánh giá trên validation, rồi chọn cấu trúc có validation robust score tốt nhất.

Ví dụ model final:

```text
ARX_na12_nb6_nk1_alpha10
```

Nghĩa là:

```text
na = 12
nb = 6
nk = 1
alpha = 10
```

`alpha=10` không phải là input vật lý của nhà kính. Nó là tham số huấn luyện để điều chỉnh độ ổn định của hệ số ARX.

## 10. Liên hệ với code

Trong `build_arx_matrix`, code làm đúng các bước:

```text
1. Lấy y = Soil_Moisture
2. Tạo các cột y(k-1) đến y(k-na)
3. Với từng input, tạo các cột u(k-nk) đến u(k-nk-nb+1)
4. Thêm cột bias toàn số 1
5. Trả về X và y_target
```

Trong `fit_arx`, code học hệ số:

```text
theta = nghiệm least squares hoặc ridge
```

Vậy ARX ở đây không phải chỉ là hồi quy tuyến tính thường. Nó là hồi quy tuyến tính trên một vector có cấu trúc thời gian, gồm đầu ra quá khứ và đầu vào quá khứ.

## 11. Câu trả lời nếu thầy hỏi "ARX khác linear regression gì?"

Có thể trả lời:

```text
Về thuật toán ước lượng hệ số, ARX có thể dùng least squares giống linear regression. Nhưng điểm khác là cách xây dựng biến đầu vào. Linear regression thường dùng các biến tại cùng một thời điểm, còn ARX xây dựng vector hồi quy từ đầu ra quá khứ và đầu vào quá khứ, có tham số na, nb, nk biểu diễn bộ nhớ và độ trễ của hệ động. Vì vậy ARX là mô hình động tuyến tính, không chỉ là hồi quy tuyến tính tĩnh.
```
