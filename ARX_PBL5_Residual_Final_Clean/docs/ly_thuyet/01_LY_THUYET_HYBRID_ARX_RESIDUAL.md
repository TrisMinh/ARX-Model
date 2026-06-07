# Lý thuyết Hybrid ARX residual

## 1. ARX backbone

ARX dự đoán đầu ra hiện tại bằng đầu ra quá khứ và input quá khứ:

```text
y(k) = y_arx(k) + e_arx(k)
```

Dạng công thức đẹp:

$$
y(k)=y_{\mathrm{arx}}(k)+e_{\mathrm{arx}}(k)
$$

Trong đó:

```text
y_arx(k) =
    a1*y(k-1) + ... + a_na*y(k-na)
  + b1*u(k-nk) + ... + b_nb*u(k-nk-nb+1)
  + c
```

Dạng công thức đẹp:

$$
y_{\mathrm{arx}}(k)
=
\sum_{i=1}^{n_a}c_i y(k-i)
+
\sum_{m=1}^{M}\sum_{j=1}^{n_b}d_{m,j}u_m(k-n_k-j+1)
+ c_0
$$

Sai số của riêng ARX là:

```text
e_arx(k) = y_true(k) - y_arx(k)
```

Dạng công thức đẹp:

$$
e_{\mathrm{arx}}(k)=y_{\mathrm{true}}(k)-y_{\mathrm{arx}}(k)
$$

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
residual_arx(k) = e_arx(k) = y_true(k) - y_arx_sim(k)
```

Dạng công thức đẹp:

$$
r_{\mathrm{arx}}(k)
=
e_{\mathrm{arx}}(k)
=
y_{\mathrm{true}}(k)-y_{\mathrm{arx,sim}}(k)
$$

Nếu residual còn có quy luật theo quá khứ, ta có thể học phần sai số đó bằng một model phụ.

## 3. Công thức model cuối

Model cuối:

```text
y_hybrid(k) = y_arx_sim(k) + shrink * r_hat(k)
```

Dạng công thức đẹp:

$$
y_{\mathrm{hybrid}}(k)
=
y_{\mathrm{arx,sim}}(k)
+
\lambda\hat{r}(k)
$$

Trong đó `\lambda` chính là `shrink`.

Trong đó:

```text
r_hat(k) = residual_model(x_residual(k))
```

Dạng công thức đẹp:

$$
\hat{r}(k)=f_{\mathrm{residual}}\left(x_{\mathrm{residual}}(k)\right)
$$

Trong đó `x_residual(k)` chỉ gồm thông tin hợp lệ tại thời điểm dự đoán:

- dự đoán ARX tại hiện tại;
- các lag của quỹ đạo ARX mô phỏng;
- các lag input quá khứ;
- không dùng `Soil_Moisture` thật tương lai.

Điểm rất quan trọng:

```text
r_hat(k) không phải là e(k) cuối cùng.
```

`r_hat(k)` chỉ là phần residual model dự đoán để sửa dự đoán ARX. Sai số cuối cùng của model hybrid là:

```text
e_hybrid(k) = y_true(k) - y_hybrid(k)
```

Dạng công thức đẹp:

$$
e_{\mathrm{hybrid}}(k)
=
y_{\mathrm{true}}(k)
-
y_{\mathrm{hybrid}}(k)
$$

Thay công thức `y_hybrid(k)` vào:

```text
e_hybrid(k) = y_true(k) - [y_arx_sim(k) + shrink*r_hat(k)]
```

Dạng công thức đẹp:

$$
e_{\mathrm{hybrid}}(k)
=
y_{\mathrm{true}}(k)
-
\left[
y_{\mathrm{arx,sim}}(k)+\lambda\hat{r}(k)
\right]
$$

Vì vậy:

```text
Khi phân tích riêng ARX backbone:
e(k) = e_arx(k) = y_true(k) - y_arx(k)

Khi báo cáo mô hình cuối Hybrid:
e(k) = e_hybrid(k) = y_true(k) - y_hybrid(k)
```

Ví dụ ngắn:

```text
y_true = 56.0
y_arx  = 55.7
```

Residual ARX:

```text
e_arx = y_true - y_arx = 0.3
```

Residual model dự đoán:

```text
r_hat = 0.2
shrink = 0.5
```

Hybrid:

```text
y_hybrid = 55.7 + 0.5*0.2 = 55.8
```

Sai số cuối:

```text
e_hybrid = y_true - y_hybrid = 56.0 - 55.8 = 0.2
```

Kết luận:

```text
e_arx là sai số trước khi sửa.
r_hat là phần sửa do residual model dự đoán.
e_hybrid là sai số cuối của mô hình bảo vệ.
```

## 4. Vì sao không leakage?

Leakage nghĩa là dữ liệu tương lai hoặc dữ liệu test bị dùng lén trong quá trình học/chọn mô hình. Với bản Hybrid ARX residual, phải cẩn thận hơn ARX thường vì ta có thêm một model phụ học sai số.

Pipeline giữ các nguyên tắc dưới đây.

### Nguyên tắc 1: Chia dữ liệu theo thời gian

Dữ liệu được chia:

```text
70% đầu      -> train
15% tiếp theo -> validation
15% cuối     -> test
```

Ý nghĩa:

```text
train      = đoạn dùng để học mô hình
validation = đoạn dùng để chọn model/cấu hình
test       = đoạn giữ lại để chấm kết quả cuối
```

Không shuffle random vì đây là dữ liệu chuỗi thời gian. Nếu trộn ngẫu nhiên, mẫu tương lai có thể nằm trong train và mẫu quá khứ nằm trong test, làm kết quả đẹp ảo.

### Nguyên tắc 2: Scaler chỉ fit trên train

Scaler là bước tính `mean` và `std` để chuẩn hóa dữ liệu:

```text
z = (x - mean_train) / std_train
```

Chỉ được tính:

```text
mean_train, std_train từ train
```

Sau đó dùng lại `mean_train`, `std_train` cho validation và test.

Không được tính mean/std trên toàn bộ data vì như vậy validation/test đã ảnh hưởng vào bước tiền xử lý.

### Nguyên tắc 3: ARX train trên train

Đầu tiên, chỉ dùng tập train để học hệ số ARX:

```text
train data -> học theta_arx
```

Sau bước này ARX có thể dự đoán:

```text
y_arx_train(k)
y_arx_validation(k)
y_arx_test(k)
```

Nhưng hệ số ARX chỉ được học từ train.

### Nguyên tắc 4: "Residual train trên train" nghĩa là gì?

Câu này nghĩa là residual model chỉ được học từ sai số ARX trên tập train.

Làm từng bước:

```text
Bước 1:
ARX dự đoán trên train:
y_arx_train(k)

Bước 2:
Tính sai số ARX trên train:
residual_train(k) = y_true_train(k) - y_arx_train(k)

Bước 3:
Tạo feature hợp lệ cho residual model:
x_residual_train(k)

Bước 4:
Train residual model:
x_residual_train(k) -> residual_train(k)
```

Nói dễ hiểu:

```text
ARX đoán trên train.
Ta xem ARX sai bao nhiêu trên train.
Residual model học cách sửa sai số đó, nhưng chỉ học từ train.
```

Ví dụ:

```text
y_true_train = 56.0
y_arx_train  = 55.7
residual_train = 56.0 - 55.7 = 0.3
```

Residual model học rằng trong điều kiện tương tự, ARX có thể đang bị thấp khoảng `0.3`.

Điểm quan trọng:

```text
Không được lấy residual_test = y_true_test - y_arx_test để train.
```

Vì nếu lấy `y_true_test` để tính residual rồi train hoặc chỉnh model, tức là test đã bị lộ.

### Nguyên tắc 5: Validation dùng để chọn residual model và shrink

Sau khi train residual model trên train, ta thử nó trên validation.

Ví dụ có nhiều candidate:

```text
Candidate A: Ridge residual, shrink=0.25
Candidate B: Ridge residual, shrink=0.50
Candidate C: HGB residual,   shrink=0.25
Candidate D: HGB residual,   shrink=0.50
```

Ở đây:

```text
Ridge residual = dùng mô hình Ridge để học sai số ARX
HGB residual   = dùng mô hình HistGradientBoosting để học sai số ARX
```

Hai mô hình này không thay thế ARX. Chúng chỉ là hai cách khác nhau để dự đoán:

```text
r_hat(k) ≈ residual_train(k)
```

Trong đó:

```text
residual_train(k) = y_true_train(k) - y_arx_train(k)
```

#### Ridge residual là gì?

Ridge residual là một mô hình tuyến tính có regularization. Nó học quan hệ:

```text
x_residual(k) -> residual_train(k)
```

Dạng đơn giản:

```text
r_hat(k) = w1*x1(k) + w2*x2(k) + ... + bias
```

Dạng công thức đẹp:

$$
\hat{r}(k)=w^Tx_{\mathrm{residual}}(k)+b
$$

Ridge có thêm phần phạt hệ số lớn:

$$
\min_w
\left[
\sum_k\left(r(k)-\hat{r}(k)\right)^2
+\alpha\sum_i w_i^2
\right]
$$

Ý nghĩa:

- dễ giải thích hơn HGB;
- ít bị overfit hơn hồi quy tuyến tính thường;
- chỉ học quan hệ tuyến tính giữa feature residual và sai số ARX.

#### HGB residual là gì?

HGB là viết tắt của:

```text
HistGradientBoostingRegressor
```

Hiểu đơn giản: đây là mô hình cây quyết định tăng cường. Nó ghép nhiều cây nhỏ lại để học quan hệ phi tuyến.

Nó cũng học cùng mục tiêu:

```text
x_residual(k) -> residual_train(k)
```

nhưng khác Ridge ở chỗ HGB có thể học các kiểu quan hệ như:

```text
nếu quạt vừa bật và độ ẩm không khí thấp thì ARX có xu hướng sai khác
nếu ánh sáng cao và đất đang khô thì residual có dạng khác
nếu ARX đang trôi theo một hướng thì cần sửa nhẹ theo hướng ngược lại
```

Ý nghĩa:

- có thể học quan hệ phi tuyến tốt hơn Ridge;
- đôi khi cải thiện free-run tốt hơn;
- khó giải thích hơn Ridge;
- dễ overfit hơn nếu không chọn bằng validation.

Vì vậy pipeline không mặc định tin HGB. Nó để Ridge, HGB và cả `shrink=0` cùng cạnh tranh trên validation. Candidate nào ổn định nhất theo validation robust score thì được chọn.

#### Vì sao chọn Ridge và HGB, không chọn hàng loạt model khác?

Mục tiêu của residual correction trong đồ án này không phải thay ARX bằng một mô hình quá phức tạp. Mục tiêu là:

```text
giữ ARX làm backbone
chỉ sửa phần sai số ARX còn có quy luật
không làm mô hình khó giải thích quá mức
không tăng nguy cơ overfit quá cao
```

Vì vậy chọn hai họ residual model có vai trò bổ sung cho nhau:

```text
Ridge residual -> baseline tuyến tính, đơn giản, dễ giải thích
HGB residual   -> mô hình phi tuyến nhẹ, kiểm tra xem residual có quan hệ phi tuyến không
```

Lý do chọn Ridge:

- cùng tinh thần với ARX tuyến tính;
- dễ giải thích khi bảo vệ;
- train nhanh;
- ít tham số hơn neural network;
- có regularization để giảm overfit;
- dùng làm baseline: nếu Ridge đã đủ tốt thì không cần mô hình phức tạp.

Lý do chọn HGB:

- residual của ARX có thể còn quan hệ phi tuyến nhẹ;
- HGB học được điều kiện dạng "nếu ... thì ...";
- không cần scale dữ liệu quá nhạy như neural network;
- chạy ổn với dữ liệu bảng/time-lag feature;
- có thể kiểm soát độ phức tạp bằng `max_leaf_nodes`, `l2_regularization`, validation và shrink.

Vì sao không ưu tiên neural network residual?

- cần nhiều dữ liệu thật hơn để tránh overfit;
- khó giải thích hơn trong đồ án ARX;
- phải tuning nhiều thứ như learning rate, batch size, epochs, optimizer;
- kết quả có thể dao động theo seed;
- nếu dùng không khéo, validation đẹp nhưng test/free-run xấu.

Vì sao không dùng quá nhiều model khác như Random Forest, SVR, KNN?

- thử quá nhiều model sẽ làm báo cáo giống "săn điểm" hơn là nghiên cứu có kiểm soát;
- nhiều model khó giải thích trong bối cảnh MPC/ARX;
- càng nhiều lựa chọn, càng cần validation/test nghiêm ngặt hơn để tránh chọn nhầm do may mắn;
- PBL5 cần mô hình gọn, logic và có thể bảo vệ được.

Do đó, lựa chọn Ridge và HGB là có chủ đích:

```text
Ridge kiểm tra phần residual tuyến tính còn sót.
HGB kiểm tra phần residual phi tuyến nhẹ còn sót.
shrink kiểm soát mức sửa.
validation robust score quyết định có nên dùng residual hay quay về ARX.
```

Câu bảo vệ:

```text
Em không chọn quá nhiều mô hình residual để tránh biến bài toán thành săn model. Em chọn Ridge làm baseline tuyến tính, dễ giải thích và cùng tinh thần với ARX. Em chọn HGB như một mô hình phi tuyến nhẹ để kiểm tra phần residual còn quan hệ phi tuyến hay không. Cả hai đều được chọn bằng validation robust score và có shrink để tránh sửa quá tay. Nếu residual không giúp, grid có shrink=0 để quay lại ARX backbone.
```

Mỗi candidate đều:

```text
train residual trên train
chấm điểm trên validation
```

Candidate nào có validation robust score tốt nhất thì được chọn.

`shrink` là mức độ tin vào phần sửa residual:

```text
y_hybrid(k) = y_arx_sim(k) + shrink*r_hat(k)
```

Nếu `shrink` nhỏ, residual sửa nhẹ. Nếu `shrink` lớn, residual sửa mạnh hơn.

### Nguyên tắc 6: Test chỉ dùng sau khi đã chọn xong

Sau khi đã chọn xong:

```text
ARX structure
residual model
shrink
feature set
```

mới chạy trên test để báo cáo kết quả cuối.

Không được dùng test để chọn:

- chọn `na`, `nb`, `nk`;
- chọn `alpha`;
- chọn residual model;
- chọn `shrink`;
- chọn feature;
- chọn lần chạy đẹp nhất.

Nếu dùng test để chọn, test bị biến thành validation và kết quả test không còn khách quan.

### Nguyên tắc 7: Có quyền quay về ARX nếu residual không giúp

Trong grid có:

```text
shrink = 0
```

Khi:

```text
shrink = 0
```

thì:

```text
y_hybrid(k) = y_arx_sim(k) + 0*r_hat(k)
            = y_arx_sim(k)
```

Nghĩa là nếu residual không giúp trên validation, pipeline được phép chọn quay về ARX backbone. Đây là cách làm an toàn, vì residual không bị ép phải sửa nếu nó không thật sự cải thiện.

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
