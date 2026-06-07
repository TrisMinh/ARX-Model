# Lý thuyết nền tảng nhận dạng hệ thống

File này là lớp nền trước khi đọc ARX. Mục tiêu là hiểu bài toán đang làm là gì, vì sao phải có dữ liệu vào/ra theo thời gian, và vì sao không thể chỉ nhìn một con số FIT rồi kết luận mô hình tốt.

## 1. Bài toán nhận dạng hệ thống là gì?

Nhận dạng hệ thống là quá trình xây dựng mô hình toán học từ dữ liệu đo được.

Trong bài này, hệ thống là mô hình nhà kính nhỏ. Ta không cố viết đầy đủ toàn bộ phương trình vật lý của đất, không khí, bay hơi, quạt, bơm, phun sương. Thay vào đó, ta đo dữ liệu theo thời gian rồi tìm mô hình mô tả quan hệ:

```text
đầu vào quá khứ  ->  độ ẩm đất hiện tại hoặc tương lai
```

Ký hiệu tổng quát:

```text
y(k) = độ ẩm đất tại mẫu thứ k
u(k) = vector tín hiệu đầu vào tại mẫu thứ k
e(k) = nhiễu hoặc phần mô hình chưa giải thích được
T_s  = chu kỳ lấy mẫu
```

Với project hiện tại:

```text
T_s = 20 giây
y(k) = Soil_Moisture(k)
u(k) gồm Temperature_In, Humidity_In, Light_In, Drip, Mist, Fan và các feature phụ
```

## 2. Vì sao phải dùng dữ liệu quá khứ?

Độ ẩm đất không phản ứng tức thời hoàn toàn.

Ví dụ bật bơm tại thời điểm hiện tại:

```text
Drip(k) = 1
```

Độ ẩm đất có thể chưa tăng ngay tại `k`, mà tăng sau vài mẫu:

```text
Soil_Moisture(k+1), Soil_Moisture(k+2), Soil_Moisture(k+3), ...
```

Vì vậy mô hình phải dùng các giá trị trễ:

```text
y(k-1), y(k-2), ...
u(k-1), u(k-2), ...
```

Đây là lý do ARX có các tham số `na`, `nb`, `nk`.

## 3. Đầu ra, đầu vào và nhiễu

Một hệ thống thực tế luôn có nhiễu:

```text
độ ẩm đất đo được = độ ẩm thật + sai số cảm biến + nhiễu môi trường
```

Trong mô hình:

```text
y(k) = phần mô hình giải thích được + e(k)
```

`e(k)` không phải lúc nào cũng là lỗi xấu. Nó đại diện cho những phần chưa đo được hoặc chưa đưa vào mô hình, ví dụ:

- cảm biến đất đặt chưa đều;
- nước ngấm không đồng nhất;
- gió ngoài môi trường;
- sai số cảm biến;
- độ trễ cơ khí của bơm/quạt;
- mô hình tuyến tính chưa mô tả hết phi tuyến.

Điểm rất quan trọng: `e(k)` không phải là một cột dữ liệu đo trực tiếp và cũng không phải output của residual model. `e(k)` luôn là sai số còn lại sau mô hình đang xét.

Nếu đang xét ARX backbone:

```text
y_arx(k) = phần ARX dự đoán được
e_arx(k) = y_true(k) - y_arx(k)
```

Khi đó:

```text
y_true(k) = y_arx(k) + e_arx(k)
```

Nếu đang xét mô hình cuối Hybrid:

```text
y_hybrid(k) = y_arx(k) + shrink*r_hat(k)
```

Trong đó:

```text
r_hat(k) = phần sai số ARX mà residual model dự đoán để sửa thêm
```

Sai số cuối của mô hình Hybrid là:

```text
e_hybrid(k) = y_true(k) - y_hybrid(k)
```

Hay viết đầy đủ:

```text
e_hybrid(k) = y_true(k) - [y_arx(k) + shrink*r_hat(k)]
```

Vậy:

```text
e_arx(k)     = sai số của ARX trước khi sửa
r_hat(k)     = phần sửa do residual model dự đoán
e_hybrid(k)  = sai số cuối sau khi đã sửa
```

Nếu báo cáo bản cuối đang dùng Hybrid ARX residual, thì `e(k)` cuối cùng nên hiểu là `e_hybrid(k)`, không phải `r_hat(k)`.

## 4. Vì sao lấy mẫu 20 giây thay vì 5 phút?

Với nhà kính nhỏ `30x50x30 cm`, thể tích nhỏ nên môi trường thay đổi nhanh hơn nhà kính thật kích thước lớn. Quạt, mist và bơm có thể ảnh hưởng trong vài chục giây đến vài phút.

Nếu lấy mẫu 5 phút:

```text
00:00:00
00:05:00
00:10:00
```

thì nhiều đoạn quá độ đã bị bỏ qua. Mô hình chỉ thấy trạng thái trước và sau, nhưng không thấy quá trình diễn ra ở giữa.

Nếu lấy mẫu 20 giây:

```text
00:00:00
00:00:20
00:00:40
00:01:00
...
```

mô hình có cơ hội học rõ hơn độ trễ và tốc độ phản ứng của hệ.

## 5. Train, validation và test

Dữ liệu chuỗi thời gian không nên trộn ngẫu nhiên. Nếu trộn random, mẫu tương lai có thể nằm trong train và mẫu quá khứ nằm trong test, làm kết quả đẹp ảo.

Project hiện tại dùng cách chia theo thời gian:

```text
70% đầu      -> train
15% tiếp theo -> validation
15% cuối     -> test
```

Ý nghĩa:

- `train`: dùng để học hệ số mô hình;
- `validation`: dùng để chọn cấu trúc mô hình, ví dụ chọn `na`, `nb`, `nk`, `alpha`;
- `test`: chỉ dùng sau cùng để báo cáo kết quả cuối.

## 6. Vì sao cần validation?

Nếu thử nhiều mô hình rồi chọn mô hình có test tốt nhất, test không còn khách quan nữa. Khi đó ta đã dùng test để chọn mô hình, tức là leakage.

Quy trình đúng:

```text
train mô hình trên train
chọn mô hình bằng validation
khóa lựa chọn
đánh giá một lần trên test
```

Project hiện tại chọn mô hình bằng `validation robust score`, không chọn trực tiếp bằng test.

## 7. Vì sao phải chuẩn hóa dữ liệu?

Các biến có thang đo rất khác nhau:

```text
Soil_Moisture: khoảng 50-65
Temperature_In: khoảng 20-40
Humidity_In: khoảng 40-100
Light_In: có thể lên hàng trăm hoặc hơn 1000
Drip/Fan/Mist: chỉ 0 hoặc 1
```

Nếu không chuẩn hóa, biến có giá trị lớn như ánh sáng có thể làm bài toán hồi quy mất cân bằng số học.

Chuẩn hóa thường dùng:

```text
z = (x - mean_train) / std_train
```

Quan trọng: `mean_train` và `std_train` chỉ được tính trên train. Validation và test dùng lại thống kê của train. Không được tính mean/std trên toàn bộ data vì sẽ làm lộ thông tin test.

Giải thích dễ hiểu: train là dữ liệu quá khứ mà mô hình được phép học. Validation và test đại diện cho dữ liệu tương lai. Khi triển khai thật, ta không biết trước dữ liệu tương lai sẽ có trung bình và độ lệch chuẩn bao nhiêu. Vì vậy, nếu dùng cả validation/test để tính `mean` và `std`, nghĩa là ta đã lén nhìn một phần phân bố tương lai.

Ví dụ:

```text
train = [55, 56, 57]
test  = [58, 59]
```

Tính đúng:

```text
mean_train = 56
std_train  ≈ 0.816
```

Chuẩn hóa test bằng thống kê train:

```text
58 -> (58 - 56) / 0.816 ≈ 2.45
59 -> (59 - 56) / 0.816 ≈ 3.67
```

Tính sai là lấy toàn bộ dữ liệu:

```text
all_data = [55, 56, 57, 58, 59]
mean_all = 57
std_all  ≈ 1.414
```

Lúc này `mean_all` đã bị ảnh hưởng bởi `[58, 59]` của test. Tức là khi xử lý train, ta đã dùng thông tin từ test. Đây là leakage.

Trong code project, luồng đúng là:

```python
inside_stats = fit_scale_stats(train, INSIDE_INPUT_COLS)
train_inside_z = apply_scale(train, inside_stats)
val_inside_z = apply_scale(val, inside_stats)
test_inside_z = apply_scale(test, inside_stats)
```

Nghĩa là:

```text
1. Fit scaler trên train.
2. Dùng lại scaler đó cho train.
3. Dùng lại scaler đó cho validation.
4. Dùng lại scaler đó cho test.
```

Không có bước nào fit scaler trên toàn bộ data.

## 8. Kết luận nền tảng

Bài toán này không chỉ là "fit một đường tuyến tính". Đây là bài toán nhận dạng hệ thống động:

- dữ liệu có thứ tự thời gian;
- đầu vào tác động có độ trễ;
- độ ẩm đất phụ thuộc vào chính quá khứ của nó;
- actuator phải được kích thích đủ để mô hình học được;
- test phải tách theo thời gian;
- metric phải đọc cùng với cách mô phỏng.

Sau khi hiểu các ý này, đọc ARX sẽ dễ hơn nhiều.
