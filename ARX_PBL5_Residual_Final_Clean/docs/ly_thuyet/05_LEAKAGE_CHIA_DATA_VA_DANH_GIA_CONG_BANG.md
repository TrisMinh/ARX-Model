# Leakage, chia dữ liệu và đánh giá công bằng

File này giải thích các nguyên tắc để kết quả không bị ảo. Với bài mô hình hóa chuỗi thời gian, phần này quan trọng không kém thuật toán.

## 1. Leakage là gì?

Leakage là khi thông tin không được phép xuất hiện trong train lại lọt vào quá trình train hoặc chọn mô hình.

Ví dụ dễ hiểu:

```text
Muốn dự đoán ngày mai nhưng lại lén dùng dữ liệu ngày mai trong lúc train.
```

Trong chuỗi thời gian, leakage có thể xảy ra rất âm thầm.

## 2. Các loại leakage thường gặp

### Dùng tương lai của target

Sai:

```text
dùng Soil_Moisture(k+1) để dự đoán Soil_Moisture(k)
```

Vì khi triển khai thực tế, tại thời điểm `k`, ta chưa biết `Soil_Moisture(k+1)`.

### Dùng actuator tương lai

Sai nếu mục tiêu là mô phỏng tự do:

```text
dùng Drip(k+3), Fan(k+5) nếu các lệnh này chưa được quyết định tại thời điểm dự đoán
```

Trong MPC, ta có thể dùng chuỗi điều khiển tương lai giả định do MPC đề xuất, nhưng phải nói rõ đó là input kế hoạch của bộ điều khiển, không phải dữ liệu đo tương lai.

### Tính scale trên toàn bộ dữ liệu

Sai:

```text
mean = mean(train + validation + test)
std  = std(train + validation + test)
```

Đúng:

```text
mean_train = mean(train)
std_train  = std(train)
```

Sau đó dùng `mean_train`, `std_train` để scale validation và test.

Ví dụ cụ thể:

```text
train = [55, 56, 57]
test  = [58, 59]
```

Nếu làm đúng:

```text
mean_train = 56
std_train  ≈ 0.816
```

Chuẩn hóa train:

```text
55 -> (55 - 56) / 0.816 ≈ -1.225
56 -> (56 - 56) / 0.816 = 0
57 -> (57 - 56) / 0.816 ≈ 1.225
```

Chuẩn hóa test bằng cùng thống kê train:

```text
58 -> (58 - 56) / 0.816 ≈ 2.45
59 -> (59 - 56) / 0.816 ≈ 3.67
```

Nếu làm sai:

```text
all_data = [55, 56, 57, 58, 59]
mean_all = 57
std_all  ≈ 1.414
```

Khi đó test `[58, 59]` đã tham gia vào `mean_all` và `std_all`. Mô hình chưa train nhưng bước tiền xử lý đã biết dữ liệu tương lai cao hơn train. Đây là leakage nhẹ nhưng vẫn sai nguyên tắc.

Cách hiểu ngắn:

```text
Scaler cũng là một phần của mô hình.
Đã là một phần của mô hình thì chỉ được fit trên train.
```

Trong code hiện tại, project làm đúng:

```python
inside_stats = fit_scale_stats(train, INSIDE_INPUT_COLS)
train_inside_z = apply_scale(train, inside_stats)
val_inside_z = apply_scale(val, inside_stats)
test_inside_z = apply_scale(test, inside_stats)
```

Dòng đầu tiên là dòng quan trọng nhất: `fit_scale_stats(train, ...)` chỉ nhận `train`, không nhận toàn bộ data.

### Chọn model bằng test

Sai:

```text
thử 100 model
model nào test cao nhất thì chọn
```

Đúng:

```text
chọn bằng validation
test chỉ báo cáo cuối cùng
```

## 3. Validation robust score là gì?

`Validation robust score` là điểm dùng để chọn mô hình ổn định trên validation, không chỉ chọn mô hình có một con số validation đẹp nhất.

Nói thẳng theo quy trình chạy model: project tạo ra nhiều mô hình ứng viên, train từng mô hình trên train, chạy thử từng mô hình trên validation, rồi chọn mô hình có điểm validation tốt nhất. Sau khi chọn xong mới đem mô hình đó chạy trên test.

Luồng đúng:

```text
Tạo nhiều candidate:
  A = ARX(na=5,  nb=1, nk=2, alpha=1)
  B = ARX(na=12, nb=3, nk=2, alpha=0.1)
  C = ARX(na=12, nb=6, nk=1, alpha=10)
  ...

Với từng candidate:
  train trên tập train -> học theta
  chạy trên validation -> tính FIT_sim, FIT_12, FIT_60, robust score

Chọn candidate có validation robust score tốt nhất.

Khóa candidate đã chọn.

Chạy candidate đó trên test để báo cáo kết quả cuối.
```

Vậy đúng là có "một đống model" được chạy thử, nhưng điểm quan trọng là:

```text
validation dùng để chọn model
test không dùng để chọn model
```

Nếu dùng test để chọn model thì test bị lộ. Khi đó kết quả test không còn là đánh giá khách quan nữa.

Ví dụ:

```text
Candidate A:
train xong -> validation robust score = 72

Candidate B:
train xong -> validation robust score = 78

Candidate C:
train xong -> validation robust score = 75
```

Khi đó chọn Candidate B. Sau khi đã chọn B rồi mới chạy B trên test.

Bảng vai trò:

| Tập dữ liệu | Dùng để làm gì? | Có được dùng để chọn model không? |
| --- | --- | --- |
| Train | học hệ số `theta` của từng candidate | không trực tiếp chọn cuối |
| Validation | so sánh các candidate và chọn cấu trúc | có |
| Test | báo cáo kết quả cuối sau khi đã chọn xong | không |

Bình thường có thể chọn model bằng:

```text
val_FIT_sim cao nhất
```

Nhưng cách này có rủi ro. Một mô hình có thể rất tốt ở một đoạn validation nhưng lại rất tệ ở đoạn khác. Với nhà kính, điều này dễ xảy ra vì:

- buổi sáng khác buổi trưa;
- trưa nắng nóng khác chiều mát;
- đoạn bật quạt khác đoạn bật bơm;
- đoạn độ ẩm đất gần setpoint khác đoạn đất đang khô.

Vì vậy project chia validation thành nhiều block thời gian. Mỗi block tính một `FIT_sim` riêng.

Ví dụ chia validation thành 4 block:

```text
Block 1: FIT_sim = 85
Block 2: FIT_sim = 84
Block 3: FIT_sim = 83
Block 4: FIT_sim = 82
```

Mô hình này có:

```text
mean = 83.5
std  ≈ 1.12
```

Một mô hình khác:

```text
Block 1: FIT_sim = 95
Block 2: FIT_sim = 88
Block 3: FIT_sim = 70
Block 4: FIT_sim = 60
```

Mô hình này có đoạn rất cao nhưng dao động mạnh:

```text
mean = 78.25
std  ≈ 13.48
```

Project dùng công thức:

```text
robust_score = mean(block_FIT_sim) - 0.5 * std(block_FIT_sim)
```

Tức là:

```text
điểm robust = điểm trung bình - hình phạt dao động
```

Tính cho hai mô hình:

```text
Model A:
robust_score = 83.5 - 0.5*1.12
             = 82.94

Model B:
robust_score = 78.25 - 0.5*13.48
             = 71.51
```

Vậy chọn Model A vì nó ổn định hơn. Model B có lúc đạt 95 nhưng không đáng tin nếu đem áp dụng thực tế.

Trong code hiện tại, công thức nằm ở logic:

```python
val_robust_score = mean_score - cfg.validation_std_penalty * std_score
```

với:

```text
validation_std_penalty = 0.5
```

Cách hiểu ngắn:

```text
val_FIT_sim cao -> model fit tốt trên validation
std thấp        -> model ổn định giữa các block
robust cao      -> model vừa fit tốt vừa ổn định
```

Câu bảo vệ:

```text
Em không chọn mô hình chỉ dựa vào một giá trị validation tổng. Em chia validation thành nhiều block thời gian, tính FIT_sim từng block, rồi lấy trung bình trừ đi một phần độ lệch chuẩn. Nhờ đó mô hình được chọn không chỉ fit cao mà còn ổn định giữa các điều kiện vận hành khác nhau.
```

## 4. Vì sao không shuffle random?

Với dữ liệu độc lập như ảnh hoặc bảng tĩnh, random split có thể dùng được. Nhưng với chuỗi thời gian, các mẫu gần nhau rất giống nhau.

Nếu shuffle random:

```text
mẫu 10: train
mẫu 11: test
mẫu 12: train
```

thì test không còn thật sự độc lập. Mô hình đã nhìn thấy gần như cùng trạng thái ở train.

Do đó project chia theo thời gian:

```text
70% đầu      -> train
15% tiếp theo -> validation
15% cuối     -> test
```

## 5. Đánh giá công bằng là gì?

Muốn so sánh hai mô hình, phải giữ các điều kiện giống nhau:

```text
cùng dữ liệu
cùng train/validation/test split
cùng input hợp lệ
cùng metric
cùng quy tắc chọn model
```

Không được lấy:

```text
ARX trên data cũ
so với Hybrid trên data mới
```

rồi kết luận Hybrid hơn ARX 17 điểm. Đó là so sánh lẫn cả data và model.

Kết luận công bằng trong project hiện tại là:

```text
Trên cùng dữ liệu mới:
ARX backbone FIT_sim = 82.496%
Hybrid residual FIT_sim = 83.433%
Residual tăng khoảng 0.937 điểm FIT_sim
```

## 6. Vì sao data mới fit cao hơn data cũ?

Data mới không chỉ là "đổi vài số". Data mới là protocol khác:

- lấy mẫu 20 giây thay vì 5 phút;
- phù hợp hơn với mô hình nhà kính nhỏ;
- có kích thích actuator có kiểm soát;
- có audit missing, duplicate, sampling;
- có chia train/validation/test theo thời gian;
- không dùng trạng thái ẩn mô phỏng để train.

Vì vậy khi trình bày phải nói:

```text
Kết quả cao hơn chủ yếu do xây dựng lại protocol dữ liệu phù hợp hơn với hệ nhỏ. Trên cùng data mới, residual chỉ tăng thêm khoảng 0.94 điểm so với ARX backbone.
```

## 7. Test thật khi có dữ liệu cảm biến

Nếu thu dữ liệu thật 1-2 ngày, nên giữ một đoạn cuối làm test thật. Đoạn test này không dùng để:

- chỉnh mô hình;
- chọn `na`, `nb`, `nk`;
- chọn residual;
- quyết định feature;
- chỉnh luật làm giàu dữ liệu.

Nếu có làm giàu dữ liệu, chỉ làm giàu từ train. Không làm giàu test.

## 8. Câu trả lời nhanh khi thầy hỏi "có ăn gian data không?"

Có thể trả lời:

```text
Em tránh leakage bằng cách chia dữ liệu theo thời gian, chỉ fit scaler trên train, chọn cấu trúc mô hình bằng validation robust score và chỉ báo cáo test sau cùng. Các feature chỉ dùng thông tin tại hiện tại hoặc quá khứ. Với dữ liệu làm giàu, em chỉ làm giàu trên train, không làm giàu validation/test để tránh làm đẹp kết quả.
```
