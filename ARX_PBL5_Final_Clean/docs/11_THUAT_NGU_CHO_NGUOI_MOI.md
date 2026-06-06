# Thuật ngữ cho người mới

## ARX

ARX là viết tắt của `AutoRegressive with eXogenous input`. Nghĩa là model dùng:

- quá khứ của chính đầu ra;
- quá khứ của các đầu vào ngoài;

để dự đoán đầu ra hiện tại.

## Output

Trong bài này output là:

```text
Soil_Moisture
```

Tức độ ẩm đất.

## Input

Input là các biến được dùng để dự đoán, ví dụ:

- `Temperature_In`;
- `Humidity_In`;
- `Light_In`;
- `Drip`;
- `Mist`;
- `Fan`.

## `na`

Số mẫu quá khứ của output. Nếu `na=12` và sampling `20 giây`, model nhớ:

```text
12 * 20 = 240 giây = 4 phút
```

## `nb`

Số mẫu quá khứ của mỗi input. Nếu `nb=3`, model dùng 3 mẫu input theo delay `nk`.

## `nk`

Độ trễ input. Nếu `nk=2`, input bắt đầu ảnh hưởng sau:

```text
2 * 20 = 40 giây
```

## Regularization

Regularization là cách phạt hệ số quá lớn để model ít nhạy hơn với nhiễu. Trong code dùng `alpha` của Ridge.

## FIT

Chỉ số phần trăm cho biết dự đoán tốt hơn trung bình bao nhiêu:

```text
FIT = 100 * (1 - norm(error) / norm(y_true - mean(y_true)))
```

FIT cao hơn là tốt hơn. FIT âm nghĩa là dự đoán còn tệ hơn việc lấy giá trị trung bình làm dự đoán.

## One-step prediction

Dự đoán từng bước một, mỗi bước được hỗ trợ bởi giá trị thật gần nhất. Chỉ số này thường cao, nhưng dễ làm ta lạc quan quá mức.

## Free-run simulation

Model tự dự đoán liên tục, dùng dự đoán trước đó làm đầu vào cho bước sau. Đây là bài kiểm tra khó hơn và sát với điều khiển hơn.

## FIT_60

Dự đoán trong 60 bước. Với sampling `20 giây`:

```text
60 bước = 20 phút
```

## Leakage

Leakage là dùng thông tin không được phép, ví dụ dùng dữ liệu test để chọn model hoặc dùng output tương lai để dự đoán hiện tại.

## Validation

Tập dùng để chọn model. Không phải test.

## Test

Tập dùng để báo cáo kết quả cuối sau khi đã chọn model. Không được dùng test để chọn cấu hình.

## Residual

Residual là sai số:

```text
residual = y_true - y_pred
```

Hybrid ARX residual dùng model phụ để học phần sai số còn lại của ARX.

## Backbone

Backbone là model nền. Trong Hybrid ARX residual, backbone là ARX.

## Hybrid ARX residual

Model kết hợp:

```text
y_hybrid = y_arx + shrink * residual_model(features)
```

Nền vẫn là ARX, nhưng có thêm tầng sửa sai số, nên không gọi là ARX thuần.

## MPC

MPC là `Model Predictive Control`: bộ điều khiển dùng model dự đoán tương lai để chọn lệnh điều khiển hiện tại.

