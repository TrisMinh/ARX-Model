# Thuật ngữ cho người mới

## ARX

ARX là viết tắt của:

```text
AutoRegressive with eXogenous input
```

Nghĩa là dự đoán output bằng output quá khứ và input bên ngoài.

## Backbone

Backbone là model nền. Trong folder này, backbone là ARX 16 input.

## Residual

Residual là sai số:

```text
residual = y_true - y_pred
```

Nếu residual còn có quy luật, ta có thể học quy luật đó để sửa dự đoán.

## Hybrid ARX residual

Là model lai:

```text
y_hybrid = y_arx + shrink * residual_model(features)
```

Nền vẫn là ARX, nhưng có thêm tầng sửa sai số.

## Shrink

`shrink` là hệ số làm mềm correction. Nếu `shrink=0.25`, residual model chỉ sửa 25% mức nó dự đoán.

Lý do cần shrink:

- tránh sửa quá tay;
- giảm overfit;
- cho validation chọn mức sửa hợp lý.

## Leakage

Leakage là khi thông tin không được phép lọt vào train hoặc khâu chọn model.

Ví dụ sai:

- dùng dữ liệu test để chọn model;
- dùng `Soil_Moisture` tương lai làm input;
- fit scaler trên toàn bộ dữ liệu.

## FIT_sim

`FIT_sim` là điểm fit khi model chạy free-run dài. Đây là chỉ số khó hơn one-step vì model phải tự đi theo quỹ đạo dự đoán.

## Validation robust score

Điểm chọn model:

```text
mean(block FIT_sim) - 0.5 * std(block FIT_sim)
```

Điểm này ưu tiên model vừa fit tốt vừa ổn định giữa các đoạn validation.
