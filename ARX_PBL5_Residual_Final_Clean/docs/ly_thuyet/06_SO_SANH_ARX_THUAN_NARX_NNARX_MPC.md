# So sánh ARX thuần, Hybrid residual, NARX, NNARX và MPC

## 1. Bảng phân biệt

| Model | Bản chất | Ưu điểm | Nhược điểm | Vai trò |
|---|---|---|---|---|
| ARX thuần | tuyến tính theo tham số | dễ giải thích, dễ đưa vào MPC tuyến tính | FIT_sim thấp hơn | backbone |
| Hybrid ARX residual | ARX + model học sai số | fit cao hơn, vẫn giữ nền ARX | không còn ARX thuần | bản bảo vệ chính |
| NARX đa thức | phi tuyến theo biến trễ | học tương tác phi tuyến | dễ nhiều tham số | đối chứng học thuật |
| NNARX | mạng neural với lag | có thể fit cao khi data nhiều | khó giải thích, cần data nhiều | hướng sau |

## 2. Vì sao không chỉ dùng ARX thuần?

ARX thuần đạt:

```text
FIT_sim = 82.496
```

Đây là mức tốt, nhưng free-run residual còn có cấu trúc. Residual correction cải thiện thành:

```text
FIT_sim = 83.433
```

Vì vậy nếu ưu tiên kết quả dự báo, Hybrid residual hợp lý hơn.

## 3. Vì sao không gọi Hybrid residual là NARX?

NARX học trực tiếp:

```text
y(k) = F(y quá khứ, u quá khứ)
```

Hybrid residual học theo hai tầng:

```text
y_arx = ARX(...)
y_final = y_arx + residual_model(...)
```

Do đó tên chính xác là:

```text
Hybrid ARX residual correction
```

## 4. Vì sao chưa chọn NNARX?

NNARX có thể mạnh hơn khi:

- dữ liệu thật nhiều;
- sensor ổn định;
- input kích thích đủ rộng;
- có thời gian tune hyperparameter;
- có cách tích hợp điều khiển phi tuyến.

Trong PBL5, dữ liệu thật dự kiến ít và cần giải thích rõ. Hybrid residual là bước trung gian hợp lý hơn: tốt hơn ARX thuần nhưng vẫn giữ backbone dễ giải thích.

## 5. Liên hệ MPC

Nếu làm MPC tuyến tính đơn giản:

- dùng ARX backbone để dự đoán trong tối ưu;
- dùng residual như cảnh báo hoặc hiệu chỉnh ngoài vòng tối ưu.

Nếu muốn dùng Hybrid residual trực tiếp trong tối ưu:

- bài toán gần nonlinear MPC;
- cần thêm thời gian và kiến thức điều khiển phi tuyến.

Với đồ án môn học, cách trình bày hợp lý là:

```text
ARX backbone phục vụ cấu trúc điều khiển. Hybrid residual phục vụ nâng chất lượng dự báo và là hướng mở rộng khi triển khai thực tế.
```
