# Không so sánh trực tiếp các version khác dữ liệu

## 1. Vì sao cần file này?

Trong quá trình làm, có nhiều con số FIT khác nhau: `66`, `69`, `71`, `78.95`, `81.10`. Nếu đặt chúng cạnh nhau mà không nói rõ dữ liệu và cấu trúc model, thầy có thể hỏi:

```text
Em tăng fit do cải tiến model hay do đổi dữ liệu?
```

Đây là câu hỏi đúng. Vì vậy phải trả lời minh bạch.

## 2. Bảng phân loại đúng

| Nhóm | Dữ liệu | Model | FIT_sim | Có được so trực tiếp? |
|---|---|---|---:|---|
| ARX cũ | `greenhouse_data.csv` cũ | ARX `na=5, nb=3, nk=2` | 66.49 | Chỉ so với model trên cùng data cũ |
| Hybrid cũ | `greenhouse_data.csv` cũ | ARX + residual causal | 70.29-71.31 | Có thể so với ARX cũ |
| Diagnostic cũ | `greenhouse_data.csv` cũ | có future actuator | 77.53 | Không production-safe |
| ARX final gọn | data mô phỏng mini 20 giây | ARX thuần `na=12, nb=3, nk=2, alpha=0.1` | 78.95 | Sản phẩm chính |
| Hybrid final | data mô phỏng mini 20 giây | ARX + residual | 81.10 | Hướng mở rộng |

## 3. Câu nào không nên nói?

Không nên nói:

```text
Em cải tiến ARX trực tiếp từ 66 lên 78.95.
```

Câu này thiếu điều kiện vì `66` và `78.95` không cùng dataset/protocol.

## 4. Câu đúng khi bảo vệ

Cách nói chuẩn:

```text
Trên dữ liệu cũ, ARX thuần đạt khoảng 66%. Khi thêm residual correction hợp lệ, kết quả tăng lên khoảng 69-71%. Các bản 78.95% và 81.10% thuộc protocol dữ liệu mới cho mô hình nhỏ lấy mẫu 20 giây, nên không so trực tiếp với 66%. Trên cùng protocol mới, ARX final gọn đạt 78.95% và Hybrid residual đạt 81.10%.
```

## 5. Ý nghĩa học thuật

So sánh model chỉ công bằng khi giữ:

- cùng dữ liệu;
- cùng train/validation/test split;
- cùng chỉ số;
- cùng chính sách chọn model;
- không dùng test để chọn cấu hình.

Nếu một trong các điều kiện thay đổi, kết quả vẫn có giá trị, nhưng phải ghi rõ là một thí nghiệm khác.

## 6. Kết luận cho báo cáo

Trong báo cáo, nên tách thành hai mục:

- **Kết quả trên protocol final:** ARX gọn `78.95`, Hybrid residual `81.10`.
- **Lịch sử phát triển trên dữ liệu cũ:** ARX cũ `66`, residual cũ `69-71`.

Không trộn hai nhóm số vào một bảng duy nhất nếu không có cột `Dữ liệu`.
