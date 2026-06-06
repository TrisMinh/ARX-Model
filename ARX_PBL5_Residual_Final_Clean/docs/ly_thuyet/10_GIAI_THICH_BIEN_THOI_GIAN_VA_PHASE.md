# Giải thích biến thời gian và Phase

## 1. `Hour_sin` và `Hour_cos` là gì?

Đây là cách mã hóa giờ trong ngày bằng hai biến tuần hoàn:

```text
Hour_sin = sin(2*pi*hour/24)
Hour_cos = cos(2*pi*hour/24)
```

Không dùng trực tiếp `hour = 0..23` vì 23:59 và 00:00 thực tế rất gần nhau, nhưng nếu dùng số thẳng thì một bên là 23.98, một bên là 0.00. Dùng sin/cos giúp model hiểu đây là vòng tròn 24 giờ.

Ý nghĩa vật lý:

- sáng, trưa, chiều, tối có nhiệt, ánh sáng và tốc độ khô khác nhau;
- mô hình nhỏ chịu ảnh hưởng mạnh theo chu kỳ ngày;
- biến này tính được từ timestamp, không phải dữ liệu tương lai.

Kết luận: `Hour_sin/cos` là biến hợp lệ. Nó không phải cảm biến bắt buộc, nhưng có logic vật lý.

## 2. `Day_sin` và `Day_cos` là gì?

Đây là cách mã hóa ngày theo chu kỳ 7 ngày:

```text
Day_sin = sin(2*pi*day_index/7)
Day_cos = cos(2*pi*day_index/7)
```

Trong dữ liệu mô phỏng 16 ngày, biến này giúp model học biến thiên chậm theo ngày:

- lệch nhiệt/ngày;
- mây/ánh sáng từng ngày;
- thay đổi nền khô/ẩm chậm.

Với dữ liệu thật chỉ 1 đến 2 ngày, `Day_sin/cos` có thể không mạnh vì số ngày ít. Tuy nhiên nó vẫn tính được từ timestamp và không gây leakage.

## 3. `Phase_identification` là gì?

Trước đây pipeline tạo:

```text
Phase_identification = 1 nếu Protocol_Phase == identification_safe_excitation
Phase_identification = 0 nếu không
```

Đây là nhãn giai đoạn trong dữ liệu mô phỏng:

- commissioning;
- identification;
- deployment validation.

Nó không phải cảm biến thực tế. Khi chạy phần cứng thật, ta thường không có một biến vật lý tương đương. Vì vậy dùng nó trong model bảo vệ dễ bị hỏi:

```text
Ngoài thực tế em lấy biến này từ đâu?
```

## 4. Kết quả ablation

Đã thử lại bằng cùng pipeline residual, không chọn theo test để train model, chỉ dùng test để đánh giá sau khi chạy:

| Cấu hình | ARX FIT_sim | Hybrid FIT_sim | Nhận xét |
|---|---:|---:|---|
| Bản cũ có `Phase_identification` | 78.950 | 81.102 | phase làm lệch phân phối train/test |
| Bỏ `Phase_identification`, giữ `Hour/Day` | 82.496 | 83.433 | bản final hiện tại |
| Bỏ cả `Hour/Day/Phase` | 77.352 | 82.040 | mất thông tin chu kỳ thời gian |
| Bỏ riêng `Day_sin/cos` | 77.218 | 81.179 | giảm rõ |
| Bỏ riêng `Hour_sin/cos` | 78.019 | 81.395 | validation ổn nhưng test giảm |

## 5. Quyết định cuối

Giữ:

```text
Hour_sin, Hour_cos, Day_sin, Day_cos
```

Bỏ:

```text
Phase_identification
```

Lý do:

- `Hour/Day` tính được từ timestamp, có ý nghĩa vật lý, không phải leakage;
- `Phase_identification` là nhãn protocol mô phỏng, không phải cảm biến thực tế;
- bỏ `Phase_identification` làm kết quả tốt hơn và lập luận bảo vệ sạch hơn.

## 6. Câu trả lời khi thầy hỏi

```text
Hour_sin/cos và Day_sin/cos là biến mã hóa chu kỳ thời gian từ timestamp, giúp model biết sáng-trưa-chiều-tối và biến thiên chậm theo ngày. Chúng không dùng tương lai nên không leakage. Riêng Phase_identification là nhãn protocol mô phỏng, không phải biến đo thực tế, nên em đã bỏ khỏi bản final.
```
