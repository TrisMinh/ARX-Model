# So sánh dữ liệu cũ và dữ liệu mới

File này trả lời câu hỏi: tại sao kết quả trên bộ dữ liệu mới cao hơn bộ dữ liệu cũ khá nhiều, và có được phép nói đó là cải tiến hoàn toàn do mô hình hay không.

Kết luận ngắn: không được nói đơn giản rằng "đổi mô hình nên fit tăng từ 66% lên 83%". Hai con số này nằm trên hai protocol dữ liệu khác nhau. Phần tăng lớn đến từ việc đổi cách tạo/thu dữ liệu cho đúng bài toán nhà kính mini, lấy mẫu nhanh hơn và có kích thích actuator rõ ràng hơn. Cải tiến residual chỉ đóng góp thêm khoảng 0.94 điểm FIT_sim so với ARX backbone trên cùng bộ dữ liệu mới.

## 1. Bảng so sánh chính

| Tiêu chí | Dữ liệu cũ | Dữ liệu mới |
| --- | ---: | ---: |
| File | `_archive_old_versions_20260605/root_files/greenhouse_data.csv` | `ARX_PBL5_Final_Clean/results/mini_greenhouse_20s_data.csv` |
| Số dòng | 105120 | 69120 |
| Thời gian mô phỏng | gần 365 ngày | gần 16 ngày |
| Chu kỳ lấy mẫu | 300 giây, tức 5 phút/mẫu | 20 giây/mẫu |
| Số cột | 12 | 31 |
| Độ lệch chuẩn độ ẩm đất toàn bộ data | 3.324 | 1.306 |
| Khoảng biến thiên độ ẩm đất toàn bộ data | 21.895 | 7.635 |
| Độ lệch chuẩn độ ẩm đất tập test | 2.913 | 0.960 |
| Khoảng biến thiên độ ẩm đất tập test | 17.758 | 4.719 |
| Drip bật trung bình | 22.80% | 2.42% |
| Mist bật trung bình | 7.89% | 14.80% |
| Fan bật trung bình | 16.89% | 42.12% |

## 2. Khác nhau quan trọng nhất

Dữ liệu cũ là bộ legacy theo chu kỳ 5 phút. Với mô hình nhà kính nhỏ 30x50x30 cm, 5 phút là khá thưa vì quạt, phun sương và bơm có thể làm thay đổi môi trường trong vài chục giây đến vài phút. Khi lấy mẫu thưa, nhiều đoạn quá độ bị bỏ qua, nên ARX phải học một hệ đã bị mất thông tin động học.

Dữ liệu mới dùng chu kỳ 20 giây/mẫu. Chu kỳ này hợp lý hơn với mô hình nhỏ vì bắt được phản ứng nhanh sau khi bật/tắt actuator. Vì vậy cùng một dạng ARX, mô hình có nhiều thông tin quá khứ sát thực tế hơn để mô phỏng.

Dữ liệu mới còn có protocol kích thích hệ thống: có các lệnh planned drip, fan, mist theo lịch và có safety override. Mục đích không phải làm giả kết quả, mà là làm giống quy trình nhận dạng hệ thống: muốn học được tác động của actuator thì actuator phải được bật/tắt đủ đa dạng trong vùng an toàn. Nếu chỉ để bộ điều khiển tự giữ ẩm quá ổn định, dữ liệu sẽ ít thông tin để học.

## 3. Vì sao FIT_sim tăng mạnh?

Công thức FIT_sim thường có dạng:

```text
FIT = 100 * (1 - RMSE / std(y_test))
```

Nên FIT không chỉ phụ thuộc vào sai số RMSE, mà còn phụ thuộc vào độ biến thiên của tín hiệu thật trong tập test.

Trên dữ liệu cũ:

```text
ARX cũ test RMSE_sim = 0.976
std(y_test) = 2.913
RMSE / std(y_test) = 0.335
FIT_sim xấp xỉ 66.49%
```

Trên dữ liệu mới:

```text
ARX backbone test RMSE_sim = 0.168
std(y_test) = 0.960
RMSE / std(y_test) = 0.175
FIT_sim xấp xỉ 82.50%
```

Hybrid residual trên dữ liệu mới:

```text
Hybrid test RMSE_sim = 0.159
std(y_test) = 0.960
RMSE / std(y_test) = 0.166
FIT_sim xấp xỉ 83.43%
```

Nói cách khác, trên dữ liệu mới, sai số chuẩn hóa của mô hình giảm từ khoảng 0.335 xuống khoảng 0.166. Đây là lý do toán học trực tiếp làm FIT tăng.

## 4. Có phải do mô hình tốt hơn không?

Có, nhưng phải tách ra hai phần.

Phần 1 là do protocol dữ liệu mới tốt hơn cho bài toán nhà kính mini. Đây là phần chính. Dữ liệu mới có chu kỳ lấy mẫu phù hợp hơn, có kích thích actuator rõ hơn, có chia train/validation/test theo thời gian, có audit missing/duplicate/sampling, và không dùng dữ liệu tương lai.

Phần 2 là do mô hình được làm sạch và chọn cấu trúc tốt hơn. Bản hiện tại dùng ARX backbone được chọn bằng validation robust score, sau đó thêm residual correction rất nhỏ và có shrink để tránh học quá tay. Trên cùng dữ liệu mới, ARX backbone đạt 82.496% FIT_sim, hybrid residual đạt 83.433% FIT_sim. Vậy residual chỉ tăng khoảng 0.937 điểm, không phải tự nó làm tăng 17 điểm.

## 5. Cách trình bày đúng khi bảo vệ

Không nên nói:

```text
Em cải tiến ARX nên từ 66% lên 83%.
```

Nên nói:

```text
Kết quả 66% là trên bộ dữ liệu legacy lấy mẫu 5 phút, không thật sự phù hợp với mô hình nhà kính nhỏ vì bỏ qua nhiều quá độ nhanh. Sau đó em xây dựng lại protocol dữ liệu cho mô hình 30x50x30 cm với chu kỳ 20 giây/mẫu, có kích thích actuator có kiểm soát và chia train/validation/test theo thời gian. Trên protocol mới, ARX backbone đạt 82.50% FIT_sim. Phần residual correction chỉ cải thiện thêm lên 83.43%, nên em xem residual là phần hiệu chỉnh nhỏ sau ARX, không phải thay thế bản chất ARX.
```

## 6. Nếu thầy hỏi có công bằng không?

Câu trả lời là: không dùng hai bộ dữ liệu khác nhau để kết luận mô hình A hơn mô hình B. Hai bộ dữ liệu khác nhau chỉ dùng để chứng minh rằng protocol thu thập dữ liệu ảnh hưởng rất mạnh đến nhận dạng hệ thống.

Muốn so sánh công bằng mô hình, phải chạy các mô hình trên cùng một bộ dữ liệu, cùng cách chia train/validation/test và cùng metric. Trong project hiện tại, kết luận công bằng là:

```text
Trên dữ liệu mới:
ARX backbone: 82.496% FIT_sim
Hybrid residual: 83.433% FIT_sim
Mức tăng do residual: +0.937 điểm FIT_sim
```

Do đó, khi bảo vệ, nên nhấn mạnh rằng thành quả chính là xây dựng lại quy trình dữ liệu đúng hơn cho mô hình nhỏ, còn residual là cải tiến phụ để giảm sai số mô phỏng còn lại.
