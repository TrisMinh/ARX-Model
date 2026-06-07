# One-step, multi-step và free-run simulation

Một lỗi rất hay gặp khi báo cáo mô hình ARX là chỉ đưa `FIT_1step`. Với dữ liệu lấy mẫu nhanh 20 giây, `FIT_1step` thường cao vì mô hình được dùng giá trị độ ẩm thật ngay trước đó. Để đánh giá áp dụng thực tế, phải hiểu ba kiểu dự đoán dưới đây.

## 1. One-step prediction

One-step prediction nghĩa là dự đoán một mẫu tiếp theo, nhưng khi dự đoán mỗi điểm, mô hình được dùng `y` thật trong quá khứ.

Ví dụ dự đoán `y(k)`:

```text
y_hat(k) = f(
  y_true(k-1), y_true(k-2), ...,
  u(k-1), u(k-2), ...
)
```

Điểm mạnh:

- kiểm tra mô hình học quan hệ cục bộ tốt hay không;
- thường ổn định;
- dễ đạt FIT cao.

Điểm yếu:

- có thể quá lạc quan;
- không phản ánh hoàn toàn khi mô hình phải tự chạy nhiều bước.

Với bài điều khiển/MPC, one-step chưa đủ.

## 2. Multi-step prediction

Multi-step prediction nghĩa là mô phỏng nhiều bước trước, ví dụ 12 bước hoặc 60 bước.

Trong project hiện tại:

```text
T_s = 20 giây
12 bước = 240 giây = 4 phút
60 bước = 1200 giây = 20 phút
```

Khi mô phỏng nhiều bước, sau bước đầu tiên, mô hình bắt đầu dùng chính dự đoán của nó làm quá khứ:

```text
y_hat(k+1) dùng y_true(k)
y_hat(k+2) dùng y_hat(k+1)
y_hat(k+3) dùng y_hat(k+2)
...
```

Multi-step quan trọng vì nó gần với tình huống điều khiển hơn one-step.

## 3. Free-run simulation

Free-run simulation là mô phỏng liên tục trên toàn đoạn test. Sau một số mẫu khởi tạo ban đầu, mô hình tự dùng đầu ra dự đoán của mình để chạy tiếp.

Công thức ý tưởng:

```text
ban đầu có vài y_true để khởi tạo
sau đó:
y_hat(k) = f(y_hat(k-1), y_hat(k-2), ..., u quá khứ)
```

Đây là bài kiểm tra khó hơn one-step vì sai số có thể tích lũy.

Nếu mô hình có bias nhỏ nhưng kéo dài, free-run sẽ làm lộ ra. Vì vậy `FIT_sim` thường thấp hơn `FIT_1step`.

## 4. Vì sao FIT_1step cao chưa chắc tốt?

Giả sử độ ẩm đất thay đổi chậm. Nếu `y(k-1)` gần bằng `y(k)`, mô hình chỉ cần học:

```text
y_hat(k) gần y(k-1)
```

là đã có `FIT_1step` cao.

Nhưng khi free-run, nếu mô hình cứ dự đoán hơi lệch một chút, sai số có thể cộng dồn:

```text
sai số nhỏ ở bước 1
sai số lớn hơn ở bước 20
sai số trôi rõ ở bước 200
```

Do đó báo cáo phải có:

```text
FIT_1step
FIT_12
FIT_60
FIT_sim
```

## 5. Cách đọc kết quả project hiện tại

Trong bản residual final hiện tại:

```text
ARX backbone FIT_1step khoảng 95.53%
ARX backbone FIT_12 khoảng 92.10%
ARX backbone FIT_60 khoảng 89.25%
ARX backbone FIT_sim khoảng 82.50%
Hybrid residual FIT_sim khoảng 83.43%
```

Cách đọc đúng:

- one-step rất tốt, nghĩa là quan hệ cục bộ được học tốt;
- multi-step 4 phút và 20 phút vẫn tốt, có ích cho điều khiển ngắn hạn;
- free-run thấp hơn, chứng tỏ khi chạy dài vẫn còn drift;
- residual giảm nhẹ drift, nhưng không làm thay đổi bản chất backbone ARX.

## 6. Câu trả lời nếu thầy hỏi nên báo cáo metric nào

Có thể trả lời:

```text
Em không chỉ báo cáo FIT_1step vì dữ liệu lấy mẫu 20 giây có quán tính lớn nên one-step dễ cao. Em báo cáo thêm FIT_12, FIT_60 và FIT_sim. Trong đó FIT_sim là khó nhất vì mô hình phải tự mô phỏng liên tục, còn FIT_60 hữu ích cho bài toán điều khiển ngắn hạn/MPC.
```
