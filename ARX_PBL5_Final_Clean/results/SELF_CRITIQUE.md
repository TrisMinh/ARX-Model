# Tự Phản Biện Bản ARX Chỉ Cảm Biến Trong

## Những điểm thầy có thể bắt lỗi

- Nếu nói đây là dữ liệu thật thì sai. Đây là dữ liệu mô phỏng vật lý để kiểm thử pipeline và thiết kế thí nghiệm.
- Nếu dùng `Temperature_Air_True` hoặc `Humidity_Air_True` để train thì leakage vì đó là trạng thái ẩn của mô phỏng. Pipeline không đưa các cột này vào input.
- Nếu chọn model theo test thì leakage. Pipeline chọn model bằng validation robust score, sau đó mới báo cáo test.
- Nếu chỉ nhìn `FIT_1step` thì dễ ảo vì sampling 20 giây. Báo cáo phải có `FIT_12`, `FIT_60` và `FIT_sim`.
- Không đưa baseline cũ `ARX(5,1,2)` vào kết luận chính vì baseline đó không đại diện cho pipeline cuối.

## Vì sao bỏ bản có cảm biến ngoài khỏi sản phẩm chính

- Lợi ích FIT_sim của bản ngoài trong lần thử trước gần như bằng 0, nên đưa vào final dễ làm rối lập luận.
- PBL5 cần mô hình rõ, không cần cảm biến ngoài, dễ đưa vào MPC tuyến tính; ARX chỉ cảm biến trong đáp ứng tốt hơn.
- Bản final hiện tại đạt `FIT_sim = 78.950` và `FIT_60 = 88.047`.

## Cách nói khi bảo vệ

Nhóm chọn ARX chỉ cảm biến trong 17 input làm sản phẩm chính vì mô hình đủ tốt, gọn, dễ giải thích và dễ triển khai điều khiển. Các thử nghiệm mở rộng được tách ra folder riêng để không làm lẫn với bản bảo vệ.
