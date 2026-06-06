# Tự Phản Biện Bản Có Cảm Biến Ngoài

## Những điểm thầy có thể bắt lỗi

- Nếu nói đây là dữ liệu thật thì sai. Đây là dữ liệu mô phỏng vật lý để kiểm thử pipeline và thiết kế thí nghiệm.
- Nếu dùng `Temperature_Air_True` hoặc `Humidity_Air_True` để train thì leakage vì đó là trạng thái ẩn của mô phỏng. Pipeline không đưa các cột này vào input.
- Nếu chọn model theo test thì leakage. Pipeline chọn model bằng validation robust score, sau đó mới báo cáo test.
- Nếu chỉ nhìn `FIT_1step` thì dễ ảo vì sampling 20 giây. Báo cáo phải có `FIT_12`, `FIT_60` và `FIT_sim`.
- Nếu thêm quá nhiều feature ngoài nhưng không có cải thiện test thì nên bỏ bớt. Bản này giữ cả mô hình chỉ-cảm-biến-trong để đối chứng.

## Vì sao thêm cảm biến ngoài là hợp lý

- Quạt không chỉ là actuator làm mát; nó tạo trao đổi khối khí giữa trong và ngoài.
- Cùng một lệnh quạt nhưng nếu bên ngoài khô hơn thì đất khô nhanh hơn; nếu bên ngoài ẩm hơn thì tác động khác.
- Cảm biến trong có độ trễ và chỉ đo tại một vị trí, nên biến ngoài giúp giải thích phần nhiễu môi trường trước khi nó phản ánh đầy đủ vào cảm biến trong.
- Trong lần chạy này, thêm biến ngoài chênh `FIT_sim` khoảng `0.003` điểm và chênh `FIT_60` khoảng `0.790` điểm so với ARX chỉ dùng cảm biến trong.

## Cách nói khi bảo vệ

Nhóm không giả định nhà kính là hệ kín. Với mô hình nhỏ, khi quạt bật thì không khí ngoài đi vào và không khí trong đi ra, làm thay đổi tốc độ bay hơi và độ ẩm đất. Vì vậy nhóm đề xuất log thêm `Temperature_Out` và `Humidity_Out` như nhiễu đo được. Để kiểm chứng, nhóm train hai ARX trên cùng dữ liệu và cùng split: một bản chỉ dùng cảm biến trong, một bản có thêm cảm biến ngoài. Model được chọn bằng validation robust score và đánh giá cuối trên test theo thứ tự thời gian.
