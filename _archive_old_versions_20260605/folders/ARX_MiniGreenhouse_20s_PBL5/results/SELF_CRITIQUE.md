# Tự Phản Biện Và Cách Bảo Vệ

## Điểm có thể bị bắt lỗi

- Dữ liệu là simulation, nên không được kết luận thay thế hoàn toàn dữ liệu thật.
- Sampling 20 giây tạo nhiều mẫu gần nhau, nên 1-step fit có thể cao. Vì vậy report thêm 4 phút, 20 phút và free-run.
- Actual Drip vẫn phụ thuộc soil do safety rescue. Đây là thực tế closed-loop, nên log thêm `Planned_Drip` để tách planned excitation với safety override.
- NARX không được tune vô hạn. Nếu tune theo test sẽ leakage. Ở đây NARX chọn bằng validation robust score.
- ARX có engineered features, nên nên gọi là linear ARX với transformed exogenous inputs, không phải ARX raw-only.

## Tại sao kết quả chấp nhận được

- ARX test `FIT_sim = 76.598`, không chỉ đẹp ở 1-step.
- ARX test `FIT_60` 20 phút = `91.904`, phù hợp hơn cho MPC horizon.
- Planned Drip corr với soil margin = `-0.007`, gần độc lập với soil.
- NARX free-run = `-3.637`, kém ARX, nên không có lý do đổi sang NARX trong đồ án ARX/MPC.

## Cách nói trước hội đồng

Ban đầu, hệ chạy rule-based safety để không làm khô/úng đất. Trong giai đoạn nhận dạng, nhóm chèn các pulse nhỏ đã lên lịch trước để tạo persistent excitation. Model ARX được train bằng dữ liệu quá khứ và chọn bằng validation theo thời gian. Test được giữ riêng để đánh giá cuối. Kết quả báo cáo cả 20 giây, 4 phút, 20 phút và free-run để tránh đánh giá ảo do 1-step.
