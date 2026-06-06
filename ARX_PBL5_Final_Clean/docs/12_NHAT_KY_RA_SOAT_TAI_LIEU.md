# Nhật ký rà soát tài liệu

## 1. Mục tiêu rà soát

Tài liệu phải đạt 4 yêu cầu:

- người mới đọc vào hiểu được bài toán;
- người làm code biết chạy lại pipeline;
- người bảo vệ biết trả lời câu hỏi khó;
- giảng viên đọc không thấy nhập nhằng giữa dữ liệu mô phỏng, dữ liệu thật, ARX thuần và Hybrid residual.

## 2. Vòng rà soát 1: phân biệt phạm vi

Vấn đề phát hiện:

- trước đó dễ hiểu nhầm `78.95` là cải tiến trực tiếp từ `66`;
- bản cảm biến ngoài và baseline cũ làm nhiễu thông điệp;
- residual có thể bị gọi nhầm là ARX thuần.

Chỉnh sửa:

- thêm [08_KHONG_SO_SANH_TRUC_TIEP_CAC_VERSION.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/08_KHONG_SO_SANH_TRUC_TIEP_CAC_VERSION.md>);
- ghi rõ ARX thuần final thời điểm đó `78.950`;
- ghi rõ Hybrid residual thời điểm đó `81.102`;
- ghi rõ dữ liệu hiện tại là mô phỏng vật lý, chưa phải hardware.

## 3. Vòng rà soát 2: người mới đọc

Vấn đề phát hiện:

- thuật ngữ `na`, `nb`, `nk`, `FIT_sim`, `free-run`, `leakage` có thể khó hiểu;
- docs trước đó hơi giống ghi chú kết quả.

Chỉnh sửa:

- viết lại [00_TONG_QUAN_CHO_NGUOI_MOI.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/00_TONG_QUAN_CHO_NGUOI_MOI.md>);
- thêm [11_THUAT_NGU_CHO_NGUOI_MOI.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/11_THUAT_NGU_CHO_NGUOI_MOI.md>);
- thêm [00_MUC_LUC_VA_LO_TRINH_DOC.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/00_MUC_LUC_VA_LO_TRINH_DOC.md>).

## 4. Vòng rà soát 3: phương pháp và leakage

Vấn đề phát hiện:

- cần nói rõ train/validation/test;
- cần nói rõ scaler fit trên train;
- cần nói rõ residual chọn bằng validation, không chọn bằng test.

Chỉnh sửa:

- viết lại [03_BUILD_TRAIN_VALIDATE_TEST.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/03_BUILD_TRAIN_VALIDATE_TEST.md>);
- bổ sung nguyên tắc `shrink=0` trong residual;
- bổ sung câu không báo candidate test cao hơn nếu validation không chọn.

## 5. Vòng rà soát 4: góc nhìn giảng viên

Vấn đề phát hiện:

- tài liệu cần có phần tự phản biện thay vì chỉ báo kết quả;
- cần dự đoán câu hỏi khó.

Chỉnh sửa:

- thêm [09_TU_DANH_GIA_NHU_GIANG_VIEN.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/09_TU_DANH_GIA_NHU_GIANG_VIEN.md>);
- viết lại [06_CAU_HOI_BAO_VE_NHANH.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/06_CAU_HOI_BAO_VE_NHANH.md>);
- thêm [10_DE_CUONG_BAO_CAO_VA_SLIDE_BAO_VE.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Final_Clean/docs/10_DE_CUONG_BAO_CAO_VA_SLIDE_BAO_VE.md>).

## 6. Vòng rà soát 5: dọn bản clean

Vấn đề phát hiện:

- folder bảo vệ dễ bị lộn giữa sản phẩm chính và các thử nghiệm mở rộng;
- docs cũ nhắc quá nhiều nhánh thử nghiệm, làm người đọc khó biết bản nào cần dùng;
- bản clean cần chỉ giữ pipeline ARX final và tài liệu phục vụ bảo vệ.

Chỉnh sửa:

- tách thử nghiệm mở rộng ra folder riêng;
- giữ trong `ARX_PBL5_Final_Clean` chỉ pipeline `arx_final_pipeline.py`;
- chỉnh lại docs để câu chuyện bảo vệ tập trung vào ARX thuần 17 input;
- ghi rõ Hybrid residual là extension, không phải model chính.

## 7. Đánh giá sau rà soát

Tài liệu hiện tại đạt mức tốt cho:

- giải thích phương pháp;
- bảo vệ logic không leakage;
- tách rõ ARX thuần và Hybrid residual;
- hướng dẫn thu dữ liệu thật.

Điểm chưa thể đạt nếu chưa có hardware:

- chưa thể gọi kết quả là thực nghiệm phần cứng;
- chưa thể kết luận model áp dụng thực tế chắc chắn;
- chưa có plot từ dữ liệu thật để kiểm chứng cảm biến và actuator.

## 8. Checklist trước khi nộp

- [x] Có mục lục đọc theo vai trò.
- [x] Có lý thuyết ARX.
- [x] Có quy trình thu dữ liệu thật.
- [x] Có pipeline train/validate/test.
- [x] Có kết quả ARX thuần.
- [x] Có kết quả Hybrid residual.
- [x] Có cảnh báo không so version khác data.
- [x] Có câu hỏi bảo vệ nhanh.
- [x] Có tự đánh giá như giảng viên.
- [ ] Có dữ liệu phần cứng thật.
- [ ] Có kết quả test thật sau khi chạy `--data-csv`.
