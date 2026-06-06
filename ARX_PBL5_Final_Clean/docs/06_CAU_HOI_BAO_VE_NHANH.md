# Câu hỏi bảo vệ nhanh

## 1. Đề tài của em làm gì?

```text
Em xây dựng mô hình ARX dự đoán độ ẩm đất cho mô hình nhà kính nhỏ 30x50x30 cm. Mục tiêu là tạo model dự đoán để sau này tích hợp vào điều khiển tưới/MPC.
```

## 2. Model chính là gì?

```text
Model chính là ARX thuần: ARX_na12_nb3_nk2_alpha0.1.
```

Kết quả:

```text
FIT_sim = 78.95
FIT_60  = 88.05
```

## 3. Vì sao chọn ARX?

```text
Vì ARX dễ giải thích, cần ít dữ liệu hơn neural network, có thể kiểm tra leakage rõ ràng và phù hợp hơn để đưa vào MPC tuyến tính.
```

## 4. `na=12`, `nb=3`, `nk=2` nghĩa là gì?

Với sampling `20 giây`:

```text
na=12 -> nhớ output 4 phút
nb=3  -> nhớ input 1 phút
nk=2  -> input trễ 40 giây
```

## 5. Dữ liệu này có phải dữ liệu thật không?

```text
Chưa. Đây là dữ liệu mô phỏng vật lý có kiểm soát để kiểm tra pipeline. Khi có dữ liệu phần cứng thật, em chạy lại cùng pipeline bằng tham số --data-csv và báo cáo test thật riêng.
```

## 6. Có leakage không?

```text
Pipeline chia theo thời gian 70/15/15. Scaler chỉ fit trên train. Validation dùng để chọn model. Test chỉ dùng báo cáo cuối. Free-run simulation không dùng output tương lai.
```

## 7. Vì sao không so trực tiếp 66 và 78.95?

```text
Vì 66 là trên dữ liệu cũ, còn 78.95 là trên protocol dữ liệu mới 20 giây/mẫu. So sánh công bằng phải cùng dataset. Trên cùng protocol mới, ARX thuần là 78.95 và Hybrid residual là 81.10.
```

## 8. Residual là gì?

```text
Residual là sai số còn lại của ARX: residual = y_true - y_arx. Em thử model phụ học residual này để sửa dự đoán ARX.
```

## 9. Residual có tăng không?

```text
Có. Trên cùng protocol mới, ARX thuần 17 input đạt 78.95 FIT_sim, Hybrid ARX residual chọn bằng validation đạt 81.10 FIT_sim.
```

## 10. Nếu residual tăng thì sao không chọn làm model chính?

```text
Vì model đó không còn là ARX thuần. Nó là ARX backbone cộng thêm model học sai số. Em giữ ARX thuần 17 input làm sản phẩm chính để đúng phạm vi đề tài và dễ tích hợp MPC tuyến tính; residual là hướng mở rộng.
```

## 11. Vì sao không dùng NNARX?

```text
NNARX có thể mạnh khi dữ liệu nhiều, nhưng với PBL5 và dữ liệu thật dự kiến ít, rủi ro overfit và khó giải thích cao hơn. Ngoài ra NNARX làm MPC phức tạp hơn.
```

## 12. Câu kết luận ngắn

```text
ARX thuần 17 input là sản phẩm chính, đạt FIT_sim 78.95 trên dữ liệu mô phỏng vật lý 20 giây/mẫu. Hybrid ARX residual đạt 81.10 nhưng chỉ là hướng mở rộng vì không còn là ARX thuần. Bước tiếp theo là thu dữ liệu phần cứng thật và chạy lại cùng pipeline không leakage.
```
