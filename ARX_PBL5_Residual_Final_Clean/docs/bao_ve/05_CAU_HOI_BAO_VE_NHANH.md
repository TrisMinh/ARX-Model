# Câu hỏi bảo vệ nhanh

## 1. Model chính của em là gì?

```text
Model chính là Hybrid ARX residual correction: ARX backbone cộng thêm tầng học residual để sửa sai số free-run.
```

## 2. Có phải ARX thuần không?

```text
Không. ARX là backbone. Model cuối là Hybrid ARX residual correction, vì có thêm residual model.
```

## 3. Vì sao vẫn liên quan đến ARX?

```text
Vì dự đoán nền vẫn do ARX tạo ra. Residual chỉ học phần sai số còn lại của ARX, không thay thế ARX.
```

## 4. Kết quả bao nhiêu?

```text
ARX backbone FIT_sim = 82.50.
Hybrid residual FIT_sim = 83.43.
Mức tăng = 0.937 điểm.
```

## 5. Residual học gì?

```text
Residual học sai số y_true - y_arx_sim bằng các lag hợp lệ của quỹ đạo ARX và input quá khứ.
```

## 6. Có leakage không?

```text
Không. Dữ liệu chia theo thời gian 70/15/15. Scaler fit trên train. Residual train trên train, chọn bằng validation robust score. Test chỉ dùng báo cáo cuối.
```

## 7. Vì sao không chọn candidate test cao hơn?

```text
Vì chọn theo test là leakage lựa chọn model. Em chỉ chọn candidate đứng đầu validation robust score, nên kết quả 83.43 là kết quả hợp lệ.
```

## 8. Dữ liệu này là dữ liệu thật chưa?

```text
Chưa. Đây là dữ liệu mô phỏng vật lý có kiểm soát. Khi có dữ liệu phần cứng thật, em chạy lại cùng pipeline bằng --data-csv và báo cáo kết quả thật riêng.
```

## 9. Ảnh hưởng đến MPC thế nào?

```text
ARX backbone vẫn phù hợp MPC tuyến tính. Hybrid residual có thể dùng như tầng dự báo nâng cao hoặc tầng hiệu chỉnh. Nếu đưa residual vào tối ưu trực tiếp thì bài toán trở thành gần nonlinear MPC hơn, nên em trình bày rõ ranh giới này.
```

## 10. Áp dụng thực tế lấy biến đâu để dự đoán?

```text
Soil_Moisture, Temperature_In, Humidity_In, Light_In lấy từ cảm biến. Drip, Mist, Fan lấy từ lệnh điều khiển mà hệ thống phát ra. Khi dự đoán nhiều bước, Soil_Moisture tương lai do model tự dự đoán, Drip/Mist/Fan tương lai do controller đưa ra dưới dạng chuỗi lệnh giả định/kế hoạch, còn Temperature/Humidity/Light tương lai trong horizon ngắn 4-20 phút được giữ bằng giá trị đo mới nhất. Sau mỗi 20 giây hệ thống đọc cảm biến mới và dự đoán lại.
```

## 11. FIT_sim có phải là kết quả tự tưới không?

```text
Không. FIT_sim là đánh giá mô hình mô phỏng Soil_Moisture dưới chuỗi input đã biết trong test. Nó chưa phải kết quả controller tự chọn lệnh tưới. ARX/Hybrid là mô hình dự đoán, còn quyết định tưới thuộc controller/MPC hoặc rule điều khiển.
```

## 12. Câu kết luận an toàn

```text
Em chọn Hybrid ARX residual correction làm bản kết quả chính vì nó giữ ARX làm backbone nhưng cải thiện free-run simulation từ 82.50 lên 83.43. Em không gọi đây là ARX thuần, và toàn bộ quá trình chọn residual dùng validation, không chọn theo test.
```
