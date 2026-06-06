# So sánh ARX, Hybrid ARX residual, NARX và NNARX

## 1. Mục tiêu của file

File này dùng để trả lời các câu hỏi:

```text
Vì sao chọn ARX?
Vì sao không dùng NARX/NNARX?
Hybrid ARX residual có khác gì NARX không?
Nếu residual không được validation chọn thì nên kết luận thế nào?
```

## 2. Bảng phân biệt nhanh

| Model | Bản chất | Ưu điểm | Nhược điểm | Vai trò trong project |
|---|---|---|---|---|
| ARX thuần 17 input | tuyến tính theo tham số | dễ giải thích, dễ đưa vào MPC tuyến tính | fit thấp hơn bản mở rộng | sản phẩm chính |
| Hybrid ARX residual | ARX backbone + model sửa sai số | tăng FIT từ 78.95 lên 81.10 | không còn ARX thuần | hướng mở rộng |
| NARX đa thức | hồi quy phi tuyến theo biến trễ | học được tương tác phi tuyến | dễ nhiều số hạng, overfit | đối chứng |
| NNARX | neural network với input/output lag | có thể fit cao nếu dữ liệu nhiều | khó giải thích, cần data nhiều, MPC khó hơn | hướng nghiên cứu sau |

## 3. ARX thuần là gì?

ARX thuần có dạng:

```text
y(k) = f_tuyến_tính(y(k-1..k-na), u(k-nk..))
```

Trong project:

```text
ARX_na12_nb3_nk2_alpha0.1
FIT_sim = 78.950
```

Đây là sản phẩm chính vì:

- dễ bảo vệ;
- phù hợp đề tài ARX;
- đủ tốt trên protocol mô phỏng;
- thuận lợi cho MPC tuyến tính.

## 4. Hybrid ARX residual là gì?

Hybrid residual dùng ARX làm nền:

```text
y_arx = ARX(...)
```

Sau đó học sai số còn lại:

```text
residual = y_true - y_arx
```

Và dự đoán cuối:

```text
y_hybrid = y_arx + shrink * residual_model(features)
```

Trong project:

```text
Hybrid ARX residual FIT_sim = 81.102
```

Nó không phải NARX/NNARX, nhưng cũng không còn là ARX thuần. Tên chính xác:

```text
Hybrid ARX residual correction
```

## 5. NARX/NNARX khác gì?

NARX tổng quát hơn ARX vì hàm dự đoán có thể phi tuyến:

```text
y(k) = F(y quá khứ, u quá khứ)
```

Nếu `F` là đa thức, ta có NARX đa thức.

Nếu `F` là neural network, ta có NNARX.

NNARX có thể đạt fit rất cao trong tài liệu khi:

- data nhiều;
- input kích thích đủ;
- nhiễu thấp;
- train/validation/test cùng phân phối;
- chỉ số đánh giá là prediction ngắn hạn hoặc bài toán phù hợp.

Nhưng với PBL5:

- dữ liệu thật dự kiến ít;
- cần giải thích dễ;
- cần liên kết với MPC;
- cần tránh overfit;

nên ARX thuần vẫn hợp lý làm sản phẩm chính.

## 6. Ảnh hưởng tới MPC

ARX thuần:

- dễ viết dự đoán nhiều bước;
- gần tuyến tính;
- dễ đưa vào MPC tuyến tính;
- dễ ràng buộc actuator.

Hybrid ARX residual:

- vẫn có ARX backbone;
- nhưng residual model có thể phi tuyến;
- nếu đưa vào MPC cần xử lý như nonlinear correction hoặc dùng nó như mô-đun dự báo ngoài;
- khó chứng minh hơn ARX thuần.

NNARX:

- bộ dự đoán phi tuyến rõ ràng;
- thường cần nonlinear MPC hoặc tuyến tính hóa;
- khó hơn cho đồ án PBL5 nếu chưa có nền điều khiển đủ mạnh.

## 7. Kết quả nội bộ nên trình bày thế nào?

Nên trình bày:

| Thí nghiệm | Dữ liệu | Model | FIT_sim |
|---|---|---|---:|
| Final ARX | protocol mô phỏng 20 giây | ARX thuần 17 input | 78.950 |
| Final residual | cùng protocol mô phỏng 20 giây | Hybrid ARX residual | 81.102 |
| Data cũ | `greenhouse_data.csv` | ARX cũ | 66.49 |
| Data cũ | `greenhouse_data.csv` | Hybrid residual causal | 69-71 |

Không nên nói:

```text
ARX cải tiến trực tiếp từ 66 lên 78.95
```

Vì hai con số này không cùng dataset/protocol. Cách nói đúng là ARX cũ đạt khoảng 66 trên dữ liệu cũ, còn ARX final đạt 78.95 trên protocol mô phỏng 20 giây mới.

## 8. Câu trả lời khi thầy hỏi

```text
Nhóm chọn ARX thuần 17 input làm sản phẩm chính vì phù hợp phạm vi đề tài, dễ giải thích và thuận lợi cho MPC tuyến tính. Nhóm có thử Hybrid ARX residual, giúp FIT_sim tăng từ 78.95 lên 81.10 trên cùng protocol, nhưng trình bày như hướng mở rộng vì không còn là ARX thuần. NARX/NNARX là hướng có tiềm năng khi dữ liệu thật nhiều hơn, nhưng hiện tại rủi ro overfit và khó tích hợp điều khiển hơn.
```

## 9. Tài liệu tham khảo

- MathWorks ARX: https://www.mathworks.com/help/ident/ref/arx.html
- MathWorks Nonlinear ARX: https://www.mathworks.com/help/ident/ref/nlarx.html
- MathWorks Nonlinear ARX overview: https://www.mathworks.com/help/ident/nonlinear-arx-models.html
- Narendra và Parthasarathy, neural network cho nhận dạng và điều khiển hệ động học: https://pubmed.ncbi.nlm.nih.gov/18282820/
