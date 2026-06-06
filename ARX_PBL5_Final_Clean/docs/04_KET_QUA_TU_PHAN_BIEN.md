# Kết quả và tự phản biện

## 1. Kết quả ARX thuần final

Model chính:

```text
ARX_na12_nb3_nk2_alpha0.1
```

Thông số:

| Tham số | Giá trị |
|---|---:|
| `na` | 12 |
| `nb` | 3 |
| `nk` | 2 |
| `alpha` | 0.1 |
| số input | 17 |
| số tham số | 64 |
| output memory | 240 giây |
| input memory | 60 giây |
| input delay | 40 giây |

Kết quả test:

| Chỉ số | Giá trị |
|---|---:|
| FIT_1step | 95.373 |
| FIT_12 | 90.839 |
| FIT_60 | 88.047 |
| FIT_sim | 78.950 |
| RMSE_sim | 0.2021 |
| MAE_sim | 0.1687 |
| Bias_sim | -0.1231 |

## 2. Vì sao chọn bản gọn làm sản phẩm chính?

Với PBL5, bản 17 input dễ giải thích hơn và sát mục tiêu ARX hơn:

- ít feature phụ hơn;
- dễ viết trong báo cáo;
- dễ đưa vào MPC tuyến tính;
- ít bị hỏi vì sao thêm quá nhiều biến;
- vẫn đạt `FIT_60 = 88.047` cho horizon 20 phút.

Vì vậy chọn:

```text
ARX 17 input là sản phẩm chính.
Hybrid residual là thử nghiệm mở rộng.
```

## 3. Kết quả Hybrid ARX residual

Residual được chạy trên ARX backbone 17 input.

Model được validation chọn:

```text
hgb_leaf15_l2_0.01 | shrink = 0.25
```

Kết quả test:

| Model | FIT_sim | RMSE_sim | Bias_sim |
|---|---:|---:|---:|
| ARX thuần 17 input | 78.950 | 0.2021 | -0.1231 |
| Hybrid ARX residual | 81.102 | 0.1814 | -0.1046 |

Mức cải thiện:

```text
+2.151 điểm FIT_sim
RMSE giảm 0.0206
```

## 4. Vì sao không chọn Hybrid làm sản phẩm chính?

Hybrid residual có tăng FIT, nhưng không còn là ARX thuần. Nó là:

```text
ARX backbone + model học sai số
```

Vì đề tài trọng tâm là ARX và có thể liên quan MPC tuyến tính, nên sản phẩm chính vẫn nên là ARX thuần. Hybrid dùng để chứng minh nhóm có thử hướng nâng cao và biết tự phản biện.

## 5. Tự phản biện về dữ liệu

Dữ liệu hiện tại:

| Hạng mục | Giá trị |
|---|---:|
| số dòng | 69.120 |
| thời gian | 2026-01-01 đến 2026-01-16 |
| sampling | 20 giây |
| missing | 0 |
| timestamp trùng | 0 |
| soil mean | 55.644 |
| soil std | 1.306 |
| drip on | 2.418% |
| fan on | 42.124% |
| mist on | 14.802% |

Điểm yếu cần nói rõ:

- đây là dữ liệu mô phỏng vật lý, chưa phải dữ liệu phần cứng;
- khi có dữ liệu thật phải train/test lại;
- không được lấy chỉ số mô phỏng làm kết quả thực nghiệm cuối.

## 6. Kết luận nên dùng

```text
Bản ARX thuần 17 input đạt FIT_sim 78.95 và FIT_60 88.05 trên dữ liệu mô phỏng vật lý 20 giây/mẫu. Nhóm có thử Hybrid ARX residual đạt 81.10, nhưng giữ bản 17 input làm sản phẩm chính vì gọn, dễ giải thích và phù hợp MPC tuyến tính. Khi có dữ liệu phần cứng thật, phải chạy lại cùng pipeline.
```
