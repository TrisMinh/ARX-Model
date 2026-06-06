# Tự phản biện bộ docs build từ đầu

## 1. Mục tiêu tự soi

Docs này phải đủ để một người mới:

- hiểu project gồm những file nào;
- viết được từng hàm theo thứ tự;
- biết sau mỗi hàm phải log gì;
- hiểu output của hàm;
- biết vì sao không chọn theo test;
- chạy lại được pipeline.

## 2. Điểm đã làm được

- Có mục lục build từ đầu.
- Có giải thích cấu trúc folder.
- Có logger `debug_walkthrough.py`.
- Có giải thích từng nhóm hàm trong ARX backbone.
- Có giải thích từng hàm trong residual.
- Có bảng đọc shape và metrics.
- Có cảnh báo leakage.
- Có lệnh chạy với dữ liệu thật.

## 3. Điểm còn chưa hoàn hảo

### 3.1. Chưa biến toàn bộ docs thành tutorial copy-paste từng dòng

Docs giải thích cách viết từng hàm, nhưng không paste lại toàn bộ 1.000 dòng code. Lý do: nếu paste toàn bộ code vào docs, người đọc dễ bị ngợp và docs sẽ khó bảo trì.

Cách bù:

```text
Đọc docs song song với src/arx_backbone_pipeline.py và src/arx_residual_experiment.py.
```

### 3.2. Chưa có hình plot

Hiện docs có log dạng text. Nếu muốn trực quan hơn, có thể thêm script plot:

- y_true vs y_arx;
- y_true vs y_hybrid;
- residual correction;
- leaderboard top 10.

### 3.3. Chưa có unit test nhỏ

Nên thêm test nhỏ cho:

- `build_arx_matrix` shape;
- `fit_metrics`;
- `residual_features` shape;
- split time không shuffle.

## 4. Checklist tự đánh giá

| Tiêu chí | Trạng thái |
|---|---|
| Người mới biết chạy lệnh nào | đạt |
| Người mới hiểu vì sao cần từng file | đạt |
| Từng hàm có mục đích/input/output/log | đạt mức tốt |
| Có log trực quan sau từng bước | đạt |
| Có giải thích leakage | đạt |
| Có tự phản biện | đạt |
| Có plot hình | chưa |
| Có unit test | chưa |

## 5. Câu kết luận

Bộ docs hiện đủ để build lại project theo từng bước và debug bằng log. Nếu cần nâng thêm một mức cho báo cáo/kèm hướng dẫn lập trình, bước tiếp theo nên là thêm plot và unit test nhỏ.
