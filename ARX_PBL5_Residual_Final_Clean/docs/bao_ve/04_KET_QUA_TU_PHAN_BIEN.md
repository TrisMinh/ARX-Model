# Kết quả và tự phản biện

## 1. Kết quả chính

| Model | FIT_sim | RMSE_sim | Bias_sim |
|---|---:|---:|---:|
| ARX backbone 16 input | 82.496 | 0.1681 | -0.0381 |
| Hybrid residual selected | 83.433 | 0.1591 | -0.0207 |

Model được chọn:

```text
hgb_leaf15_l2_0.01 | shrink = 0.25
```

Mức cải thiện:

```text
FIT_sim +0.937 điểm
RMSE_sim giảm 0.0090
```

## 2. Vì sao không chọn candidate test cao hơn?

Trong `leaderboard.csv` có vài candidate test cao hơn `83.433`. Không chọn chúng vì chúng không đứng đầu theo validation robust score.

Nếu chọn theo test thì đó là leakage trong khâu chọn model. Vì vậy con số bảo vệ hợp lệ là:

```text
FIT_sim = 83.433
```

## 3. Điểm mạnh

- Vẫn có ARX backbone rõ ràng.
- Residual được chọn bằng validation, không chọn bằng test.
- Có `shrink=0` để model không bị ép phải sửa.
- Fit free-run tăng, RMSE giảm.
- Có thể chạy lại bằng CSV thật qua `--data-csv`.

## 4. Điểm yếu phải nói thật

- Đây là dữ liệu mô phỏng vật lý, chưa phải dữ liệu phần cứng.
- Model cuối không còn là ARX thuần.
- Residual dùng model phi tuyến `HistGradientBoostingRegressor`, nên nếu đưa trực tiếp vào MPC sẽ khó hơn ARX tuyến tính.
- Khi có dữ liệu thật phải train/test lại; không được lấy số mô phỏng làm kết quả phần cứng.

## 5. Nếu thầy hỏi ảnh hưởng đến MPC

Trả lời:

```text
ARX backbone vẫn có thể dùng cho MPC tuyến tính. Hybrid residual có thể dùng như bộ dự báo nâng cao hoặc tầng hiệu chỉnh ngoài MPC. Nếu muốn đưa residual vào tối ưu trực tiếp thì bài toán sẽ gần nonlinear MPC hơn, nên trong PBL5 em trình bày rõ ranh giới: ARX phục vụ cấu trúc điều khiển, residual phục vụ nâng chất lượng dự báo.
```

## 6. Kết luận bảo vệ

```text
Nhóm chọn Hybrid ARX residual correction làm bản kết quả chính vì nó cải thiện free-run simulation từ 82.50 lên 83.43 mà vẫn giữ ARX làm backbone. Quy trình chọn model dùng validation robust score và test chỉ báo cáo cuối, nên không có leakage lựa chọn model. Nhóm không gọi đây là ARX thuần mà gọi đúng là Hybrid ARX residual correction.
```
