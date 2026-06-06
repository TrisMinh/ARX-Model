# Báo Cáo Cuối: Mini Greenhouse 20 Giây

## Kết luận ngắn

- Đây là bản riêng cho mô hình `30x50x30 cm`, lấy mẫu mỗi `20 giây`.
- Sampling 20 giây hợp lý với thể tích nhỏ vì nhiệt độ và độ ẩm không khí đổi nhanh; độ ẩm đất vẫn được đánh giá bằng các horizon dài hơn.
- Dữ liệu không chỉ được tạo để FIT cao: có commissioning, rule-based safety, planned excitation và validation theo hướng triển khai thật.
- ARX là model chính vì đạt free-run tốt hơn, giải thích được và phù hợp hơn với MPC tuyến tính/RLS.
- NARX được dùng để đối chứng, nhưng không nên thay trực tiếp ARX trong MPC nếu chưa chuyển sang NMPC hoặc local linearization.
- Model ARX tốt nhất hiện tại: `ARX_na18_nb3_nk3_alpha0.01`.

## Kiểm tra dữ liệu

| Hạng mục | Giá trị |
| --- | ---: |
| Thể tích nhà kính m3 | 0.0450 |
| Thời gian lấy mẫu | 20 giây |
| Số dòng dữ liệu | 51840 |
| Tổng missing | 0 |
| Timestamp bị trùng | 0 |
| Số mẫu sai chu kỳ lấy mẫu | 0 |
| Soil min-max | 54.024 - 64.159 |
| Soil std | 1.865 |
| Drip ON % | 2.211 |
| Planned Drip % | 1.215 |
| Safety override % | 0.334 |
| Wet block % | 0.000 |
| P(Drip ON | prev soil below low) | 5.431 |
| P(Drip ON | prev soil safe mid) | 1.134 |
| corr(Drip, prev soil-center) | -0.102 |
| corr(Planned Drip, prev soil-center) | -0.007 |

`Planned_Drip` được sinh theo clock/random seed, không sinh từ soil moisture. `Drip` thực tế vẫn có safety rescue/block nên có thể phụ thuộc soil, giống hệ thật.

## So sánh mô hình

| Trường hợp | Họ mô hình | Model được chọn | FIT_1step 20s | FIT_12 4 phút | FIT_60 20 phút | FIT_sim | RMSE_sim | Ghi chú MPC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ARX baseline order cũ | ARX tuyến tính | `ARX_na5_nb1_nk2_alpha0` | 96.891 | 91.653 | 84.084 | 64.405 | 0.5541 | Baseline đơn giản tương đương order ARX(5,1,2) cũ |
| ARX mini20 robust | ARX tuyến tính | `ARX_na18_nb3_nk3_alpha0.01` | 97.152 | 95.064 | 91.904 | 76.598 | 0.3644 | Phù hợp MPC; tuyến tính theo tham số |
| NARX mini20 đối chứng | NNARX phi tuyến | `Delta_NNARX_64_32_a10` | 97.023 | 94.346 | 86.930 | -3.637 | 1.6139 | Muốn dùng trực tiếp cho điều khiển cần NMPC hoặc local linearization |

## Chi tiết ARX

- Robust validation score được chọn: `76.050`.
- Validation block `FIT_sim`: `[90.139, 65.148, 82.649, 85.08]`.
- Input delay: `60` giây.
- Output memory: `360` giây.
- Input memory: `60` giây.
- Test one-step residual max abs ACF lag 1..60: `0.098`.
- Test free-run residual max abs ACF lag 1..60: `0.989`. Residual free-run có tự tương quan cao hơn vì sai số được tích lũy theo thời gian.

## Chi tiết NARX

- NARX được chọn: `Delta_NNARX_64_32_a10`.
- Chênh lệch `FIT_sim` của NARX so với ARX: `-80.236` điểm.

## Tự phản biện

1. Dữ liệu vẫn là mô phỏng, không được nói là dữ liệu thật. Giá trị của nó là kiểm thử quy trình và tạo baseline trước khi thu trên hardware.
2. Sampling 20 giây hợp lý cho air dynamics, nhưng với soil moisture cần dùng horizon 4 phút, 20 phút và free-run để tránh ảo tưởng 1-step.
3. Planned excitation không phải gian lận. Đây là thiết kế thí nghiệm nhận dạng hệ, có safety override và được log rõ.
4. Nếu thầy yêu cầu thực nghiệm thật: dùng cùng schema log, chạy rule-based trong 1-2 ngày, sau đó chạy planned pulses nhỏ trong các phiên thu tiếp theo, rồi retrain ARX.
5. NARX có thể cần tune sâu hơn, nhưng nếu free-run kém ARX thì không có lý do đổi kiến trúc MPC trong phạm vi PBL5.

## Chạy lại kết quả

```powershell
python -B .\ARX_MiniGreenhouse_20s_PBL5\src\mini20s_pipeline.py
```
