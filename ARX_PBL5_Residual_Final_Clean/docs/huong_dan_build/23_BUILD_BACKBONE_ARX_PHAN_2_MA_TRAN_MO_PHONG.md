# Build ARX backbone phần 2: ma trận, train và mô phỏng

## 1. `arx_max_lag`

Mục đích: tính số dòng đầu tiên phải bỏ vì chưa đủ dữ liệu quá khứ.

Công thức:

```python
max(spec.na, spec.nb + spec.nk - 1)
```

Ví dụ final:

```text
na=12, nb=6, nk=1
max_lag = max(12, 6+1-1) = 12
```

Log:

```python
log_step("arx_max_lag", arx_max_lag(spec))
```

## 2. `build_arx_matrix`

Mục đích: biến DataFrame time-series thành ma trận hồi quy.

ARX cần dạng:

```text
X @ theta = y
```

Mỗi dòng của `X` gồm:

1. `Soil_Moisture(k-1) ... Soil_Moisture(k-na)`;
2. từng input tại các lag `nk ... nk+nb-1`;
3. một cột toàn 1 làm intercept.

Output:

- `X`: ma trận feature;
- `y`: vector target.

Với 16 input, `na=12`, `nb=6`, số cột:

```text
12 + 16*6 + 1 = 109
```

Log:

```python
x_train, y_train = build_arx_matrix(train_z, spec, INSIDE_INPUT_COLS)
log_step("X train", x_train)
log_step("y train", y_train)
```

Nếu shape không đúng, thường sai ở `na/nb/nk` hoặc thiếu cột input.

## 3. `fit_arx`

Mục đích: học vector hệ số `theta`.

Nếu `alpha <= 0`:

```python
np.linalg.lstsq(X, y)
```

Nếu `alpha > 0`, dùng Ridge:

```text
theta = inv(X.T X + alpha*I) X.T y
```

Không phạt intercept:

```python
penalty[-1, -1] = 0.0
```

Log:

```python
theta = fit_arx(train_z, spec, INSIDE_INPUT_COLS)
log_step("theta", {"len": len(theta), "preview": theta[:8]})
```

## 4. `arx_predict_at`

Mục đích: dự đoán một điểm thời gian `t`.

Hàm cộng lần lượt:

- hệ số output lag;
- hệ số input lag;
- intercept.

Điểm quan trọng:

- nếu one-step, `y_source` là y thật;
- nếu simulation, `y_source` là y mô phỏng đã được cập nhật.

Log không nên gọi quá nhiều vì hàm chạy trong vòng lặp. Khi debug, chỉ gọi một điểm:

```python
y_hat = arx_predict_at(y_source, inputs, t=100, theta=theta, spec=spec)
log_step("predict at 100", y_hat)
```

## 5. `predict_arx_one_step`

Mục đích: dự đoán từng bước bằng y thật quá khứ.

Vì y quá khứ là thật, bài này dễ hơn simulation.

Output:

- `y_pred`;
- `y_true`.

Log:

```python
y_one, yt_one = predict_arx_one_step(test_z, theta, spec, INSIDE_INPUT_COLS, clip)
log_step("one-step metrics", fit_metrics(inverse_y(yt_one, stats), inverse_y(y_one, stats)))
```

## 6. `simulate_arx`

Mục đích: free-run simulation.

Khác one-step:

- sau khi dự đoán `y(k)`, hàm dùng chính dự đoán đó cho các bước sau;
- nếu model sai, sai số có thể tích lũy.

Đây là chỉ số khó và trung thực hơn.

Log:

```python
y_sim, yt_sim = simulate_arx(test_z, theta, spec, INSIDE_INPUT_COLS, clip)
log_step("simulation metrics", fit_metrics(inverse_y(yt_sim, stats), inverse_y(y_sim, stats)))
```

## 7. `simulate_arx_n_step`

Mục đích: dự đoán nhiều bước nhưng có reset theo horizon.

Ví dụ `n_step=60` với sampling 20 giây:

```text
60 bước = 20 phút
```

Hàm này gần bài toán điều khiển hơn free-run dài vì MPC thường dự đoán theo horizon hữu hạn.

Log:

```python
y_60, yt_60 = simulate_arx_n_step(test_z, theta, spec, INSIDE_INPUT_COLS, clip, 60)
log_step("FIT_60", fit_metrics(inverse_y(yt_60, stats), inverse_y(y_60, stats)))
```

## 8. `evaluate_arx`

Mục đích: gom nhiều cách đánh giá vào một dict.

Output có:

- `metrics_1step`;
- `metrics_12`;
- `metrics_sim`;
- `metrics_60` nếu bật `include_control_horizon`.

Log:

```python
eval_test = evaluate_arx(test_z, theta, spec, INSIDE_INPUT_COLS, stats, clip, cfg, True)
log_step("evaluate_arx", eval_test)
```

## 9. `validation_blocks_arx`

Mục đích: kiểm tra model có ổn định trên nhiều đoạn validation không.

Quy trình:

1. Chia validation thành 4 block.
2. Mỗi block chạy `simulate_arx`.
3. Tính FIT từng block.
4. Tính robust score:

```text
mean(FIT block) - 0.5 * std(FIT block)
```

Log:

```python
robust = validation_blocks_arx(val_z, theta, spec, INSIDE_INPUT_COLS, stats, clip, cfg)
log_step("validation blocks", robust)
```

## 10. `arx_specs`

Mục đích: tạo danh sách cấu hình ARX để search.

Hiện tại thử:

```text
na = 12, 18, 24
nb = 3, 6
nk = 1, 2, 3
alpha = 0.1, 1, 10
```

Tổng:

```text
3 * 2 * 3 * 3 = 54 candidate
```

Log:

```python
specs = arx_specs()
log_step("arx_specs", {"n": len(specs), "first": specs[0].name})
```

## 11. `run_arx_search`

Mục đích: chạy toàn bộ candidate và chọn bằng validation robust score.

Các bước:

1. Lặp qua `arx_specs`.
2. `fit_arx` trên train.
3. Dự đoán validation.
4. Tính one-step, sim, robust block.
5. Sắp leaderboard giảm dần theo `val_robust_score`.
6. Lấy dòng đầu làm selected.
7. Đánh giá selected trên validation và test.

Log:

```python
arx = run_arx_search(...)
log_step("selected ARX", arx["selected_by_validation"])
log_step("top leaderboard", arx["leaderboard"].head(5))
```

Nguyên tắc bảo vệ: không chọn theo test.

## 12. `build_fixed_arx`

Mục đích: dựng baseline cố định `ARX(5,1,2)` nếu cần so lịch sử.

Trong bản residual final, hàm này không phải đường chính. Nó hữu ích để hiểu baseline cũ nhưng không dùng làm kết luận.
