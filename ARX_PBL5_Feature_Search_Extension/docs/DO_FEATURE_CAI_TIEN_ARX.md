# Dò feature cải thiện ARX

## 1. Mục tiêu

Mục tiêu của bước này là tìm thêm biến đầu vào giúp ARX mô phỏng độ ẩm đất tốt hơn, nhưng vẫn giữ các nguyên tắc:

- không dùng dữ liệu tương lai;
- không dùng `Soil_Moisture_True`;
- không dùng trạng thái ẩn mô phỏng;
- không dùng cảm biến ngoài;
- không chọn feature theo test.

Script dùng để chạy:

```powershell
python -B .\ARX_PBL5_Feature_Search_Extension\src\arx_feature_search.py
```

Kết quả lưu tại:

- [SUMMARY.md](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Feature_Search_Extension/results/SUMMARY.md>)
- [metrics.json](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Feature_Search_Extension/results/metrics.json>)
- [single_group_leaderboard.csv](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Feature_Search_Extension/results/single_group_leaderboard.csv>)
- [greedy_steps.csv](<C:/Users/minht/OneDrive/Desktop/ARX-Model/ARX_PBL5_Feature_Search_Extension/results/greedy_steps.csv>)

## 2. Vì sao thêm feature vẫn là ARX?

ARX có dạng tổng quát:

```text
y(k) = a1*y(k-1) + ... + b1*u(k-nk) + ...
```

Nếu ta thay `u` bằng một tập input rộng hơn, ví dụ:

```text
u = [Temperature_In, Humidity_In, Drip, Fan, Fan_Roll_3min, Mist_x_VPD]
```

thì model vẫn tuyến tính theo tham số:

```text
y(k) = theta1*y(k-1) + theta2*Fan_Roll_3min(k-2) + theta3*Mist_x_VPD(k-2) + ...
```

Điểm khác là input được làm giàu bằng biến vật lý. Đây là ARX với feature engineering, không phải NNARX.

## 3. Nhóm feature đã thử

Script thử các nhóm:

| Nhóm | Ví dụ feature | Nhận xét |
|---|---|---|
| cạnh bật actuator | `Drip_Start`, `Mist_Start`, `Fan_Start` | hợp lý vì thời điểm vừa bật thường gây đáp ứng khác với đang bật lâu |
| rolling actuator | `Fan_Roll_1min`, `Fan_Roll_3min`, `Fan_Roll_5min` | hợp lý vì trao đổi khí có tính tích lũy |
| tương tác phun sương | `Mist_x_Indoor_Dryness`, `Mist_x_TempIn`, `Mist_x_VPD` | hợp lý vì phun sương tác động mạnh hơn khi không khí khô/nóng |
| tương tác tưới | `Drip_x_VPD`, `Drip_x_Light_log` | test có thể tăng nhưng validation không chọn |
| dạng phi tuyến sensor | `VPD_sq`, `Light_log_sq` | không thắng validation |
| tương tác thời gian | `VPD_x_Hour_sin`, `LightLog_x_Hour_cos` | không thắng validation |

## 4. Cách chọn để tránh leakage

Quy trình:

1. Chia dữ liệu theo thời gian 70/15/15.
2. Fit scaler bằng train.
3. Dò feature bằng validation robust score.
4. Chỉ sau khi chọn xong mới đọc test để báo cáo.

Tiêu chí:

```text
validation robust score = mean(block FIT_sim) - 0.5 * std(block FIT_sim)
```

Ý nghĩa:

- `mean(block FIT_sim)` cao là tốt;
- `std(block FIT_sim)` cao nghĩa là model không ổn định giữa các đoạn validation;
- trừ `0.5 * std` để ưu tiên model ít chập chờn hơn.

## 5. Kết quả feature search

Baseline, cũng là bản ARX chính hiện tại:

| Model | Input | FIT_sim | FIT_60 | Validation robust |
|---|---:|---:|---:|---:|
| `ARX_na12_nb3_nk2_alpha0.1` | 17 | 78.950 | 88.047 | 75.270 |

Bản ứng viên nâng cao sau feature search:

| Model | Input | FIT_sim | FIT_60 | Validation robust |
|---|---:|---:|---:|---:|
| `ARX_na18_nb6_nk2_alpha10` | 26 | 83.296 | 89.326 | 77.651 |

Mức tăng:

```text
FIT_sim +4.346 điểm
FIT_60 +1.279 điểm
RMSE_sim giảm từ 0.2021 xuống 0.1604
```

## 6. Feature của ứng viên nâng cao

Ứng viên nâng cao gồm 26 cột:

```text
Temperature_In
Humidity_In
Light_In
Drip
Mist
Fan
Drip_Start
Mist_Start
Fan_Start
Light_log
TempIn_x_HumiIn
TempIn_x_Light
HumiIn_x_Light
Indoor_Dryness
VPD_Proxy_In
Mist_x_Indoor_Dryness
Mist_x_TempIn
Mist_x_VPD
Drip_x_Indoor_Dryness
Drip_x_Light_log
Drip_x_VPD
Hour_sin
Hour_cos
Day_sin
Day_cos
Phase_identification
```

## 7. Tự phản biện

Điểm mạnh:

- feature có ý nghĩa vật lý;
- tính được online khi chạy thật;
- không cần thêm cảm biến ngoài;
- cải thiện cả validation robust và test;
- residual không còn cần thiết sau khi ARX được làm giàu.

Điểm cần nói rõ:

- kết quả vẫn là trên dữ liệu mô phỏng vật lý;
- khi dùng dữ liệu thật, phải train/test lại;
- nếu actuator log sai thời điểm, các feature `Start` và `Roll` sẽ yếu đi;
- không nên tăng feature vô hạn vì sẽ làm model khó giải thích.

## 8. Câu nói khi bảo vệ

```text
Sau khi kiểm tra ablation, nhóm không giữ các biến không ảnh hưởng. Nhóm dò thêm các feature có ý nghĩa vật lý và chỉ chọn bằng validation robust score. Các feature được giữ đều có thể tính online từ cảm biến trong và actuator, không dùng tương lai. Nhờ vậy tạo được một ứng viên ARX 26 input tăng từ FIT_sim 78.95 lên 83.30 mà vẫn giữ bản chất ARX tuyến tính theo tham số. Tuy nhiên bản chính để bảo vệ vẫn là ARX 17 input vì gọn và dễ giải thích hơn.
```
