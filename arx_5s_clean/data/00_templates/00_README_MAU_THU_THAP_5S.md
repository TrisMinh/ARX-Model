# File mẫu thu thập dữ liệu 5s

Bộ mẫu này chỉ dùng các cột model ARX 5s cần:

```csv
Timestamp,Temperature,Humidity,Light,Soil_Moisture,Drip,Mist,Fan
```

Nếu không có phun sương, giữ cột `Mist` và ghi `0`.

Vì dữ liệu lấy mẫu mỗi 5 giây, mọi test bật actuator nên là bội số của 5 giây. 20 giây vẫn hợp lệ vì tương đương 4 mẫu; nếu ngoài đời em dùng 30 giây hoặc 60 giây thì cứ ghi đúng như vậy.

Các file `*_sample.csv` chỉ là trích đoạn minh họa. Khi thu thật, file raw nên ghi liên tục từng dòng 5 giây trong suốt phiên đo.

## Các file

| File | Mục đích |
|---|---|
| `01_raw_session_template.csv` | Mẫu để copy khi thu một phiên mới |
| `02_morning_anchor_raw_sample.csv` | Ví dụ dữ liệu phiên sáng |
| `03_noon_anchor_raw_sample.csv` | Ví dụ dữ liệu phiên trưa |
| `04_drip_20s_test_sample.csv` | Ví dụ test bật bơm nhỏ giọt 20 giây |
| `05_mist_20s_test_sample.csv` | Ví dụ test bật phun sương 20 giây |
| `06_fan_120s_test_sample.csv` | Ví dụ test bật quạt 120 giây |

## Cách đặt tên file thật

Ví dụ:

```text
morning_anchor_2026_06_08.csv
noon_anchor_2026_06_08.csv
afternoon_anchor_2026_06_08.csv
night_anchor_2026_06_08.csv
drip_20s_test_2026_06_08.csv
mist_20s_test_2026_06_08.csv
fan_120s_test_2026_06_08.csv
```

Không cần thêm cột giải thích lệnh. Chỉ cần log đúng trạng thái thực tế của `Drip`, `Mist`, `Fan`.

Khi gửi dữ liệu thật để build theo hệ của em, đặt các file CSV vào:

```text
data/01_raw_sessions_real/
```
