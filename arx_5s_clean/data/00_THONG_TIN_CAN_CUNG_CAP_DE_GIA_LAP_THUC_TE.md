# Thông tin em cần gửi để mình giả lập đúng hơn

File này là checklist ngắn. Em không cần điền hết, chỉ cần gửi phần nào có.

## 1. File raw thật

Chỉ cần 1 file raw thật, dài khoảng 2 giờ, đặt vào:

```text
data/01_raw_sessions_real/
```

CSV vẫn chỉ có 8 cột:

```csv
Timestamp,Temperature,Humidity,Light,Soil_Moisture,Drip,Mist,Fan
```

Nếu không đo được `Light`, em ghi `0`.

## 2. Môi trường cơ bản

| Thông tin | Em điền |
|---|---|
| Nhiệt độ không khí lúc chưa bật gì | |
| Độ ẩm không khí lúc chưa bật gì | |
| Soil_Moisture thường nằm quanh bao nhiêu | |
| Có phun sương thật không | Có / Không |
| Có quạt thật không | Có / Không |

## 3. Bơm `Drip`

| Thông tin | Em điền |
|---|---|
| Em thường bật bơm bao lâu | 20s / 40s / 60s |
| Sau khi bật bơm, soil tăng rõ không | Có / Không |
| Soil tăng mạnh nhất khoảng bao nhiêu | |

## 4. Phun sương `Mist`

| Thông tin | Em điền |
|---|---|
| Nếu có, em thường bật bao lâu | 20s / 30s / khác |
| Humidity thường tăng khoảng bao nhiêu | |
| Temperature có giảm không | Có / Không |

## 5. Quạt `Fan`

| Thông tin | Em điền |
|---|---|
| Em thường bật quạt bao lâu | 60s / 120s / khác |
| Quạt làm nhiệt giảm hay tăng | |
| Quạt làm ẩm giảm hay tăng | |

## 6. Ghi chú gửi thêm

Nếu có, em ghi thêm 3 dòng này là đủ:

```text
Độ ẩm ngoài trời khoảng ...
Light không đo được, để 0.
Drip/Mist/Fan chạy theo các mốc trong file kịch bản.
```

## 7. Sau khi gửi raw

Chạy:

```powershell
python -B .\scripts\01_build_data_from_collection.py --source real --days 12
python -B .\scripts\02_train.py --days 12 --grid quick
```
