# Kịch bản thu data thực tế 5s

File này là kịch bản thu dữ liệu theo kiểu một sinh viên đang làm đồ án. Em làm đúng các mốc bật/tắt actuator bên dưới, còn các chỉ số sensor (`Temperature`, `Humidity`, `Light`, `Soil_Moisture`) để thiết bị đo thực tế, không tự sửa theo kịch bản.

Nếu em chỉ có 1 phiên raw thật thì chỉ cần thu đúng 1 phiên khoảng 2 giờ. Các phiên khác và phần `Light` sẽ do code tự sinh. Nếu không đo được ánh sáng, cứ ghi `Light = 0`.

CSV cuối cùng vẫn chỉ có 8 cột:

```csv
Timestamp,Temperature,Humidity,Light,Soil_Moisture,Drip,Mist,Fan
```

## 1. Nguyên tắc chung

- Sampling cố định `5 giây/mẫu`.
- Một phiên thu nên dài `2 giờ`.
- Không dừng logger giữa phiên.
- Mọi actuator phải ghi trạng thái thực tế tại từng dòng: bật là `1`, tắt là `0`.
- Các mốc bên dưới tính từ lúc bắt đầu phiên. Nếu bắt đầu lúc khác, cộng offset tương ứng.
- 20 giây là hợp lệ với bản 5s vì bằng 4 mẫu: `t`, `t+5`, `t+10`, `t+15` bật; tại `t+20` tắt.
- Nếu không có phun sương, giữ cột `Mist = 0` và bỏ qua các đoạn bật `Mist`.

Quy tắc an toàn khi thu:

- Nếu `Soil_Moisture > 65%` trước một đoạn bật `Drip`, bỏ qua đoạn `Drip` đó và vẫn log tiếp.
- Nếu `Soil_Moisture < 50%`, có thể thêm một lần `Drip 20s`, nhưng ghi chú lại thời điểm.
- Nếu cảm biến trả giá trị sai rõ ràng, không sửa raw ngay lúc đó; cứ log, sau đó xử lý ở bước clean.
- Nếu relay bật trễ hoặc tắt trễ, CSV phải ghi trạng thái thực tế, không ghi theo kế hoạch.

## 2. File nên thu

Khuyến nghị thu 4 phiên, mỗi phiên 2 giờ:

| File | Thời gian gợi ý | Mục tiêu |
|---|---:|---|
| `01_morning_2h_real.csv` | 07:00-09:00 | Ánh sáng tăng, đất khô tự nhiên, test `Drip` nhẹ |
| `02_noon_2h_real.csv` | 11:30-13:30 | Nhiệt/ánh sáng cao, test `Fan`, `Mist`, tổ hợp |
| `03_afternoon_2h_real.csv` | 15:00-17:00 | Ánh sáng giảm, test nhiều mức `Drip` |
| `04_night_2h_real.csv` | 20:00-22:00 | Ít sáng, đất khô chậm, xem trôi sensor ban đêm |

Nếu chỉ kịp thu ít, ưu tiên:

```text
01_morning_2h_real.csv
02_noon_2h_real.csv
03_afternoon_2h_real.csv
```

## 3. Kịch bản phiên sáng 07:00-09:00

Mục tiêu: lấy baseline buổi sáng, test bơm ngắn, xem đất tăng sau tưới và hồi phục.

| Mốc phiên | Trạng thái actuator | Ghi chú |
|---:|---|---|
| 00:00-10:00 | `Drip=0, Mist=0, Fan=0` | Baseline, không can thiệp |
| 10:00-10:20 | `Drip=1, Mist=0, Fan=0` | Test bơm 20s |
| 10:20-25:00 | `Drip=0, Mist=0, Fan=0` | Chờ soil phản ứng |
| 25:00-27:00 | `Drip=0, Mist=0, Fan=1` | Test quạt 120s |
| 27:00-35:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 35:00-35:20 | `Drip=0, Mist=1, Fan=0` | Test phun sương 20s |
| 35:20-50:00 | `Drip=0, Mist=0, Fan=0` | Xem độ ẩm không khí giảm lại |
| 50:00-50:40 | `Drip=1, Mist=0, Fan=0` | Test bơm 40s nếu soil chưa quá 65% |
| 50:40-65:00 | `Drip=0, Mist=0, Fan=0` | Chờ soil phản ứng |
| 65:00-67:00 | `Drip=0, Mist=0, Fan=1` | Quạt sau khi đất vừa được tưới |
| 67:00-80:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 80:00-80:20 | `Drip=1, Mist=0, Fan=1` | Tổ hợp bơm + quạt 20s |
| 80:20-82:00 | `Drip=0, Mist=0, Fan=1` | Quạt chạy tiếp |
| 82:00-100:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục dài |
| 100:00-100:20 | `Drip=0, Mist=1, Fan=1` | Tổ hợp phun sương + quạt 20s |
| 100:20-102:00 | `Drip=0, Mist=0, Fan=1` | Quạt chạy tiếp |
| 102:00-120:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục cuối phiên |

## 4. Kịch bản phiên trưa 11:30-13:30

Mục tiêu: lấy dữ liệu lúc nhiệt/ánh sáng cao, quạt và phun sương có tác động rõ hơn.

| Mốc phiên | Trạng thái actuator | Ghi chú |
|---:|---|---|
| 00:00-12:00 | `Drip=0, Mist=0, Fan=0` | Baseline trưa |
| 12:00-14:00 | `Drip=0, Mist=0, Fan=1` | Test quạt 120s |
| 14:00-25:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 25:00-25:20 | `Drip=0, Mist=1, Fan=0` | Test phun sương 20s |
| 25:20-40:00 | `Drip=0, Mist=0, Fan=0` | Xem độ ẩm không khí hồi về |
| 40:00-40:20 | `Drip=0, Mist=1, Fan=1` | Phun sương khi có quạt |
| 40:20-42:00 | `Drip=0, Mist=0, Fan=1` | Quạt chạy tiếp |
| 42:00-55:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 55:00-55:20 | `Drip=1, Mist=0, Fan=0` | Bơm 20s, chỉ làm nếu soil chưa quá 65% |
| 55:20-75:00 | `Drip=0, Mist=0, Fan=0` | Chờ soil phản ứng |
| 75:00-77:00 | `Drip=0, Mist=0, Fan=1` | Quạt khi đất đang ẩm hơn |
| 77:00-90:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 90:00-90:40 | `Drip=1, Mist=0, Fan=1` | Tổ hợp bơm + quạt 40s nếu soil chưa quá 65% |
| 90:40-92:00 | `Drip=0, Mist=0, Fan=1` | Quạt chạy tiếp |
| 92:00-120:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục cuối phiên |

## 5. Kịch bản phiên chiều 15:00-17:00

Mục tiêu: lấy đoạn ánh sáng giảm, test các mức bơm khác nhau để model học đáp ứng của `Drip`.

| Mốc phiên | Trạng thái actuator | Ghi chú |
|---:|---|---|
| 00:00-10:00 | `Drip=0, Mist=0, Fan=0` | Baseline chiều |
| 10:00-10:20 | `Drip=1, Mist=0, Fan=0` | Bơm 20s |
| 10:20-25:00 | `Drip=0, Mist=0, Fan=0` | Chờ soil phản ứng |
| 25:00-25:40 | `Drip=1, Mist=0, Fan=0` | Bơm 40s nếu soil chưa quá 65% |
| 25:40-45:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục dài |
| 45:00-47:00 | `Drip=0, Mist=0, Fan=1` | Test quạt |
| 47:00-60:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 60:00-60:20 | `Drip=0, Mist=1, Fan=0` | Phun sương 20s |
| 60:20-75:00 | `Drip=0, Mist=0, Fan=0` | Xem humidity giảm lại |
| 75:00-76:00 | `Drip=1, Mist=0, Fan=0` | Bơm 60s nếu soil chưa quá 65% |
| 76:00-95:00 | `Drip=0, Mist=0, Fan=0` | Chờ soil phản ứng |
| 95:00-97:00 | `Drip=0, Mist=0, Fan=1` | Quạt sau bơm dài |
| 97:00-120:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục cuối phiên |

Nếu đất dễ bị quá ẩm, bỏ đoạn bơm 60s và thay bằng `Drip 20s`.

## 6. Kịch bản phiên tối 20:00-22:00

Mục tiêu: lấy dữ liệu ít ánh sáng, nhiệt/ẩm ổn định hơn, đất khô chậm hơn ban ngày.

| Mốc phiên | Trạng thái actuator | Ghi chú |
|---:|---|---|
| 00:00-15:00 | `Drip=0, Mist=0, Fan=0` | Baseline tối dài |
| 15:00-15:20 | `Drip=1, Mist=0, Fan=0` | Bơm 20s |
| 15:20-35:00 | `Drip=0, Mist=0, Fan=0` | Chờ soil phản ứng |
| 35:00-37:00 | `Drip=0, Mist=0, Fan=1` | Test quạt ban đêm |
| 37:00-50:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 50:00-50:20 | `Drip=0, Mist=1, Fan=0` | Phun sương 20s nếu có |
| 50:20-70:00 | `Drip=0, Mist=0, Fan=0` | Xem humidity hồi về |
| 70:00-70:40 | `Drip=1, Mist=0, Fan=0` | Bơm 40s nếu soil chưa quá 65% |
| 70:40-90:00 | `Drip=0, Mist=0, Fan=0` | Chờ soil phản ứng |
| 90:00-92:00 | `Drip=0, Mist=0, Fan=1` | Quạt |
| 92:00-120:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục cuối phiên |

## 7. Nếu chỉ thu được một phiên 2 giờ

Nếu em chỉ kịp thu đúng một phiên, dùng kịch bản tổng hợp này:

| Mốc phiên | Trạng thái actuator | Ghi chú |
|---:|---|---|
| 00:00-10:00 | `Drip=0, Mist=0, Fan=0` | Baseline |
| 10:00-10:20 | `Drip=1, Mist=0, Fan=0` | Bơm 20s |
| 10:20-25:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 25:00-27:00 | `Drip=0, Mist=0, Fan=1` | Quạt 120s |
| 27:00-40:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 40:00-40:20 | `Drip=0, Mist=1, Fan=0` | Phun sương 20s |
| 40:20-55:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 55:00-55:40 | `Drip=1, Mist=0, Fan=0` | Bơm 40s nếu soil chưa quá 65% |
| 55:40-75:00 | `Drip=0, Mist=0, Fan=0` | Chờ soil |
| 75:00-77:00 | `Drip=0, Mist=0, Fan=1` | Quạt |
| 77:00-90:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục |
| 90:00-90:20 | `Drip=0, Mist=1, Fan=1` | Phun sương + quạt |
| 90:20-92:00 | `Drip=0, Mist=0, Fan=1` | Quạt chạy tiếp |
| 92:00-120:00 | `Drip=0, Mist=0, Fan=0` | Hồi phục cuối |

Một phiên duy nhất là đủ cho workflow hiện tại. Code sẽ lấy phản ứng thật trong phiên đó rồi tự đặt lại vào các khung sáng, trưa, chiều, tối để sinh data train nhiều ngày.

## 8. Trước và sau khi thu

Trước khi bấm start:

1. Kiểm tra logger đang xuất đủ 8 cột.
2. Kiểm tra sampling đúng 5 giây.
3. Đặt `Drip=0`, `Mist=0`, `Fan=0`.
4. Đợi sensor ổn định khoảng 2-3 phút.
5. Bắt đầu ghi file CSV.

Sau khi thu:

1. Không sửa trực tiếp file raw.
2. Đặt file vào `data/01_raw_sessions_real/`.
3. Ghi thêm một note ngắn: ngày thu, giờ thu, có bỏ qua mốc nào không, đất có bị quá ẩm không.
4. Chạy build:

```powershell
python -B .\scripts\01_build_data_from_collection.py --source real --days 12
```

## 9. Cách ghi actuator trong CSV

Ví dụ đoạn `Drip 20s` bắt đầu lúc `07:10:00`:

```csv
Timestamp,Temperature,Humidity,Light,Soil_Moisture,Drip,Mist,Fan
2026-06-08 07:09:55,28.7,75.0,240.0,55.2,0,0,0
2026-06-08 07:10:00,28.7,75.0,241.0,55.2,1,0,0
2026-06-08 07:10:05,28.7,75.0,242.0,55.2,1,0,0
2026-06-08 07:10:10,28.7,74.9,243.0,55.2,1,0,0
2026-06-08 07:10:15,28.7,74.9,244.0,55.2,1,0,0
2026-06-08 07:10:20,28.7,74.9,245.0,55.3,0,0,0
```

Chỉ số sensor trong ví dụ trên là minh họa. Khi em thu thật, giữ nguyên chỉ số máy đo được.
