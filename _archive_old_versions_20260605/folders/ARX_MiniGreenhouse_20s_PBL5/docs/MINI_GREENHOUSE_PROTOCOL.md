# Quy Trình Dữ Liệu Cho Mini Greenhouse

## Hệ mô phỏng

Kích thước mục tiêu: `30x50x30 cm`.

Do đây là thể tích nhỏ, các biến vi khí hậu thay đổi nhanh:

- Fan có thể làm nhiệt độ/độ ẩm không khí thay đổi trong vài chục giây.
- Mist có thể tăng độ ẩm không khí gần như ngay lập tức.
- Drip ảnh hưởng độ ẩm đất chậm hơn không khí, nhưng với chậu nhỏ và cảm biến gần vùng tưới, đáp ứng có thể thay đổi sau khoảng `40-120 giây`.

Vì vậy sampling `20 giây/mẫu` hợp lý hơn `1-5 phút/mẫu` cho mô hình nhỏ. Tuy nhiên, khi đánh giá độ ẩm đất không chỉ nhìn 1-step 20 giây mà phải có:

- `FIT_1step`: 20 giây.
- `FIT_12`: 4 phút.
- `FIT_60`: 20 phút.
- `FIT_sim`: free-run simulation.

## Thu dữ liệu đúng quy trình

1. Giai đoạn kiểm tra ban đầu:
   - Dùng rule-based safety để giữ cây/hệ an toàn.
   - Kiểm tra sensor, relay, actuator.

2. Giai đoạn nhận dạng hệ:
   - Vẫn có safety supervisor.
   - Thêm planned pulses nhỏ cho Drip/Mist/Fan theo clock/random seed.
   - Planned pulses không sinh từ `Soil_Moisture`.

3. Giai đoạn validation/test:
   - Dùng các ngày sau cùng, không train.
   - Vẫn log planned command và safety override.

## Vì sao không để AI tự điều khiển ngay?

AI/MPC cần model. Model cần dữ liệu. Ban đầu chưa có dữ liệu nên phải dùng rule-based safety:

```text
Rule-based safety -> thu dữ liệu -> train ARX -> validate -> MPC dùng ARX
```

## Vì sao có pulse excitation?

Nếu bơm chỉ bật khi đất đã khô, data bị closed-loop feedback mạnh. Khi đó model khó phân biệt:

- đất khô làm bơm bật,
- hay bơm làm đất tăng.

Planned excitation là kỹ thuật nhận dạng hệ thống: kích thích nhỏ, có giới hạn an toàn, được log rõ nguồn lệnh.
