# Nguồn biến khi triển khai thực tế

File này trả lời câu hỏi: khi đưa mô hình ra chạy thật, các biến đầu vào lấy từ đâu, biến nào đo bằng cảm biến, biến nào lấy từ lệnh điều khiển, biến nào do mô hình tự dự đoán.

## 1. Kết luận ngắn

Mô hình ARX/Hybrid không tự tạo toàn bộ biến đầu vào. Nó cần dữ liệu từ hệ thống thật.

Trong bản triển khai PBL5 hiện tại, chốt cách dùng như sau:

```text
Soil_Moisture quá khứ  -> lấy từ cảm biến, sau đó dùng dự đoán của model khi chạy nhiều bước
Temperature_In quá khứ -> lấy từ cảm biến
Humidity_In quá khứ    -> lấy từ cảm biến
Light_In quá khứ       -> lấy từ cảm biến
Drip/Mist/Fan quá khứ  -> lấy từ lệnh điều khiển đã phát ra
Drip/Mist/Fan tương lai -> lấy từ chuỗi lệnh giả định/kế hoạch của controller
Temperature/Humidity/Light tương lai -> giữ bằng giá trị đo mới nhất trong horizon ngắn
Feature phụ             -> tự tính từ các biến trên
```

Nói ngắn:

```text
Cảm biến đo môi trường.
Controller biết lệnh actuator.
Model tự dự đoán Soil_Moisture tương lai.
```

## 2. Bảng nguồn biến

| Biến | Nguồn khi chạy thật | Có phải model tự tạo không? | Ghi chú |
| --- | --- | --- | --- |
| `Soil_Moisture` quá khứ | cảm biến độ ẩm đất | không | dùng làm lịch sử ban đầu cho ARX |
| `Soil_Moisture` tương lai | model dự đoán | có | khi multi-step/free-run, model dùng lại giá trị nó dự đoán |
| `Temperature_In` quá khứ | cảm biến nhiệt độ trong mô hình | không | đo mỗi 20 giây |
| `Humidity_In` quá khứ | cảm biến độ ẩm không khí trong mô hình | không | đo mỗi 20 giây |
| `Light_In` quá khứ | cảm biến ánh sáng trong mô hình | không | đo mỗi 20 giây |
| `Temperature_In` tương lai | giữ bằng giá trị đo mới nhất | không hẳn | giả định horizon ngắn, ví dụ 4-20 phút |
| `Humidity_In` tương lai | giữ bằng giá trị đo mới nhất | không hẳn | giả định horizon ngắn |
| `Light_In` tương lai | giữ bằng giá trị đo mới nhất | không hẳn | nếu thay đổi ánh sáng mạnh thì cần nâng cấp forecast |
| `Drip` quá khứ | log lệnh bơm đã phát | không | Arduino/ESP biết nó vừa bật hay tắt |
| `Mist` quá khứ | log lệnh phun sương đã phát | không | tương tự |
| `Fan` quá khứ | log lệnh quạt đã phát | không | tương tự |
| `Drip/Mist/Fan` tương lai | chuỗi lệnh giả định/kế hoạch của controller | không | ARX không tự quyết định tưới |
| `Hour_sin`, `Hour_cos` | tính từ timestamp | có, nhưng không phải dự đoán | biết chính xác thời gian hiện tại/tương lai |
| `Day_sin`, `Day_cos` | tính từ timestamp | có, nhưng không phải dự đoán | biết chính xác ngày |
| `Light_log` | tính từ `Light_In` | có | feature phụ |
| `Indoor_Dryness` | tính từ `Humidity_In` | có | `100 - Humidity_In` |
| `VPD_Proxy_In` | tính từ `Temperature_In` và `Humidity_In` | có | feature phụ |
| `TempIn_x_HumiIn`, `TempIn_x_Light`, `HumiIn_x_Light` | tính từ các cảm biến/feature trên | có | feature tương tác |

## 3. Khi dự đoán một bước

Nếu chỉ dự đoán 1 bước tiếp theo, ví dụ sau 20 giây, mô hình dùng toàn bộ dữ liệu quá khứ đã biết.

Với model final:

```text
na = 12
nb = 6
nk = 1
T_s = 20 giây
```

Để dự đoán `Soil_Moisture(k)`, ARX dùng:

```text
Soil_Moisture(k-1), ..., Soil_Moisture(k-12)
input(k-1), ..., input(k-6)
```

Tất cả giá trị này đều đã xảy ra trước thời điểm cần dự đoán, nên có thể lấy từ cảm biến và log lệnh.

## 4. Khi dự đoán nhiều bước trong thực tế

Khi dự đoán nhiều bước, ví dụ 60 bước = 20 phút, có 3 nhóm biến.

### Nhóm 1: Soil_Moisture tương lai

Nhóm này do model tự dự đoán.

Ví dụ:

```text
y_sim(k+1) do model dự đoán
y_sim(k+2) dùng y_sim(k+1)
y_sim(k+3) dùng y_sim(k+2), y_sim(k+1)
...
```

Sau vài bước, lịch sử `Soil_Moisture` mà ARX dùng sẽ chủ yếu là giá trị do chính model dự đoán.

### Nhóm 2: Actuator tương lai

Các biến:

```text
Drip
Mist
Fan
```

không do ARX tự sinh ra.

Trong triển khai điều khiển, chuỗi actuator tương lai do controller cung cấp.

Ví dụ controller muốn thử một phương án:

```text
20 giây tới: Drip=1, Fan=0, Mist=0
40 giây tới: Drip=1, Fan=0, Mist=0
60 giây tới: Drip=0, Fan=0, Mist=0
...
```

ARX nhận chuỗi này như input rồi dự đoán:

```text
nếu tưới theo phương án đó thì Soil_Moisture sẽ ra sao?
```

Nếu controller thử phương án khác, ARX lại mô phỏng lại với chuỗi actuator khác.

Vì vậy ARX là mô hình dự đoán hệ, không phải bộ tự quyết định tưới.

### Nhóm 3: Môi trường tương lai

Các biến:

```text
Temperature_In
Humidity_In
Light_In
```

trong tương lai chưa đo được tại thời điểm hiện tại.

Để bản PBL5 đơn giản và rõ ràng, chốt giả định triển khai như sau:

```text
Trong horizon ngắn 4-20 phút, giữ Temperature_In, Humidity_In, Light_In bằng giá trị đo mới nhất.
```

Ví dụ tại thời điểm hiện tại:

```text
Temperature_In = 30.2
Humidity_In    = 72.0
Light_In       = 640
```

Khi mô phỏng 20 phút tới, dùng:

```text
Temperature_In(k+1...k+60) = 30.2
Humidity_In(k+1...k+60)    = 72.0
Light_In(k+1...k+60)       = 640
```

Đây gọi là giả định giữ nguyên ngắn hạn. Với mô hình nhỏ và horizon ngắn, cách này dễ triển khai và dễ bảo vệ hơn việc thêm một model dự báo môi trường riêng.

Giới hạn phải nói rõ:

```text
Nếu đang ở thời điểm ánh sáng/nhiệt độ thay đổi rất nhanh, ví dụ nắng gắt đột ngột hoặc bật/tắt đèn mạnh, dự báo giữ nguyên có thể sai. Khi đó cần nâng cấp thêm forecast môi trường.
```

## 5. Vậy free-run trong kết quả hiện tại có giống triển khai thật không?

Không hoàn toàn.

Trong kết quả `FIT_sim` hiện tại, free-run dùng:

```text
Soil_Moisture -> model tự dự đoán
Drip/Mist/Fan -> lấy từ chuỗi test đã log
Temperature/Humidity/Light -> lấy từ chuỗi test đã log
```

Nghĩa là `FIT_sim` kiểm tra:

```text
Nếu biết đúng chuỗi input đã xảy ra, model mô phỏng Soil_Moisture tốt đến đâu?
```

Nó chưa chứng minh rằng controller đã tự chọn lệnh tưới tối ưu. Bài toán chọn lệnh tưới là phần điều khiển/MPC hoặc rule controller.

Khi triển khai thật, ta dùng mô hình theo kiểu receding horizon:

```text
1. Đọc cảm biến hiện tại.
2. Cập nhật buffer dữ liệu quá khứ.
3. Controller tạo chuỗi actuator giả định.
4. Môi trường tương lai giữ bằng giá trị đo mới nhất.
5. ARX/Hybrid dự đoán Soil_Moisture tương lai.
6. Controller chọn hoặc thực thi lệnh.
7. Sau 20 giây, đọc cảm biến mới và lặp lại.
```

Vì cứ 20 giây cập nhật lại bằng cảm biến thật, sai số dự báo môi trường không bị để trôi quá lâu.

## 6. Vậy hiện tại nếu không dùng test/validation thì model dự đoán kiểu gì?

Điểm cần hiểu:

```text
Code hiện tại là pipeline train + đánh giá.
Nó chưa phải script triển khai realtime hoàn chỉnh.
```

Trong pipeline đánh giá, khi chạy validation/test, dataframe đã có sẵn toàn bộ chuỗi input:

```text
Temperature_In(k)
Humidity_In(k)
Light_In(k)
Drip(k)
Mist(k)
Fan(k)
```

cho mọi thời điểm trong validation/test. Vì vậy hàm mô phỏng có thể lấy input quá khứ từ dataframe đó để chạy nhiều bước/free-run.

Khi chạy thật ngoài đời, ta không có sẵn "test dataframe tương lai". Ta chỉ có:

```text
dữ liệu cảm biến đến thời điểm hiện tại
lệnh actuator đã phát đến thời điểm hiện tại
```

Muốn dự đoán nhiều bước, phải có thêm một script inference tạo "kịch bản tương lai".

## 7. Một bước và nhiều bước khác nhau ở đâu?

Với `nk=1`, để dự đoán một bước tiếp theo, model cần input gần nhất vừa biết.

Ví dụ đang ở thời điểm `k`, muốn dự đoán `y(k+1)`:

```text
y(k+1) cần y(k), y(k-1), ...
y(k+1) cần input(k), input(k-1), ...
```

Các giá trị này đều đã biết tại thời điểm hiện tại:

```text
y(k) lấy từ cảm biến
input(k) lấy từ cảm biến và lệnh điều khiển hiện tại
```

Vì vậy dự đoán 1 bước là trực tiếp.

Nhưng muốn dự đoán 2 bước, tức `y(k+2)`, model cần:

```text
y(k+1)     -> có thể dùng y vừa dự đoán
input(k+1) -> chưa biết nếu chưa tạo kịch bản tương lai
```

Vì vậy vấn đề không nằm ở `Soil_Moisture`. `Soil_Moisture` tương lai có thể lấy từ dự đoán trước đó. Vấn đề nằm ở input tương lai.

Nói ngắn:

```text
ARX tự cuốn được Soil_Moisture.
ARX không tự biết actuator/môi trường tương lai nếu ta không cung cấp kịch bản.
```

## 8. Có cần viết thêm script không?

Có. Nếu muốn dùng model sau khi train để dự đoán thật, cần thêm script inference/realtime forecast.

Script đó phải làm các việc:

```text
1. Nạp model đã train: theta ARX, scaler, input_cols, spec, residual model nếu dùng Hybrid.
2. Đọc buffer dữ liệu mới nhất từ cảm biến/log điều khiển.
3. Tạo kịch bản input tương lai cho horizon cần dự đoán.
4. Chạy ARX/Hybrid nhiều bước.
5. Trả ra chuỗi Soil_Moisture dự đoán.
```

Ví dụ kịch bản tương lai đơn giản:

```text
Horizon = 60 bước = 20 phút
Temperature/Humidity/Light tương lai = giữ bằng giá trị đo mới nhất
Fan tương lai = giữ theo trạng thái hiện tại hoặc theo rule
Mist tương lai = giữ theo trạng thái hiện tại hoặc theo rule
Drip tương lai = thử các phương án: không tưới, tưới 20 giây, tưới 40 giây
```

Sau đó dùng model dự đoán từng phương án:

```text
Phương án A: Drip toàn 0
-> dự đoán Soil_Moisture 20 phút tới

Phương án B: Drip bật 1 mẫu rồi tắt
-> dự đoán Soil_Moisture 20 phút tới

Phương án C: Drip bật 2 mẫu rồi tắt
-> dự đoán Soil_Moisture 20 phút tới
```

Nếu chỉ muốn dự báo, script có thể xuất cả 3 đường dự đoán để người dùng xem.

Nếu muốn điều khiển tự động, controller/MPC sẽ chọn phương án tốt nhất rồi chỉ thực thi bước đầu tiên. Sau 20 giây, hệ thống đo lại cảm biến và lặp lại.

## 9. Vậy "chỉ ARX" có dự đoán nhiều bước được không?

Có, nhưng phải nói đầy đủ:

```text
ARX dự đoán nhiều bước được nếu có chuỗi input tương lai giả định.
ARX không tự chọn chuỗi input tương lai.
```

Nếu không cung cấp kịch bản input tương lai, thì đúng là model chỉ dự đoán chắc chắn được 1 bước gần nhất dựa trên dữ liệu hiện tại.

Vì vậy câu đúng là:

```text
Chỉ ARX vẫn chạy multi-step được, nhưng cần script tạo future scenario.
Nếu không có future scenario, chỉ dự đoán 1 bước là rõ ràng nhất.
```

## 10. Câu trả lời khi bảo vệ

Nếu thầy hỏi: "Áp dụng thực tế lấy biến đâu mà dự đoán?", trả lời:

```text
Khi chạy thật, các biến trạng thái như Soil_Moisture, Temperature_In, Humidity_In, Light_In được lấy từ cảm biến. Các biến actuator Drip, Mist, Fan được lấy từ chính lệnh điều khiển mà hệ thống phát ra nên luôn biết trạng thái bật/tắt. Khi dự đoán nhiều bước, Soil_Moisture tương lai là do model tự dự đoán, còn Drip/Mist/Fan tương lai là chuỗi lệnh giả định hoặc kế hoạch của controller. Với bản PBL5 hiện tại, các biến môi trường tương lai trong horizon ngắn 4-20 phút được giữ bằng giá trị đo mới nhất. Sau mỗi 20 giây hệ thống đọc cảm biến mới và dự đoán lại, nên đây là dự báo kiểu receding horizon.
```

Nếu thầy hỏi: "Vậy FIT_sim hiện tại có phải kết quả điều khiển tự động không?", trả lời:

```text
Không. FIT_sim hiện tại là đánh giá mô hình dự đoán dưới chuỗi input đã biết trong test. Nó chứng minh mô hình mô phỏng độ ẩm đất tốt khi biết input. Phần quyết định tưới thuộc controller/MPC, không phải ARX tự sinh lệnh.
```
