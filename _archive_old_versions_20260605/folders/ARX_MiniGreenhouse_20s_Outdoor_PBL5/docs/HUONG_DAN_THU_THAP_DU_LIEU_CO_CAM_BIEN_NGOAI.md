# Hướng Dẫn Thu Dữ Liệu Có Cảm Biến Ngoài

## 1. Lắp cảm biến

Nên có hai cụm cảm biến:

- Cụm trong nhà kính: đặt gần vùng cây, tránh chạm nước trực tiếp.
- Cụm ngoài nhà kính: đặt ngoài hộp, cùng phòng/thí nghiệm, tránh gió thổi trực tiếp từ quạt nếu quạt xả thẳng vào cảm biến.

Với mô hình nhỏ, chỉ cần cảm biến ngoài cách hộp khoảng `20-50 cm` là đã có ý nghĩa, miễn là nó đo môi trường mà không khí sẽ đi vào khi quạt hút/thổi.

## 2. Các cột cần log

Tối thiểu:

| Cột | Ý nghĩa |
| --- | --- |
| `Timestamp` | thời điểm lấy mẫu |
| `Soil_Moisture` | độ ẩm đất đo được |
| `Temperature_In` | nhiệt độ trong nhà kính |
| `Humidity_In` | độ ẩm không khí trong nhà kính |
| `Temperature_Out` | nhiệt độ ngoài nhà kính |
| `Humidity_Out` | độ ẩm ngoài nhà kính |
| `Light_In` | ánh sáng trong nhà kính |
| `Drip` | bơm tưới |
| `Fan` | quạt |
| `Mist` | phun sương |
| `Planned_Drip` | xung tưới đã lên lịch |
| `Planned_Fan` | xung quạt đã lên lịch |
| `Safety_Override` | luật an toàn có can thiệp |
| `Command_Source` | nguồn lệnh actuator |

## 3. Lịch thu trong 1-2 ngày

### Nếu chỉ có 1 ngày

| Giai đoạn | Thời lượng | Việc cần làm |
| --- | ---: | --- |
| Kiểm tra cảm biến | 30 phút | log không actuator, bật thử từng actuator |
| Rule-based safety | 1-2 giờ | chạy an toàn, không planned pulse |
| Nhận dạng hệ | 4-6 giờ | planned drip/fan/mist nhỏ, có safety |
| Test cuối | 1-2 giờ | giữ riêng để đánh giá |

### Nếu có 2 ngày

Ngày 1 dùng để kiểm tra và nhận dạng chính. Ngày 2 dùng thêm planned pulse nhẹ và giữ 2 giờ cuối làm test.

## 4. Planned fan pulse

Vì câu hỏi chính là ảnh hưởng của môi trường ngoài khi bật quạt, nên cần planned fan pulse:

```text
Mỗi ngày 3-4 lần, mỗi lần 1-4 phút.
Không bật liên tục quá lâu.
Không đặt pulse dựa trên Soil_Moisture tương lai.
```

Khi quạt bật, cần log đầy đủ:

- `Fan = 1`;
- `Planned_Fan = 1` nếu là xung lên lịch;
- `Command_Source = planned_fan_ventilation_probe`;
- `Temperature_In`, `Humidity_In`, `Temperature_Out`, `Humidity_Out`.

## 5. Cách giải thích trước hội đồng

Có thể nói:

```text
Nhóm không xem nhà kính nhỏ là hệ kín. Khi quạt bật, trao đổi khí phụ thuộc vào điều kiện ngoài hộp. Vì vậy nhóm log thêm Temperature_Out và Humidity_Out để mô hình ARX nhìn thấy nhiễu đo được. Nhóm kiểm chứng bằng cách train hai ARX trên cùng dữ liệu: một model chỉ dùng cảm biến trong, một model có thêm cảm biến ngoài. Nếu model có cảm biến ngoài cải thiện FIT_sim trên test, điều đó chứng minh biến ngoài có giá trị nhận dạng.
```

## 6. Điều không được làm

- Không lấy dữ liệu ngoài từ tương lai.
- Không dùng test để chọn feature.
- Không xóa đoạn quạt làm model xấu đi.
- Không gọi dữ liệu mô phỏng là dữ liệu thật.
- Không báo cáo chỉ số 1-step rồi bỏ qua free-run.

