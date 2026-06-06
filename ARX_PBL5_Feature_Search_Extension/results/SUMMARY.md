# Dò Feature Cho ARX Thuần

## Nguyên tắc

- Chỉ thử biến đo được hoặc suy ra từ cảm biến trong, actuator và thời gian.
- Không dùng `Soil_Moisture_True`, trạng thái ẩn mô phỏng, cảm biến ngoài hoặc dữ liệu tương lai.
- Feature được chọn bằng validation robust score; test chỉ dùng để báo cáo sau khi đã chọn.

## Kết quả ngắn

- Baseline: `FIT_sim = 78.950`, validation robust `75.270`.
- Bản chọn theo validation: `FIT_sim = 83.296`, validation robust `77.651`.
- Chênh lệch test so với baseline: `4.346` điểm.

## Nhóm feature được chọn

- `actuator_event_edges`: Drip_Start, Mist_Start, Fan_Start
- `mist_context_interactions`: Mist_x_Indoor_Dryness, Mist_x_TempIn, Mist_x_VPD
- `drip_context_interactions`: Drip_x_Indoor_Dryness, Drip_x_Light_log, Drip_x_VPD

## Cảnh báo

Nếu một feature làm test tăng nhưng validation robust giảm, không đưa vào bản final vì đó là chọn theo test và dễ bị xem là leakage phương pháp.
