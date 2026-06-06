# ARX Mini Greenhouse 20 Giây Có Cảm Biến Ngoài

Folder này kiểm chứng giả thuyết: với mô hình nhà kính nhỏ `30x50x30 cm`, khi bật quạt thì không khí ngoài đi vào và không khí trong đi ra, nên `Temperature_Out` và `Humidity_Out` có thể giúp ARX dự đoán độ ẩm đất tốt hơn.

Điểm khác so với bản trước:

- Log rõ `Temperature_In`, `Humidity_In`, `Temperature_Out`, `Humidity_Out`, `Light_In`, `Light_Out`.
- Có trạng thái quạt làm tăng trao đổi không khí.
- So sánh cùng một dữ liệu, cùng split, cùng validation policy:
  - ARX baseline cũ chỉ cảm biến trong.
  - ARX robust chỉ cảm biến trong.
  - ARX robust có thêm cảm biến ngoài.
- Không dùng dữ liệu tương lai.
- Không dùng trạng thái ẩn của mô phỏng để train.
- Không gọi dữ liệu mô phỏng là dữ liệu phần cứng.
- Cấu hình mặc định chạy 16 ngày mô phỏng để kiểm tra pipeline và mô phỏng phần làm giàu dữ liệu có kiểm soát; khi có dữ liệu thật phải retrain/test lại.

Chạy:

```powershell
python -B .\ARX_MiniGreenhouse_20s_Outdoor_PBL5\src\outdoor_pipeline.py
```

Tệp kết quả:

- `results/FINAL_REPORT.md`
- `results/SELF_CRITIQUE.md`
- `results/metrics.json`
- `results/comparison.csv`
- `results/mini_greenhouse_20s_outdoor_data.csv`
- `results/arx_inside_leaderboard.csv`
- `results/arx_outdoor_leaderboard.csv`
- `results/test_predictions.csv`
