# ARX Mini Greenhouse 20s PBL5

Bản này được thiết kế riêng cho mô hình nhà kính nhỏ `30x50x30 cm`, lấy mẫu mỗi `20 giây`.

Mục tiêu:
- Dùng logic thu dữ liệu thực tế cho đồ án môn học.
- Có rule-based safety trước khi dùng AI/MPC.
- Có planned excitation nhỏ/an toàn để ARX học được tác động của actuator.
- ARX là model chính; NARX chỉ là đối chứng.
- Báo cáo có tự phản biện để tránh bị bắt lỗi học thuật.

Chạy:

```powershell
python -B .\ARX_MiniGreenhouse_20s_PBL5\src\mini20s_pipeline.py
```

Tệp kết quả:
- `results/FINAL_REPORT.md`
- `results/SELF_CRITIQUE.md`
- `results/metrics.json`
- `results/comparison.csv`
- `results/mini_greenhouse_20s_data.csv`
- `results/arx_leaderboard.csv`
- `results/narx_leaderboard.csv`
- `results/test_predictions.csv`
