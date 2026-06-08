# ARX PBL5 5s Clean

Pipeline ARX 5 giây gọn lại: thu raw, clean raw, sinh data train, chạy model.

Hướng dẫn dựng source code từ đầu: `00_BUILD_SRC_FROM_ZERO.md`.
Hướng dẫn build lại toàn bộ từ raw data: `00_BUILD_FROM_RAW.md`.

## Cấu trúc chính

```text
arx_5s_clean/
  data/
    00_templates/
    01_raw_sessions/
    01_raw_sessions_real/
    02_cleaned_sessions/
  results/
  scripts/
    01_build_data_from_collection.py
    02_train.py
  src/arx5s_clean/
    data/collection/
    preprocessing/
    algorithm/
    evaluation/
    pipeline.py
```

## Chạy

```powershell
cd C:\Users\minht\OneDrive\Desktop\ARX-Model\arx_5s_clean
python -B .\scripts\01_build_data_from_collection.py --source legacy --days 12
python -B .\scripts\02_train.py --days 12 --grid quick
```

Khi có raw thật:

```powershell
python -B .\scripts\01_build_data_from_collection.py --source real --days 12
python -B .\scripts\02_train.py --days 12 --grid quick
```

## File đầu ra quan trọng

- `data/01_raw_sessions/00_raw_tong_hop.csv`
- `data/02_cleaned_sessions/00_sau_xu_ly_tong_hop.csv`
- `data/mini_greenhouse_5s_data.csv`
- `results/metrics.json`
- `results/SUMMARY.md`
