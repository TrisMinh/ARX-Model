# ARX PBL5 5s Clean

Ban nay la pipeline ARX 5 giay doc lap, tach code theo tung vai tro de de doc va de bao tri.

## Cau truc

```text
ARX_PBL5_5s_Clean/
  data/                 # Du lieu gia lap xuat ra sau khi chay train
  results/              # Leaderboard, metrics, predictions, model json
  scripts/train.py      # Lenh train tu dau den cuoi
  src/arx5s_clean/
    data/               # Sinh du lieu 5s
    preprocessing/      # Feature engineering, split, scale
    algorithm/          # Dinh nghia ARX va ham fit/mo phong
    evaluation/         # Metric va danh gia
    utils/              # Ghi file JSON/Markdown
    pipeline.py         # Noi cac buoc thanh mot workflow
```

## Chay lai

```powershell
cd ARX-Model\ARX_PBL5_5s_Clean
python -B .\scripts\train.py --days 4 --grid quick
```

Muon train bo dai hon:

```powershell
python -B .\scripts\train.py --days 16 --grid quick
```

## File ket qua chinh

- `data/mini_greenhouse_5s_data.csv`: file du lieu gia lap 5s.
- `results/leaderboard.csv`: bang so sanh cac cau hinh ARX tren validation.
- `results/metrics.json`: cau hinh, ket qua validation/test, audit du lieu.
- `results/arx_5s_model.json`: artifact de dua qua runtime sau nay.
- `results/test_predictions.csv`: y thuc te va y du doan tren test.
- `results/SUMMARY.md`: tom tat ket qua ngan gon.

