# ARX Professor Proof Protocol

Folder nay la ban lam lai tu dau theo quy trinh co the bao ve truoc giang vien ky tinh:

1. Mo phong plant nha kinh mini tach rieng voi model ARX.
2. Thu data bang rule-based safety + planned excitation nho/an toan.
3. Audit data truoc khi train.
4. Train ARX la model chinh.
5. Train NARX chi de doi chung phi tuyen.
6. Chon model bang validation, test chi bao cao cuoi.

Chay:

```powershell
python -B .\ARX_Professor_Proof_Protocol\src\protocol_grade_pipeline.py
```

Artifact:

- `results/protocol_greenhouse_data.csv`
- `results/FINAL_REPORT.md`
- `results/metrics.json`
- `results/comparison.csv`
- `results/arx_leaderboard.csv`
- `results/narx_leaderboard.csv`
- `results/test_predictions.csv`

