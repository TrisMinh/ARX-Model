# ARX Redo 70-75

Thu muc nay la ban lam lai co audit cho bai toan du doan `Soil_Moisture`.

Muc tieu:
- Giu split theo thoi gian, khong shuffle.
- Khong cat bo doan test kho.
- Khong dung `Soil_Moisture` tuong lai trong residual features.
- Bao cao rieng ARX backbone va mo hinh cuoi.
- Dat `FIT_sim` tren validation/test trong vung 70-75 neu co the.

Ket qua chinh duoc tao lai bang:

```powershell
python .\ARX_Redo_70_75\src\arx_redo_pipeline.py
```

Artifact sau khi chay:
- `results/metrics.json`: cau hinh, metrics, leaderboard.
- `results/leaderboard.csv`: cac shrink candidate chon theo validation.
- `results/test_predictions.csv`: y_true, ARX sim, hybrid sim tren test.
- `results/SUMMARY.md`: tom tat ket qua ngan gon.

Mo hinh cuoi la `Conservative Hybrid ARX Residual`:
- Backbone: ARX(5,3,2), 16 input vat ly/van hanh da z-score.
- Residual: `HistGradientBoostingRegressor` hoc sai so free-run cua ARX.
- Residual chi dung `y_arx` da mo phong va input lag `[2, 3, 6, 12, 24]`, tuc khong dung input hien tai/t-1 va khong dung output that tuong lai.

## Vong V75

Chay audit moc 75:

```powershell
python .\ARX_Redo_70_75\src\arx_v75_experiments.py
```

Artifact:
- `results_v75/SUMMARY.md`
- `results_v75/metrics.json`
- `results_v75/leaderboard.csv`

Ket luan hien tai:
- Track production-safe/causal tang len khoang `71.3%` test `FIT_sim`.
- Track diagnostic co future actuator vuot 75%, nhung khong production-safe neu actuator tuong lai chua duoc biet truoc.

## Clean NARX

Chay NARX/NNARX khong leakage:

```powershell
python .\ARX_Redo_70_75\src\narx_clean_experiments.py
```

Artifact:
- `results_narx/SUMMARY.md`
- `results_narx/metrics.json`
- `results_narx/leaderboard.csv`

## End-to-End NARX 75

Chay pipeline sinh data moi dung nguyen tac va train Delta-NNARX:

```powershell
python .\ARX_Redo_70_75\src\narx_end_to_end_75.py
```

Artifact:
- `results_end_to_end_75/SUMMARY.md`
- `results_end_to_end_75/metrics.json`
- `results_end_to_end_75/leaderboard.csv`
- `results_end_to_end_75/greenhouse_identifiable_narx.csv`

Ket qua hien tai tren data moi:
- Selected model: `delta_nnarx_mlp64_a01_rs1`.
- Test `FIT_1step`: `85.894%`.
- Test `FIT_12`: `82.205%`.
- Test `FIT_sim`: `76.295%`.
- Model duoc chon bang robust validation score, khong chon bang test.
