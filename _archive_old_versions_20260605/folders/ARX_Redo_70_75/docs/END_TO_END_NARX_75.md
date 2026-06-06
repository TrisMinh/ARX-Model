# End-to-end NARX 75

Muc tieu: neu data cu lam clean NARX free-run bi drift, tao lai data synthetic dung nguyen tac de kiem tra NARX co the dat `FIT_sim >= 75%` hay khong.

## Nguyen tac data moi

- Actuator `Drip/Mist/Fan` khong duoc sinh tu `Soil_Moisture`.
- `Drip` la lich tuoi theo gio + random persistent excitation.
- `Fan/Mist` dua tren thoi tiet/khong khi, khong dua tren do am dat.
- Output `Soil_Moisture` duoc sinh tu dynamic lagged inputs, setpoint relaxation, evaporation, actuator effects va nhieu nho.
- Split theo thoi gian 60/20/20.
- Scaling fit tren train.
- Model chi dung lag qua khu.

## Cach chay

```powershell
python .\ARX_Redo_70_75\src\narx_end_to_end_75.py
```

## Ket qua da dat

Model duoc chon: `delta_nnarx_mlp64_a01_rs1`.

| Metric | Validation | Test |
| --- | ---: | ---: |
| FIT_1step | 97.006 | 85.894 |
| FIT_12 | 92.535 | 82.205 |
| FIT_sim | 86.453 | 76.295 |
| RMSE_sim | 0.3057 | 0.3306 |

Target `FIT_sim >= 75%`: `dat`.

Model khong duoc chon bang test. Pipeline chia validation thanh 4 block lien tiep theo thoi gian va chon theo:

```text
robust_score = mean(block_FIT_sim) - 0.5 * std(block_FIT_sim)
```

Ly do: validation tong the co the qua dep nhung yeu o mot doan thoi gian; robust score uu tien mo hinh on dinh hon khi free-run.

Artifact:

- `results_end_to_end_75/greenhouse_identifiable_narx.csv`
- `results_end_to_end_75/metrics.json`
- `results_end_to_end_75/leaderboard.csv`
- `results_end_to_end_75/SUMMARY.md`

## Luu y

Ket qua tren data moi khong thay the benchmark data cu. No tra loi cau hoi data/model:

- Data cu: clean NNARX one-step cao nhung `FIT_sim` thap do closed-loop drift va feedback/input endogenous.
- Data moi: khi excitation va input policy ro rang hon, Delta-NNARX co the dat moc `FIT_sim >= 75%` ma khong can leakage.
