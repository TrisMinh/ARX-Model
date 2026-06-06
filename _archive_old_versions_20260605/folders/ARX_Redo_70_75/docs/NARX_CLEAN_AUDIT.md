# Clean NARX audit

Muc tieu la kiem tra NARX/NNARX co dat vung 88-96% nhu mot so paper hay khong, nhung khong dung leakage.

## Quy tac

- Split theo thoi gian 60/20/20.
- Z-score fit tren train.
- Regressor cua NARX chi gom output qua khu va input qua khu.
- Cac feature rolling actuator/moi truong deu shift 2 mau truoc khi rolling.
- Chon model bang validation `FIT_sim`.
- Test chi bao cao.

## Cach chay

```powershell
python .\ARX_Redo_70_75\src\narx_clean_experiments.py
```

Ket qua duoc luu trong:

- `results_narx/metrics.json`
- `results_narx/leaderboard.csv`
- `results_narx/SUMMARY.md`

## Diem can doc dung

`FIT_1step` cua NARX co the cao vi moi buoc duoc cap output that trong qua khu. `FIT_sim` moi la bai test kho neu model phai tu hoi quy bang output du doan cua chinh no.

## Ket qua hien tai

Best clean NARX chon theo validation `FIT_sim`:

```text
Delta_NNARX_MLP64_small_a10
Val  FIT_1step = 89.325%
Val  FIT_12    = 69.107%
Val  FIT_sim   = 66.199%

Test FIT_1step = 88.885%
Test FIT_12    = 67.950%
Test FIT_sim   = 64.674%
```

Ket luan: NARX/NNARX sach dat vung `~89%` o one-step, dung voi nhieu bao cao NNARX. Tuy nhien khi chuyen sang free-run simulation, model bi drift va kem hon hybrid ARX residual. Do do khong nen so sanh `FIT_1step` cua NNARX voi `FIT_sim` cua ARX.
