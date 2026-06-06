# ARX Practical Final 2026

Muc tieu folder nay: lam lai bai toan theo logic co the ap dung thuc te, uu tien ARX vi day la de tai chinh, va dung NARX nhu doi chung phi tuyen.

Nguyen tac:
- Split theo thoi gian, khong shuffle.
- Scaler/preprocess chi fit tren train.
- Chon model bang validation, test chi bao cao sau cung.
- Danh gia 3 che do: 1-step, 12-step, free-run simulation.
- Kiem tra data closed-loop truoc khi ket luan model.
- Khong dung `Soil_Moisture` tuong lai, khong dung future actuator neu khong phai planned command.
- Neu lien quan MPC, uu tien model tuyen tinh/on dinh/trien khai duoc.

Chay benchmark:

```powershell
python -B .\ARX_Practical_Final_2026\src\run_practical_benchmark.py
```

Artifact chinh:
- `results/FINAL_SUMMARY.md`
- `results/final_metrics.json`
- `results/final_comparison.csv`
- `results/arx_mpc_audit/`
- `results/narx_current_data/`
- `results/narx_identifiable_protocol/`

