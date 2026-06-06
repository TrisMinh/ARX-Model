# Final Practical ARX/NARX Summary

## Ket luan chinh

- Current data khong bi missing/duplicate va sampling dung 300s, nhung la closed-loop feedback data.
- Tren current data, ARX van la lua chon dung logic hon cho MPC: free-run on dinh hon NARX va plug-compatible voi MPC tuyen tinh/RLS.
- NARX khong tao ra loi the ro tren current data: 1-step cao hon, nhung free-run thap hon ARX.
- Tren identifiable protocol data, ARX va NARX deu dat moc >75 FIT_sim; chenh lech free-run giua NARX va ARX rat nho.
- Huong cai thien thuc te nen uu tien data collection/excitation va ARX validation, khong nen doi thang sang NARX neu MPC chua doi kien truc.

## Data audit

| Dataset | Rows | Missing | Duplicate time | Sampling s | Drip on % | P_drip_below_low % | P_drip_safe_mid % | corr_drip_prev_margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current feedback data | 105120 | 0 | 0 | 300 | 22.800 | 75.337 | 5.914 | -0.611 |
| Identifiable protocol data | 105120 | 0 | 0 | 300 | 3.528 | NA | 3.588 | -0.008 |

Doc bang nay: current data co `P_drip_below_low` rat cao va corr am manh, tuc actuator bi chi phoi boi feedback soil. Day khong phai leakage tuong lai, nhung lam bai toan identification kho hon.

## ARX vs NARX

| Case | Dataset | Family | Selected | Test FIT_1step | Test FIT_12 | Test FIT_sim | Test RMSE_sim | MPC note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| PBL report previous ARX | current feedback data | Linear ARX | `ARX(5,1,2), LS, report split 70/15/15` | 85.850 | 67.160 | 66.420 | 0.9780 | baseline in PBL report |
| ARX robust search | current feedback data | Linear ARX | `ARX_na5_nb1_nk2_alpha0` | 85.955 | 67.160 | 66.418 | 0.9782 | MPC-friendly linear plant model |
| Clean NARX | current feedback data | Nonlinear NNARX | `Delta_NNARX_MLP64_small_a10` | 88.885 | 67.950 | 64.674 | 1.0290 | Not plug-compatible with current linear/RLS MPC |
| ARX robust search | identifiable protocol data | Linear ARX | `ARX_na12_nb5_nk2_alpha0` | 85.895 | 82.210 | 76.229 | 0.3314 | MPC-friendly linear plant model |
| Delta-NNARX identifiable protocol | identifiable protocol data | Nonlinear NNARX | `delta_nnarx_mlp64_a01_rs1` | 85.894 | 82.205 | 76.295 | 0.3306 | Needs NMPC/local linearization for direct control use |

## Chenh lech can noi ro

- Current data: NARX - ARX free-run = `-1.743` diem FIT_sim. NARX te hon ARX trong simulation dai han.
- Identifiable protocol data: NARX - ARX free-run = `0.066` diem FIT_sim. Chenh lech khong ro ve free-run.
- Identifiable protocol data: NARX - ARX 12-step = `-0.006` diem FIT_12. Gan nhu ngang nhau trong run hien tai.
- Khi chay clean NARX tren current data, sklearn co canh bao mot MLP cham `max_iter=90`. Dieu nay khong anh huong ket qua ARX chinh, nhung la ly do khong nen lay NARX lam ket luan trien khai neu chua tune/validate sau hon.

## Khuyen nghi thuc te

1. Cho bao cao/de tai ARX: dung ARX robust search tren current data lam ket qua trung thuc; neu can cai thien, trinh bay them ARX residual/hybrid nhu huong mo rong, khong goi la ARX thuan.
2. Cho MPC hien tai: giu ARX lam plant model. NARX chi nen dung monitoring hoac advisory prediction neu MPC chua chuyen sang NMPC/local linearization.
3. Cho lan thu data sau: thiet ke excitation doc lap nho/an toan cho Drip/Mist/Fan, log planned command, disturbance va sensor; khi data sach hon, ARX da co the vuot 75 ma khong can NARX.
4. Bao cao khong nen noi 'NARX ly thuyet 90 nen chac tot hon'. So lieu hien tai cho thay free-run moi la bai test quyet dinh.

## Reproduce

```powershell
python -B .\ARX_Practical_Final_2026\src\run_practical_benchmark.py
```
