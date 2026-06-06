# MPC Model Change Audit

## Ket luan ngan

- Doi ARX sang NARX co anh huong den MPC neu MPC hien tai dung mo hinh tuyen tinh ARX/RLS nhu trong bao cao.
- NARX khong the thay truc tiep vao Adaptive MPC dang cap nhat he so ARX bang RLS; can NMPC, local linearization, hoac giu ARX cho MPC va dung NARX cho du bao/monitoring.
- Tren data cu, clean NNARX khong vuot ARX o free-run. Tren data moi co excitation dung nguyen tac, ca ARX va Delta-NNARX deu dat tren 75% FIT_sim.

## Bao cao PBL da doc

- Bao cao chon `ARX(5,1,2)` cho do am dat.
- Ket qua cu trong bao cao: `FIT_1step ~= 85.85%`, `FIT_12 ~= 67.16%`, `FIT_sim ~= 66.42%`, `RMSE_sim ~= 0.978`.
- Phan MPC trong bao cao mo ta Adaptive MPC cap nhat he so ARX bang `Recursive Least Squares (RLS)`.
- Khong tim thay source code MPC rieng trong repo ngoai bao cao, nen nhan dinh MPC dua tren kien truc da mo ta trong report.

## Data audit

| Dataset | Missing | Duplicate time | Sampling s | Drip on % | P_drip_below_low % | P_drip_safe_mid % | corr_drip_prev_margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old feedback data | 0 | 0 | 300 | 22.800 | 75.337 | 5.914 | -0.611 |
| New identifiable data | 0 | 0 | 300 | 3.528 | NA | 3.588 | -0.008 |

Doc dung so lieu nay nhu sau: old data la closed-loop/feedback data, nen actuator Drip co dau vet tu trang thai do am truoc do. New data sinh actuator theo clock/weather/random excitation, khong tu `Soil_Moisture`.

## ARX search rerun

| Dataset | Selected ARX | Val robust | Test FIT_1step | Test FIT_12 | Test FIT_sim | Test RMSE_sim |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Old feedback data | `ARX_na5_nb1_nk2_alpha0` | 66.366 | 85.955 | 67.160 | 66.418 | 0.9782 |
| New identifiable data | `ARX_na12_nb5_nk2_alpha0` | 81.831 | 85.895 | 82.210 | 76.229 | 0.3314 |

## Model comparison

| Case | Dataset | Family | Test FIT_1step | Test FIT_12 | Test FIT_sim | Note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| PBL report previous result | old simulated feedback data | linear ARX | 85.850 | 67.160 | 66.420 | baseline in report |
| ARX robust search rerun | old simulated feedback data | linear ARX | 85.955 | 67.160 | 66.418 | linear, MPC-friendly |
| ARX robust search rerun | new identifiable data | linear ARX | 85.895 | 82.210 | 76.229 | linear, MPC-friendly |
| Clean NNARX rerun | old simulated feedback data | nonlinear NNARX | 88.885 | 67.950 | 64.674 | not better in free-run on old data |
| End-to-end Delta-NNARX | new identifiable data | nonlinear NNARX | 85.894 | 82.205 | 76.295 | target met on new identifiable data, but needs NMPC/local linearization for MPC |

## MPC impact

Neu giu MPC tuyen tinh hien tai: nen giu ARX/hybrid-ARX lam plant model cho MPC, va dung NARX nhu du bao phu hoac canh bao drift.

Neu muon dung NARX trong dieu khien: phai doi sang mot trong ba huong:

1. NMPC voi model phi tuyen va optimizer phu hop.
2. Local linearization cua NARX tai moi buoc de cap ma tran tuyen tinh cho MPC.
3. Gain-scheduled ARX/NARX-linearized theo mua/giai doan cay.

Ket luan thuc te: dung NARX de nang fit la hop ly ve mat du bao, nhung khong nen noi la thay ARX trong MPC ma khong sua MPC. Do la thay doi kien truc dieu khien.
