# Protocol-Grade ARX/NARX Final Report

## Ket luan

- Data duoc sinh theo quy trinh thu thap thuc te: rule-based safety truoc, planned excitation nho/an toan sau.
- Plant simulation tach rieng voi model ARX, co nonlinear evaporation, drainage, sensor noise va actuator dynamics.
- ARX la model chinh vi phu hop MPC tuyen tinh/RLS va de giai thich.
- NARX chi la doi chung phi tuyen; khong thay truc tiep ARX trong MPC neu chua doi sang NMPC/local linearization.
- Model ARX tot nhat: `ARX_na3_nb3_nk1_alpha0.1`.

## Data audit

| Check | Value |
| --- | ---: |
| Rows | 17280 |
| Missing total | 0 |
| Duplicate timestamps | 0 |
| Median sampling seconds | 300 |
| Irregular sampling count | 0 |
| Drip ON % | 4.062 |
| Planned Drip % | 2.656 |
| Safety override % | 0.723 |
| P(Drip ON | prev soil below low) | 14.763 |
| P(Drip ON | prev soil safe mid) | 2.594 |
| corr(Drip, prev soil-center) | -0.132 |
| corr(Planned Drip, prev soil-center) | -0.035 |

Diem quan trong: `Planned_Drip` la lich excitation sinh tu clock/random seed, khong tu soil moisture. `Drip` thuc te van co safety override nen co the phu thuoc soil, dung voi he that.

## Model comparison

| Case | Family | Selected | Test FIT_1step | Test FIT_12 | Test FIT_sim | Test RMSE_sim | MPC note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| ARX robust protocol | Linear ARX | `ARX_na3_nb3_nk1_alpha0.1` | 92.375 | 89.389 | 77.328 | 0.4061 | MPC-friendly linear plant model |
| NARX protocol comparison | Nonlinear NNARX | `Delta_NNARX_64_32_a05` | 92.285 | 88.255 | 49.837 | 0.8992 | Needs NMPC/local linearization for direct MPC use |

## ARX details

- Selected by robust validation score: `72.933`.
- Validation block FIT_sim: `[77.53, 63.177, 83.32, 84.787]`.
- Test one-step residual max abs ACF lag 1..24: `0.194`.
- Test free-run residual max abs ACF lag 1..24: `0.911`. Free-run residual tu tuong quan cao hon vi loi duoc tich luy theo thoi gian.
- Clip bounds are learned from train 0.5%-99.5% scaled quantiles: `[-1.48, 2.782]`.

## NARX details

- Selected NARX: `Delta_NNARX_64_32_a05`.
- NARX - ARX FIT_sim difference: `-27.491` points.

## Cach bao ve voi thay

1. Khong noi data nay la data that. Noi day la protocol-grade simulation de kiem thu quy trinh; khi co nha kinh mini se log cung schema va retrain.
2. Neu thay hoi AI dieu khien the nao khi chua co data: tra loi rule-based safety thu data truoc, ARX/MPC dung sau.
3. Neu thay hoi vi sao co pulse excitation: do la persistent excitation de nhan dang he, co safety supervisor nen khong nguy hiem.
4. Neu thay hoi vi sao khong dung NARX: vi ARX dat muc tot, giai thich duoc, va dung voi MPC tuyen tinh; NARX khong plug-compatible.
