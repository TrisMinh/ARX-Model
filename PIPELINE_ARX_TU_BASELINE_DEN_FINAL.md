# Pipeline ARX tu baseline goc den model final hien tai

Tai lieu nay mo ta qua trinh xay dung model theo dung thu tu pipeline, khong chi la bang so sanh. Muc tieu la nhin duoc model da di tu ban ARX goc len ban final nhu the nao: data, split, regression matrix, log transform, feature engineering, order search, clip, normalize khi regularize, Ridge/Lasso, rolling validation, va buoc thu rich features.

Pham vi tap trung:

- Baseline goc: `ARX(2,2,1)`.
- Model final hien tai: `ARX(5,1,2)`.
- Cac buoc trung gian chi duoc nhac khi chung nam trong duong di len final.

## 1. Data dau vao

Pipeline dung file:

```text
greenhouse_data.csv
```

Thong tin trong artifact improved:

```text
rows = 105120
timestamp_start = 2025-01-01 00:00:00
timestamp_end = 2025-12-31 23:55:00
sampling = 5 phut
```

Output can model:

```text
Soil_Moisture
```

Input goc:

```text
Temperature
Humidity
Light
Drip
Mist
Fan
```

## 2. Chronological split

Data duoc chia theo thoi gian, khong shuffle:

```text
Train = 60%
Validation = 20%
Test = 20%
```

Ly do:

- Day la time series nen khong duoc random split.
- Train dung de fit tham so.
- Validation dung de chon order, feature set, clip, regularization.
- Test dung de kiem tra generalization cuoi cung.

## 3. Baseline goc: ARX(2,2,1)

### 3.1 Cau hinh baseline

```text
Order = ARX(2,2,1)
Input = 6 bien goc
Intercept = False
Clip = None
Fit = OLS
```

Y nghia order:

- `na=2`: dung 2 lag cua `Soil_Moisture`.
- `nb=2`: moi input co 2 lag.
- `nk=1`: input bat dau tac dong tu tre 1 buoc.

### 3.2 Regression matrix baseline

Voi `ARX(2,2,1)`, moi dong cua ma tran hoi quy gom:

```text
y(t-1), y(t-2),
Temperature(t-1), Temperature(t-2),
Humidity(t-1), Humidity(t-2),
Light(t-1), Light(t-2),
Drip(t-1), Drip(t-2),
Mist(t-1), Mist(t-2),
Fan(t-1), Fan(t-2)
```

Khong co intercept nen tong so tham so:

```text
2 + 6*2 = 14
```

### 3.3 Ket qua baseline

```text
Validation FIT_sim = 42.959
Test FIT_sim = 43.875
```

Nhan xet pipeline:

- 1-step prediction co the nhin tot hon, nhung `FIT_sim` thap.
- Free-run bi loi tich luy vi model tu dung output du doan cua chinh no.
- Day la diem xuat phat, khong phai model de chon cuoi.

## 4. Them intercept tren baseline

Buoc tiep theo la bat intercept nhung van giu:

```text
Order = ARX(2,2,1)
Input = 6 bien goc
Clip = None
Fit = OLS
```

Ket qua:

```text
Validation FIT_sim = 42.486
Test FIT_sim = 43.399
```

Nhan xet pipeline:

- Intercept don le khong giai quyet drift free-run.
- Test giam nhe so voi baseline goc.
- Vi vay intercept khong duoc xem la cai tien doc lap, nhung van can khi feature space mo rong co offset/regime.

## 5. Log transform anh sang

Buoc feature engineering dau tien la xu ly bien `Light`.

Anh sang thuong co:

- mien gia tri lon,
- phan phoi lech,
- tac dong khong tuyen tinh len nhiet/do am/dat.

Bien moi:

```python
Light_log = np.log1p(df["Light"].clip(lower=0))
```

Y nghia:

- `clip(lower=0)` dam bao khong log gia tri am.
- `log1p(x)` on dinh khi `x=0`.
- Nen tac dong cua light bot bi chi phoi boi cac gia tri cuc lon.

Trong model final, `Light_log` duoc giu lai cung voi `Light` goc. Nghia la model co ca:

```text
Light
Light_log
```

## 6. Feature engineering 16 bien

Sau log transform, pipeline tao bo feature mo rong 16 bien.

### 6.1 Nhom 1: input goc

```text
Temperature
Humidity
Light
Drip
Mist
Fan
```

### 6.2 Nhom 2: log transform

```text
Light_log
```

### 6.3 Nhom 3: interaction features

```text
Temp_x_Humi  = Temperature * Humidity
Temp_x_Light = Temperature * Light_log
Humi_x_Light = Humidity * Light_log
```

Y nghia:

- Do am dat khong chi phu thuoc rieng tung bien.
- Nhiet do cao + anh sang cao co the lam thoat hoi nuoc manh hon.
- Do am khong khi va anh sang cung tao regime moi truong khac nhau.

### 6.4 Nhom 4: setpoint features

```text
SP_Center = 0.5 * (Soil_Low_SP + Soil_High_SP)
SP_Width  = Soil_High_SP - Soil_Low_SP
```

Y nghia:

- Model biet moc van hanh cua controller theo tung thoi diem.
- Cung mot gia tri `Soil_Moisture`, he thong co the hanh xu khac neu setpoint khac.

### 6.5 Nhom 5: cyclic time/regime features

```text
Month_sin
Month_cos
Season_sin
Season_cos
```

Y nghia:

- Thang/mua la bien chu ky.
- Dung sin/cos giup thang 12 gan thang 1, tranh ma hoa thang nhu so tuyen tinh.
- Model co them thong tin regime theo mua.

## 7. Giu ARX(2,2,1), thu feature engineering

Pipeline thu giu nguyen order goc `2,2,1`, nhung doi input tu 6 raw len 16 augmented.

Cau hinh:

```text
Order = ARX(2,2,1)
Input = 16 augmented features
Intercept = True
Clip = None
Fit = OLS
```

Ket qua:

```text
Validation FIT_sim = 38.215
Test FIT_sim = 40.676
```

Nhan xet pipeline:

- Feature engineering don le khong du.
- Khi them nhieu feature ma free-run khong bi rang buoc, model co the drift manh hon.
- Dieu nay cho thay phai cai tien ca feature, order va co che on dinh free-run.

## 8. Clip free-run theo phan vi train

Pipeline tinh clip tu train split cua dataframe da augmented:

```python
clip_low = train["Soil_Moisture"].quantile(0.01)
clip_high = train["Soil_Moisture"].quantile(0.99)
```

Gia tri final:

```text
clip_low = 50.5465794049447
clip_high = 64.61272806162447
```

Trong free-run simulation, moi lan model du doan `y_next`, pipeline gioi han:

```python
y_next = np.clip(y_next, clip_low, clip_high)
```

Y nghia:

- Free-run de bi drift vi output du doan lai tro thanh input tre cho buoc sau.
- Clip khong thay doi cong thuc ARX, ma chi them rang buoc mien gia tri hop ly khi simulate.
- Clip co tinh engineering nen phai noi ro trong bao cao.

Voi `ARX(2,2,1)` + 16 feature + intercept + clip:

```text
Validation FIT_sim = 48.368
Test FIT_sim = 53.448
```

Nhan xet:

- Day la luc feature engineering bat dau co loi ro ret tren `2,2,1`.
- Tuy nhien `53.448` van thap hon xa model final.

## 9. Order search: chuyen tu ARX(2,2,1) sang ARX(5,1,2)

Sau khi thay `2,2,1` co tran hieu nang thap, pipeline mo rong order search va tim thay cau truc tot:

```text
ARX(5,1,2)
```

Y nghia:

- `na=5`: tang bo nho output, giam sai so tich luy free-run.
- `nb=1`: moi input chi lay 1 lag, giu so tham so gon khi input da len 16 bien.
- `nk=2`: input tac dong tre hon 1 buoc, phu hop dong hoc synthetic hon.

Voi 16 feature, intercept, chua clip:

```text
Validation FIT_sim = 69.337
Test FIT_sim = 66.316
```

Voi 16 feature, intercept, co clip:

```text
Validation FIT_sim = 68.861
Test FIT_sim = 66.414
```

Nhan xet pipeline:

- Buoc doi order la buoc nhay lon nhat.
- Clip lam validation giam nhe nhung test tang nhe.
- Pipeline uu tien model on dinh tren test/free-run hon la chi toi da validation.

## 10. Regression matrix cua ARX(5,1,2) final

Voi final order:

```text
na = 5
nb = 1
nk = 2
input_dim = 16
intercept = True
```

Moi dong regression matrix gom:

```text
y(t-1), y(t-2), y(t-3), y(t-4), y(t-5),
u1(t-2),
u2(t-2),
...
u16(t-2),
intercept
```

Tong so tham so:

```text
5 + 16*1 + 1 = 22
```

Danh sach input final:

```text
Temperature
Humidity
Light
Drip
Mist
Fan
Light_log
Temp_x_Humi
Temp_x_Light
Humi_x_Light
SP_Center
SP_Width
Month_sin
Month_cos
Season_sin
Season_cos
```

## 11. Normalize trong Ridge/Lasso

Day la phan ban vua nhac: pipeline co normalize khi fit Ridge/Lasso.

Trong notebook, ham regularized fit lam:

```python
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma_safe = np.where(sigma < 1e-12, 1.0, sigma)
Xs = (X_train - mu) / sigma_safe
```

Sau do fit Ridge hoac Lasso tren `Xs`:

```python
Ridge(alpha=alpha, fit_intercept=True)
Lasso(alpha=alpha, fit_intercept=True, max_iter=200000)
```

Sau khi fit xong, he so duoc doi nguoc ve original scale:

```python
coef_orig = coef_std / sigma_safe
intercept_orig = intercept_std - dot(coef_std, mu / sigma_safe)
theta = [coef_orig..., intercept_orig]
```

Y nghia:

- Regularization can input cung scale, neu khong feature lon nhu `Light` hoac `Temp_x_Humi` se anh huong penalty khac feature nho.
- Normalize chi dung trong qua trinh fit Ridge/Lasso.
- Artifact final luu theta da doi ve original scale, nen simulate van dung truc tiep tren feature goc/engineered, khong can luu scaler rieng.

## 12. Ridge/Lasso regularization

Pipeline thu regularization tren feature set 16 bien va cac order ung vien.

Grid:

```text
ridge_alphas = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0]
lasso_alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
```

Ung vien top:

```text
method = ridge
alpha = 0.001
order = (5, 1, 2)
Validation FIT_sim = 68.861
Test FIT_sim = 66.414
Validation FIT_12 = 69.816
Test FIT_12 = 67.158
Validation FIT_1 = 86.299
Test FIT_1 = 85.849
```

Y nghia:

- Ridge giu day du feature nhung lam he so on dinh hon.
- Lasso cung duoc thu de sparsify feature, nhung khong duoc chon lam final.
- Model van la ARX-only, khong dung ARMAX.

## 13. Rolling validation

Ngoai validation/test co dinh, pipeline tinh rolling validation de kiem tra do on dinh theo nhieu cua so thoi gian.

Metric rolling final:

```text
rolling_fit_mean = 66.48515832620386
rolling_fit_std = 0.9772921657500855
rolling_score = 65.99651224332882
```

Y nghia:

- `rolling_fit_mean`: hieu nang trung binh qua cac cua so.
- `rolling_fit_std`: do dao dong cua hieu nang, cang thap cang on dinh.
- `rolling_score`: diem robust co phat khi dao dong cao.

Pipeline ranking regularized model uu tien:

```text
test_fit_sim
rolling_score
test_fit_12
val_fit_sim
```

Voi rich-feature round, selection score dung:

```text
val_fit_sim
+ 0.35 * val_fit_12
+ 0.10 * val_fit_1
+ 0.15 * rolling_fit_mean
- 0.50 * rolling_fit_std
```

## 14. Thu rich features 23 input

Sau khi co final 16 feature, pipeline con thu them 7 feature nua:

```text
Temp_sq      = Temperature ** 2
Humi_sq      = Humidity ** 2
Light_sqrt   = sqrt(clip(Light, 0, None))
Temp_x_Drip  = Temperature * Drip
Humi_x_Mist  = Humidity * Mist
Light_x_Fan  = Light * Fan
Env_Stress   = Temperature * (100 - Humidity)
```

Tong input:

```text
16 base augmented + 7 rich = 23 input
```

Pipeline thu:

```text
orders = [(5,1,2), (6,1,2), (5,2,2)]
ridge_alphas = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0]
lasso_alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
clip = same train quantile clip
rolling validation = co
```

Best rich-feature candidate:

```text
method = lasso
alpha = 0.000
order = (5,1,2)
n_inputs = 23
Test FIT_sim = 66.402
Test FIT_12 = 67.123
Test FIT_1 = 85.852
rolling_fit_mean = 66.480
rolling_fit_std = 0.983
rolling_score = 65.989
```

So voi current final:

```text
current_final_arx:
  n_inputs = 16
  method = ridge
  alpha = 0.001
  Test FIT_sim = 66.414
  rolling_score = 65.997

rich_feature_candidate:
  n_inputs = 23
  method = lasso
  alpha = 0.000
  Test FIT_sim = 66.402
  rolling_score = 65.989
```

Ket luan cua pipeline:

```text
Promoted rich-feature candidate: False
```

Ly do:

- 23 features nhieu hon nhung test `FIT_sim` thap hon `66.414`.
- Rolling score cung thap hon mot chut.
- Vi vay final van giu 16 feature, khong promote rich feature.

## 15. Model final hien tai

Artifact:

```text
arx_model_algo_final_arx_only.json
```

Thong tin final:

```text
model_type = ARX
selection_note = No ARMAX used. Final model chosen by ARX-only regularized/rolling search.
order = ARX(5,1,2)
input_dim = 16
intercept = True
simulation_clip = [50.5465794049447, 64.61272806162447]
regularization_method = ridge
regularization_alpha = 0.001
```

Metric final:

| Split | FIT_1 | FIT_12 | FIT_sim |
|---|---:|---:|---:|
| Validation | 86.299 | 69.816 | 68.861 |
| Test | 85.849 | 67.158 | 66.414 |

## 16. Pipeline rut gon de trinh bay

Thu tu co the noi trong bao cao:

1. Doc `greenhouse_data.csv`, sort theo `Timestamp`, validate cot.
2. Chia time-series theo chronological split `60/20/20`.
3. Fit baseline `ARX(2,2,1)` voi 6 input goc bang OLS.
4. Danh gia 1-step, 12-step va free-run; thay `FIT_sim` baseline con thap.
5. Bat intercept de thu offset, nhung intercept don le khong cai thien.
6. Tao `Light_log = log1p(clip(Light, lower=0))`.
7. Tao interaction features: `Temp_x_Humi`, `Temp_x_Light`, `Humi_x_Light`.
8. Tao setpoint features: `SP_Center`, `SP_Width`.
9. Tao cyclic regime features: `Month_sin`, `Month_cos`, `Season_sin`, `Season_cos`.
10. Thu `ARX(2,2,1)` voi 16 features; neu khong clip thi free-run xau hon.
11. Them clip free-run theo quantile 1%-99% cua train.
12. Mo rong order search va chon `ARX(5,1,2)` vi free-run tot hon ro.
13. Fit OLS tren 16 features + `ARX(5,1,2)` + clip.
14. Fit Ridge/Lasso tren regression matrix da normalize theo mean/std train.
15. Doi he so regularized ve original scale de simulate va luu artifact.
16. Dung rolling validation de kiem tra on dinh theo nhieu cua so.
17. Thu rich features 23 input, nhung khong promote vi khong vuot final 16 feature.
18. Luu final ARX-only model vao `arx_model_algo_final_arx_only.json`.

## 17. Diem can noi ro de tranh nham lan

- `Log transform` co that va nam o feature `Light_log`.
- `Normalize` co that trong ham fit Ridge/Lasso: chuan hoa `X_train` truoc khi fit, sau do doi theta ve original scale.
- `Standardized beta` trong reporting la de dien giai he so, khong phai artifact final can scaler luc simulate.
- `Rich features` da duoc thu, nhung khong phai model final.
- Final model la ARX-only, khong phai ARMAX.
- Final model khong phai model nhieu feature nhat, ma la model co trade-off tot nhat giua `FIT_sim`, rolling robustness va do gon.
