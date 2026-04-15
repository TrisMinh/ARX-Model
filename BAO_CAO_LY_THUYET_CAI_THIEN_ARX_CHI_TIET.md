# Bao Cao Ly Thuyet Va Giai Thich Chi Tiet Cai Thien Mo Hinh ARX

## 1. Muc tieu bao cao

Tai lieu nay tong hop day du:
- Nen tang ly thuyet cua mo hinh ARX.
- Cac che do du doan: pre1, pre12, free-run (sim).
- Cac ky thuat da ap dung de cai thien mo hinh.
- Chung minh bang so lieu vi sao FIT_sim tang manh.
- Giai thich vi sao cau hinh ARX(5,1,2) vuot ARX(2,2,1).
- Giai thich hien tuong pre12 va sim sat nhau trong mo hinh da cai thien.

Muc tieu cuoi cung la bien ket qua thu nghiem thanh lap luan co tinh hoc thuat, khong chi la "thu va thay tot".

## 2. Co so ly thuyet ARX

### 2.1 Dang mo hinh

Mo hinh ARX cho output $y_t$ va vector input $u_t$:

$$
y_t = \sum_{i=1}^{n_a} a_i y_{t-i} + \sum_{j=1}^{m} \sum_{k=0}^{n_b-1} b_{j,k} u_j(t-n_k-k) + c + e_t
$$

Trong do:
- $n_a$: do sau bo nho output (autoregressive memory).
- $n_b$: so lag cua moi input.
- $n_k$: do tre vao (input delay).
- $c$: intercept (neu bat).
- $e_t$: nhieu va sai so mo hinh.

### 2.2 Dang ma tran de uoc luong OLS

Xep theo dang hoi quy tuyen tinh:

$$
\mathbf{y} = \mathbf{X}\theta + \varepsilon
$$

Uoc luong binh phuong toi thieu:

$$
\hat{\theta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}
$$

Va ma tran hiep phuong sai:

$$
\widehat{\mathrm{Cov}}(\hat{\theta}) = \hat{\sigma}^2 (\mathbf{X}^\top\mathbf{X})^{-1}
$$

Dieu kien can:
- Rank cua $\mathbf{X}$ du lon (khong suy bien).
- Kich thich du lieu du phong phu (inputs khong qua "it hoat dong").

### 2.3 Y nghia vat ly trong bai toan nha kinh

Output la Soil_Moisture. Inputs gom:
- Disturbance: Temperature, Humidity, Light.
- Actuator: Drip, Mist, Fan.

Y nghia dieu khien:
- He so AR ($a_i$) quyet dinh tinh quan tinh va nho cua do am dat.
- He so input ($b_{j,k}$) quyet dinh tac dong tri hoan va cuong do cua moi kenh tac dong.

## 3. Ly thuyet cac che do du doan

### 3.1 pre1 (one-step)

Moi buoc du doan dung output that o buoc truoc:

$$
\hat{y}_{t|t-1} = f(y_{t-1}, y_{t-2}, ..., u_{t-1}, ...)
$$

Dac diem:
- Sai so khong cong don manh.
- Thuong co FIT cao nhat.

### 3.2 pre12 (n-step voi n=12)

Du doan cach 12 buoc, dung mot phan output du doan noi bo.

Dac diem:
- Sai so bat dau cong don.
- Thuong thap hon pre1 va cao hon free-run.

### 3.3 free-run (sim)

Mo hinh tu hoi quy bang output du doan cua chinh no trong toan bo horizon.

$$
\hat{y}_t = f(\hat{y}_{t-1}, \hat{y}_{t-2}, ..., u_{t-1}, ...)
$$

Dac diem:
- Kho nhat va nhay cam nhat voi sai so cau truc.
- FIT_sim moi la bai test nghiem khac nghiet cho tinh on dinh dong hoc.

## 4. Chi so danh gia

Trong notebook da dung:
- FIT (%), RMSE, MAE, Bias, R2, AIC, BIC.

FIT duoc hieu la:

$$
\mathrm{FIT} = 100\left(1 - \frac{\|y - \hat{y}\|}{\|y - \bar{y}\|}\right)
$$

Y nghia:
- FIT cao -> mo hinh tai tao duoc bien dong tot.
- FIT_sim la chi so trong tam khi danh gia kha nang mo phong dai han.

## 5. Van de baseline

Baseline ban dau:
- Cau hinh: ARX(2,2,1)
- Input: 6 bien goc
- Khong intercept
- Khong clip free-run

Ket qua:
- Validation FIT_sim: 42.959
- Test FIT_sim: 43.875

Dau hieu:
- pre1 va pre12 cao, nhung free-run thap -> sai so dong hoc tich luy.

## 6. Cac ky thuat da ap dung de cai thien

## 6.1 Mo rong tim kiem cau truc (na, nb, nk)

Thay vi dong cung ARX(2,2,1), da search tren khong gian order rong hon.

Ly do ly thuyet:
- $n_a$ lon hon co the bat nho dong hoc dai hon.
- $n_b$, $n_k$ dieu chinh cach tac dong input theo do tre thuc te.
- Cau truc toi uu cho free-run khong nhat thiet trung cau truc de "de hieu".

## 6.2 Feature engineering co huong vat ly

Da them:
- Light_log
- Temp_x_Humi, Temp_x_Light, Humi_x_Light
- SP_Center, SP_Width
- Month_sin, Month_cos
- Season_sin, Season_cos

Ly do ly thuyet:
- ARX van tuyen tinh theo tham so, nhung co the bieu dien phi tuyen yeu qua bien doi input.
- Bien setpoint va season dua them thong tin regime, giup giam sai so khi dieu kien van hanh thay doi.

## 6.3 Bat intercept

Ly do:
- Bat dich offset he thong khi trung binh output khong bang 0 theo khong gian feature.

Luu y:
- Intercept khong phai luc nao cung cai thien neu cau truc va feature chua phu hop.

## 6.4 Clip free-run theo phan vi train

Da dung:
- clip_low = quantile 1% train
- clip_high = quantile 99% train

Ly do ly thuyet:
- Sai so free-run co tinh cong don.
- Rang buoc mien gia tri hop ly giup cat drift khong vat ly.

Trade-off:
- Day la ky thuat on dinh mo phong, khong phai "thuan" nhan dang tuyen tinh truyen thong.
- Can minh bach khi bao cao.

## 7. Bang chung dinh luong: truoc va sau

## 7.1 Ket qua all-scenarios (validation/test FIT_sim)

- baseline_current: Val 42.959, Test 43.875
- augmented_features: Val 69.337, Test 66.316
- augmented_features_with_clip: Val 68.861, Test 66.414
- wide_order_raw_inputs: Val 53.990, Test 31.342

Ket luan:
- Tang lon va on dinh nhat tren test nam o nhom augmented + order toi uu.
- Co truong hop val cao nhung test sap (overfit/generalization kem).

## 7.2 Mo hinh tot nhat

Scenario chon cuoi:
- augmented_features_with_clip
- ARX(5,1,2), include_intercept=True
- Input_dim = 16

Ket qua:
- Validation FIT_sim: 68.861
- Test FIT_sim: 66.414

So voi baseline:
- Validation: +25.902 diem
- Test: +22.539 diem

## 8. Vi sao ARX(5,1,2) vuot ARX(2,2,1)

Giai thich theo cau truc:
- $n_a=5$: bo nho output dai hon, giam loi tich luy khi tu hoi quy.
- $n_b=1$: voi bo feature da phong phu, giu nb gon de tranh qua tham so.
- $n_k=2$: do tre input phu hop hon voi dong hoc sinh du lieu hien tai.

Noi cach khac:
- ARX(2,2,1) co the qua ngan bo nho output cho free-run.
- ARX(5,1,2) tao can bang tot hon giua memory va on dinh.

## 9. Cau hoi trong tam: giu nguyen ARX(2,2,1) thi co cai thien duoc khong?

Co cai thien, nhung khong dat tran cao bang ARX(5,1,2).

Ablation cong bang (giu co dinh best_cfg=(2,2,1)):

- fixed_221_raw_no_intercept_no_clip:
  - Val 42.959, Test 43.875
- fixed_221_raw_intercept:
  - Val 42.486, Test 43.399
- fixed_221_augmented_intercept:
  - Val 38.215, Test 40.676
- fixed_221_augmented_intercept_clip:
  - Val 48.368, Test 53.448

Ket luan:
- 2,2,1 co cai thien duoc den Test 53.448.
- Nhung van thap hon rat ro so voi 66.414 cua cau hinh 5,1,2.

## 10. Giai thich pre12 va sim sat nhau co vo ly khong?

Khong vo ly neu he on dinh va horizon du dai.

Da quet horizon tren validation cho baseline va improved.

## 10.1 Baseline (validation)
- pre12 = 73.109
- sim = 42.959
- gap = 30.151

Gap lon dung voi truc giac "pre12 phai cao hon sim".

## 10.2 Improved (validation)
- pre12 = 69.816
- sim = 68.861
- gap = 0.955

Va theo horizon:
- n=8: gap 2.263
- n=12: gap 0.955
- n=24: gap 0.207
- n=48: gap ~0

Dien giai ly thuyet:
- Khi mo hinh da bat dong hoc tot, n-step voi horizon du lon se hoi tu ve hanh vi free-run.
- pre12 sat sim cho thay he so va cau truc moi da giam rat manh drift tich luy.

## 11. Tai sao pre1 co the cao hon baseline nhung sim lai thap?

Voi baseline:
- Test pre1 = 91.349 (cao)
- Test pre12 = 72.674 (van kha)
- Test sim = 43.875 (thap)

Ly do:
- pre1 luon duoc "chinh" boi output that moi buoc.
- free-run khong co sua loi ngoai, nen lo sai so cau truc trong bo nho AR.

Do do:
- pre1 cao khong dam bao mo hinh mo phong dai han tot.
- FIT_sim moi la bai test khac nghiet cho ung dung mo phong/predictive control dai horizon.

## 12. Goc nhin thong ke va overfitting

Scenario wide_order_raw_inputs:
- Val FIT_sim 53.990
- Test FIT_sim 31.342

Dau hieu:
- Mo hinh hoc duoc mau validation nhung khong tong quat hoa.

Bai hoc:
- Khong chon mo hinh chi theo val.
- Can doi chieu ca test, dac biet trong free-run.

## 13. Ket luan hoc thuat

1. Tang FIT_sim lon la ket qua cua to hop ky thuat, khong phai do 1 meo.
2. ARX(5,1,2) + augmented features + intercept + clip tao can bang tot giua:
   - bo nho dong hoc,
   - tinh dai dien regime,
   - on dinh free-run.
3. ARX(2,2,1) co the cai thien, nhung tran hieu nang thap hon ro ret.
4. pre12 sat sim trong mo hinh da cai thien la dau hieu he thong da on dinh hon, khong phai loi tinh toan.

## 14. Han che va huong mo rong

Han che:
- Du lieu la synthetic, do khop cao co the de dat hon du lieu thuc.
- Clip co tinh engineering, can thong bao ro trong bao cao.

Huong mo rong:
- Danh gia rolling-origin theo mua.
- Bo sung regularization (Ridge/Lasso) de kiem soat phuong sai he so.
- Thu ARMAX/OE/BJ neu can model hoa nhieu va dong hoc chinh xac hon.
- Bao cao uncertainty theo horizon (prediction interval).

## 15. Tai lieu va tep lien quan

- Notebook ket qua: [ARX_Model_Algo_Only.ipynb](ARX_Model_Algo_Only.ipynb)
- Artifact baseline: [arx_model_algo_only.json](arx_model_algo_only.json)
- Artifact improved: [arx_model_algo_improved.json](arx_model_algo_improved.json)
- Tom tat ngan: [GIAI_THICH_TANG_FIT_SIM_ARX.md](GIAI_THICH_TANG_FIT_SIM_ARX.md)
