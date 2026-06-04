# So sanh tung buoc cai tien ARX: tu ARX(2,2,1) den ARX(5,1,2)

Tai lieu nay chi tap trung vao 2 cau truc:

- `ARX(2,2,1)`: model goc va cac bien the cai tien tren dung cau truc goc.
- `ARX(5,1,2)`: model cai tien tot nhat hien tai.

Muc tieu la nhin ro tung buoc cai tien da anh huong den hieu qua nhu the nao, dac biet theo `FIT_sim` vi day la metric quan trong nhat cho mo phong free-run.

## 1. Cach doc ket qua

Trong notebook va artifact, cac metric chinh gom:

- `FIT_1`: du doan 1-step ahead. Moi buoc duoc cap lai output that nen thuong cao.
- `FIT_12`: du doan nhieu buoc ngan han, o day la 12 buoc.
- `FIT_sim`: free-run simulation. Mo hinh tu dung output du doan cua chinh no, nen sai so co the tich luy. Day la metric nen uu tien khi danh gia kha nang mo phong dong hoc.

Trong cac bang ben duoi, `Val FIT_sim` dung de chon va so sanh tren validation, `Test FIT_sim` dung de kiem tra kha nang generalize.

## 2. Cong thuc ARX dang dung

Dang tong quat cua ARX:

```text
y(t) = a1*y(t-1) + ... + ana*y(t-na)
     + b1*u(t-nk) + ... + bnb*u(t-nk-nb+1)
     + intercept
     + e(t)
```

Trong do:

- `na`: so lag cua output `Soil_Moisture`.
- `nb`: so lag cua moi input.
- `nk`: delay truoc khi input tac dong vao output.
- `u`: cac input nhu nhiet do, do am, anh sang, actuator va cac feature engineer.
- `intercept`: he so offset neu bat.

## 3. Model goc: ARX(2,2,1)

### 3.1 Cau hinh

Model goc trong notebook:

```text
Order: ARX(2,2,1)
Input: 6 bien goc
Intercept: khong
Free-run clip: khong
Regularization: khong, OLS
```

6 bien goc:

```text
Temperature, Humidity, Light, Drip, Mist, Fan
```

Ket qua:

| Buoc | Cau hinh | Val FIT_sim | Test FIT_sim | Chenh test so voi goc |
|---|---|---:|---:|---:|
| 1 | `ARX(2,2,1)` + input goc + no intercept + no clip | 42.959 | 43.875 | 0.000 |

### 3.2 Nhan xet

Model nay la baseline tot de bat dau vi no gon, de giai thich va dung dung dang ARX co ban. Tuy nhien `FIT_sim` thap hon nhieu so voi cac metric ngan han. Dieu nay cho thay model co the du doan gan tot, nhung khi free-run thi loi bi tich luy.

Nguyen nhan chinh:

- `na=2` chi nho 2 buoc output gan nhat, co the qua ngan cho dong hoc do am dat.
- `nb=2` voi 6 input goc van chua co thong tin regime nhu setpoint va mua.
- `nk=1` co the chua khop delay tac dong that trong du lieu.
- Khong co clip nen free-run de drift khoi mien gia tri hop ly.

## 4. Giu ARX(2,2,1), them intercept

### 4.1 Cau hinh

```text
Order: ARX(2,2,1)
Input: 6 bien goc
Intercept: co
Free-run clip: khong
Regularization: khong, OLS
```

Ket qua:

| Buoc | Cau hinh | Val FIT_sim | Test FIT_sim | Chenh test so voi goc |
|---|---|---:|---:|---:|
| 1 | `ARX(2,2,1)` + input goc + no intercept + no clip | 42.959 | 43.875 | 0.000 |
| 2 | `ARX(2,2,1)` + input goc + intercept + no clip | 42.486 | 43.399 | -0.476 |

### 4.2 Nhan xet

Intercept don le khong giup. Test `FIT_sim` giam nhe tu `43.875` xuong `43.399`.

Y nghia:

- Van de lon khong nam o offset hang so.
- Cau truc dong hoc va feature dau vao moi la diem nghen chinh.
- Them intercept khi feature chua du thong tin co the lam model fit offset nhung khong giam drift free-run.

## 5. Giu ARX(2,2,1), them feature engineering

### 5.1 Feature da them

Bo feature engineer gom:

```text
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

Cong thuc y tuong:

```python
Light_log = log1p(max(Light, 0))
Temp_x_Humi = Temperature * Humidity
Temp_x_Light = Temperature * Light_log
Humi_x_Light = Humidity * Light_log
SP_Center = (Soil_Low_SP + Soil_High_SP) / 2
SP_Width = Soil_High_SP - Soil_Low_SP
Month_sin, Month_cos = cyclic encoding cua thang
Season_sin, Season_cos = cyclic/regime encoding cua mua
```

### 5.2 Cau hinh

```text
Order: ARX(2,2,1)
Input: 16 bien
Intercept: co
Free-run clip: khong
Regularization: khong, OLS
```

Ket qua:

| Buoc | Cau hinh | Val FIT_sim | Test FIT_sim | Chenh test so voi goc |
|---|---|---:|---:|---:|
| 1 | `ARX(2,2,1)` + input goc + no intercept + no clip | 42.959 | 43.875 | 0.000 |
| 2 | `ARX(2,2,1)` + input goc + intercept + no clip | 42.486 | 43.399 | -0.476 |
| 3 | `ARX(2,2,1)` + augmented features + intercept + no clip | 38.215 | 40.676 | -3.199 |

### 5.3 Nhan xet

Them feature ma khong clip lam ket qua free-run xau hon. Test `FIT_sim` giam xuong `40.676`.

Day la diem quan trong: feature engineering khong tu dong lam model free-run tot hon. Khi them nhieu bien, mo hinh co them do linh hoat, nhung neu cau truc lag van ngan va khong co rang buoc mien gia tri, sai so co the tich luy manh hon.

Y nghia:

- Feature moi co thong tin huu ich, nhung `ARX(2,2,1)` chua du kha nang dung chung on dinh.
- Free-run nhay cam hon 1-step prediction.
- Can them co che kiem soat drift, o day la clip.

## 6. Giu ARX(2,2,1), them feature engineering va clip

### 6.1 Cau hinh

```text
Order: ARX(2,2,1)
Input: 16 bien
Intercept: co
Free-run clip: co
Regularization: khong, OLS
```

Clip free-run gioi han `Soil_Moisture` du doan trong mien hop ly lay tu train, theo phan vi gan Q1% den Q99%.

Ket qua:

| Buoc | Cau hinh | Val FIT_sim | Test FIT_sim | Chenh test so voi goc |
|---|---|---:|---:|---:|
| 1 | `ARX(2,2,1)` + input goc + no intercept + no clip | 42.959 | 43.875 | 0.000 |
| 2 | `ARX(2,2,1)` + input goc + intercept + no clip | 42.486 | 43.399 | -0.476 |
| 3 | `ARX(2,2,1)` + augmented features + intercept + no clip | 38.215 | 40.676 | -3.199 |
| 4 | `ARX(2,2,1)` + augmented features + intercept + clip | 48.368 | 53.448 | +9.573 |

### 6.2 Nhan xet

Day la cai tien that su dau tien tren cau truc `ARX(2,2,1)`. Test `FIT_sim` tang tu `43.875` len `53.448`, tang `+9.573` diem.

Y nghia:

- Feature engineering co tac dung khi di kem rang buoc free-run.
- Clip cat cac gia tri drift phi thuc te, giup mo phong khong bi chay xa mien du lieu.
- Tuy nhien `53.448` van thap hon ro so voi model tot nhat `66.414`.

Ket luan rieng cho `ARX(2,2,1)`:

- Co the cai thien `2,2,1`.
- Tran hieu nang cua `2,2,1` trong bai nay khong cao bang `5,1,2`.
- Buoc co loi nhat voi `2,2,1` la `augmented features + intercept + clip`, khong phai intercept don le.

## 7. Chuyen cau truc sang ARX(5,1,2)

### 7.1 Vi sao doi order

Sau khi thay `2,2,1` van bi gioi han, notebook mo rong search order va tim thay `ARX(5,1,2)` tot hon cho free-run.

Giai thich tung tham so:

- `na=5`: dung 5 lag cua output, tang bo nho dong hoc cua `Soil_Moisture`.
- `nb=1`: moi input chi lay 1 lag. Khi da co 16 feature, giu `nb` gon de tranh qua nhieu tham so.
- `nk=2`: input tac dong sau 2 buoc, phu hop delay hon `nk=1` trong du lieu nay.

Noi gon: `ARX(5,1,2)` nho output dai hon, input gon hon va delay hop ly hon.

## 8. ARX(5,1,2) voi augmented features

### 8.1 Cau hinh

```text
Order: ARX(5,1,2)
Input: 16 bien
Intercept: co
Free-run clip: khong
Regularization: khong, OLS
```

Ket qua ghi trong bao cao:

| Buoc | Cau hinh | Val FIT_sim | Test FIT_sim | Chenh test so voi goc |
|---|---|---:|---:|---:|
| 1 | `ARX(2,2,1)` + input goc + no intercept + no clip | 42.959 | 43.875 | 0.000 |
| 4 | `ARX(2,2,1)` + augmented features + intercept + clip | 48.368 | 53.448 | +9.573 |
| 5 | `ARX(5,1,2)` + augmented features + intercept + no clip | 69.337 | 66.316 | +22.441 |

### 8.2 Nhan xet

Chi rieng viec doi order sang `5,1,2` trong khong gian feature mo rong da dua test `FIT_sim` len `66.316`.

Day la buoc nhay lon nhat:

- So voi baseline `2,2,1`: tang `+22.441` diem test.
- So voi `2,2,1` da clip: tang them `+12.868` diem test.

Y nghia:

- Gioi han chinh cua model goc la cau truc dong hoc, khong chi la feature.
- `na=5` giup giam sai so tich luy trong free-run.
- `nb=1` lam feature set gon hon theo lag, tranh mo hinh qua nang.
- `nk=2` lam delay input hop ly hon.

## 9. ARX(5,1,2) voi augmented features va clip

### 9.1 Cau hinh

```text
Order: ARX(5,1,2)
Input: 16 bien
Intercept: co
Free-run clip: co
Regularization: khong, OLS
```

Ket qua:

| Buoc | Cau hinh | Val FIT_sim | Test FIT_sim | Chenh test so voi goc |
|---|---|---:|---:|---:|
| 1 | `ARX(2,2,1)` + input goc + no intercept + no clip | 42.959 | 43.875 | 0.000 |
| 4 | `ARX(2,2,1)` + augmented features + intercept + clip | 48.368 | 53.448 | +9.573 |
| 5 | `ARX(5,1,2)` + augmented features + intercept + no clip | 69.337 | 66.316 | +22.441 |
| 6 | `ARX(5,1,2)` + augmented features + intercept + clip | 68.861 | 66.414 | +22.539 |

### 9.2 Nhan xet

Voi `ARX(5,1,2)`, clip khong tao buoc nhay lon nhu tren `2,2,1`, nhung van tang nhe test:

```text
66.316 -> 66.414, tang +0.098 diem
```

Validation giam nhe:

```text
69.337 -> 68.861, giam -0.476 diem
```

Dieu nay chap nhan duoc vi test moi la bang kiem tra generalization. Clip giup mo hinh on dinh hon tren free-run, du validation co giam nhe.

## 10. Ban final hien tai: ARX(5,1,2) + Ridge + rolling search

### 10.1 Artifact final

File final:

```text
arx_model_algo_final_arx_only.json
```

Ghi chu trong artifact:

```text
No ARMAX used. Final model chosen by ARX-only regularized/rolling search.
```

Cau hinh:

```text
Model type: ARX
Order: ARX(5,1,2)
Input: 16 bien
Intercept: co
Free-run clip: [50.5465794049447, 64.61272806162447]
Regularization: Ridge
Alpha: 0.001
```

Metric final:

| Split | FIT_1 | FIT_12 | FIT_sim |
|---|---:|---:|---:|
| Validation | 86.299 | 69.816 | 68.861 |
| Test | 85.849 | 67.158 | 66.414 |

### 10.2 Y nghia Ridge va rolling search

Ridge khong doi ban chat ARX. Mo hinh van tuyen tinh theo tham so, khong phai ARMAX/OE/BJ.

Ridge them penalty vao he so:

```text
min ||y - X theta||^2 + alpha * ||theta||^2
```

Y nghia:

- Giam phuong sai he so khi feature set lon.
- Giam rui ro he so qua nhay voi train.
- Phu hop khi co feature tuong tac va feature theo regime.

Rolling search dung de chon model on dinh hon theo nhieu cua so thoi gian, thay vi chi nhin mot validation split duy nhat.

Trong artifact final, Ridge `alpha=0.001` giu metric bang voi ban `ARX(5,1,2)` clip tot nhat:

```text
Validation FIT_sim = 68.861
Test FIT_sim = 66.414
```

## 11. Bang tong hop tung buoc

| Buoc | Order | Input | Intercept | Clip | Regularization | Val FIT_sim | Test FIT_sim | Test gain vs baseline |
|---:|---|---|---|---|---|---:|---:|---:|
| 1 | `2,2,1` | 6 raw | No | No | OLS | 42.959 | 43.875 | 0.000 |
| 2 | `2,2,1` | 6 raw | Yes | No | OLS | 42.486 | 43.399 | -0.476 |
| 3 | `2,2,1` | 16 augmented | Yes | No | OLS | 38.215 | 40.676 | -3.199 |
| 4 | `2,2,1` | 16 augmented | Yes | Yes | OLS | 48.368 | 53.448 | +9.573 |
| 5 | `5,1,2` | 16 augmented | Yes | No | OLS | 69.337 | 66.316 | +22.441 |
| 6 | `5,1,2` | 16 augmented | Yes | Yes | OLS | 68.861 | 66.414 | +22.539 |
| 7 | `5,1,2` | 16 augmented | Yes | Yes | Ridge `alpha=0.001` + rolling search | 68.861 | 66.414 | +22.539 |

## 12. Ket luan de dua vao bao cao

Ket qua cho thay cai tien khong den tu mot trick rieng le. Intercept don le khong giup. Feature engineering don le tren `ARX(2,2,1)` con lam free-run xau hon neu khong co clip. Khi them clip, `ARX(2,2,1)` cai thien ro, tu test `FIT_sim=43.875` len `53.448`.

Tuy nhien, buoc quan trong nhat la doi cau truc dong hoc sang `ARX(5,1,2)`. Voi cung bo feature mo rong, `ARX(5,1,2)` dat test `FIT_sim=66.316` khi chua clip va `66.414` khi co clip. Dieu nay chung minh `ARX(2,2,1)` co the cai thien, nhung tran hieu nang thap hon `ARX(5,1,2)`.

Ban max hien tai la `ARX(5,1,2)` voi 16 feature, intercept, clip free-run, Ridge `alpha=0.001` va chon bang regularized/rolling search. Day van la model ARX-only, khong dung ARMAX.

## 13. File lien quan trong repo

- Notebook co code feature engineering va benchmark: `ARX_Model_Algo_Only.ipynb`
- Artifact final: `arx_model_algo_final_arx_only.json`
- Artifact improved day du hon: `arx_model_algo_improved.json`
- Giai thich ngan gon ve tang FIT_sim: `GIAI_THICH_TANG_FIT_SIM_ARX.md`
- Bao cao ly thuyet chi tiet: `BAO_CAO_LY_THUYET_CAI_THIEN_ARX_CHI_TIET.md`
