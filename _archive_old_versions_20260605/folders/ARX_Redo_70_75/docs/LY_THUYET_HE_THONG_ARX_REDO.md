# Ly thuyet he thong ARX redo

## 1. Bai toan

Ta can du doan do am dat `Soil_Moisture` trong nha kinh tu cac bien moi truong va actuator:

- Disturbance: `Temperature`, `Humidity`, `Light`.
- Actuator: `Drip`, `Mist`, `Fan`.
- Lich/che do van hanh: setpoint dat, thang, mua.

Chi so chinh la `FIT_sim`:

```text
FIT = 100 * (1 - ||y - y_hat|| / ||y - mean(y)||)
```

`sim` la che do kho nhat vi output du doan o buoc truoc duoc dua lai vao model.

## 2. ARX backbone

Dang ARX:

```text
y(t) = sum_i a_i y(t-i) + sum_j sum_k b_jk u_j(t-nk-k) + c + e(t)
```

Trong ban redo:

- `na = 5`: bo nho output dai hon baseline.
- `nb = 3`: moi input co 3 lag.
- `nk = 2`: tranh tac dong input qua som, hop ly voi dong hoc dat/am.
- Co intercept.
- Z-score chi fit tren train.
- Free-run co clip theo quantile 1%-99% cua train de tranh gia tri phi vat ly.

## 3. Vi sao ARX thuan bi dung tran

Du lieu co nhieu nhieu ngau nhien va dieu khien hysteresis. OLS ARX toi uu one-step rat tot nhung free-run dai han tich luy sai so. Ket qua audit cho thay ARX thuan tot nhat van quanh 66-68% tren test, nen ep ARX thuan len 70+ bang cach chon lai test/split se khong sach.

## 4. Residual correction hop le

Residual model hoc:

```text
r(t) = y_true(t) - y_arx_sim(t)
y_final(t) = y_arx_sim(t) + shrink * r_hat(t)
```

De tranh leakage:

- Residual khong dung `Soil_Moisture` that tai thoi diem t trong validation/test.
- Residual chi dung quy dao `y_arx_sim` va cac lag cua input.
- Cau hinh conservative chi dung input lag `[2, 3, 6, 12, 24]`.
- Shrink duoc chon theo validation, test chi de bao cao cuoi.

## 5. Ung dung thuc te

Trong van hanh that:

- Sensor va actuator log duoc cap nhat lien tuc.
- Model co the chay free-run cho horizon ngan/trung han khi da co kich ban input/lich tuoi.
- Nen retrain dinh ky khi them du lieu moi.
- Neu can du bao tuong lai khong co input moi truong, can them module du bao/lich input; pipeline nay gia dinh chuoi exogenous input da biet hoac duoc du bao.

