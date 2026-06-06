# Data Collection Protocol

## Muc tieu

Muc tieu khong phai tao data lam FIT cao, ma tao data giong cach thu thap tren nha kinh mini:

- He van duoc bao ve boi rule-based safety.
- Co cac kich thich nho/an toan de model hoc duoc tac dong cua actuator.
- Moi lenh deu duoc log ro nguon: safety hay excitation.
- Model khong thay soil moisture tuong lai.

## Cac giai doan

1. Commissioning / rule-based safety
   - Kiem tra sensor va actuator.
   - Neu do am dat thap hon nguong, bom bat ngan.
   - Neu do am dat gan cao, chan pulse tuoi.

2. Identification with safe excitation
   - Chen pulse bơm nho theo lich buoi sang/trua/chieu/toi.
   - Pulse khong duoc sinh tu `Soil_Moisture`; no duoc len lich truoc theo clock va random seed.
   - Safety supervisor co quyen chan pulse neu dat qua uot, hoac bat bom neu dat qua kho.

3. Deployment validation
   - Van co rule-based safety.
   - Co mot so pulse validation nho de kiem tra model tren ngay chua train.

## Vi sao phai co rule-based ban dau?

AI/MPC khong the tu dieu khien khi chua co data. Quy trinh dung la:

```text
Rule-based safety -> thu data -> train ARX -> validate -> dua ARX vao MPC
```

Khong duoc noi:

```text
Chua co data -> AI tu biet bom
```

## Vi sao phai co excitation?

Neu actuator chi bat khi dat da kho, data se bi closed-loop feedback. Model se kho tach rieng:

- dat kho lam bom bat,
- hay bom lam dat tang.

Excitation nho/an toan giup tao cac doan actuator thay doi co kiem soat trong vung an toan, tu do ARX hoc duoc dynamic response.

## Dieu kien bao ve truoc hoi dong

- Co split theo thoi gian.
- Co test tren doan khong dung de train.
- Co log source cua lenh actuator.
- Co so sanh ARX voi NARX.
- Co noi ro NARX khong thay truc tiep ARX trong MPC tuyen tinh.

