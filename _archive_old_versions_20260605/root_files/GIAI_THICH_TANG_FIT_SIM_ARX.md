# Giai thich chi tiet viec tang FIT_sim ARX

## 1. Tom tat ket qua

Baseline ban dau trong notebook:
- Model: ARX(2,2,1), input goc, khong intercept, khong clip
- Validation FIT_sim: 42.959
- Test FIT_sim: 43.875

Model cai thien tot nhat tim duoc:
- Scenario: augmented_features_with_clip
- Cau hinh: ARX(5,1,2), include_intercept=True, co clip free-run
- Input: 16 bien (6 bien goc + bien tuong tac + bien setpoint + bien mua)
- Validation FIT_sim: 68.861
- Test FIT_sim: 66.414

Muc tang:
- Validation: +25.902 diem
- Test: +22.539 diem

Day la muc tang lon, khong phai tang nhe 1-2 diem.

## 2. Da ap dung nhung phuong phap gi de tang FIT_sim

### 2.1 Mo rong tim kiem cau truc ARX
Khong chi giu (na,nb,nk) = (2,2,1), da mo rong grid tim kiem va danh gia tren free-run.

Y nghia:
- FIT_sim phan anh sai so tich luy khi mo hinh tu quy hoi chinh no.
- Cau truc order phu hop free-run khong nhat thiet giong cau truc baseline de de giai thich.

### 2.2 Feature engineering cho dau vao
Da bo sung:
- Light_log
- Temp_x_Humi
- Temp_x_Light
- Humi_x_Light
- SP_Center, SP_Width
- Month_sin, Month_cos
- Season_sin, Season_cos

Y nghia:
- Dua them thong tin regime (mua, setpoint) va quan he phi tuyen yeu.
- ARX van la tuyen tinh theo tham so, nhung co the bieu dien hanh vi phi tuyen thong qua bien bien doi.

### 2.3 Bat intercept
Intercept giup bat offset he thong khi du lieu co dich muc nen.

### 2.4 Clip free-run de giam drift
Dung clip theo phan vi train (Q1% den Q99%) cho Soil_Moisture khi mo phong free-run.

Y nghia:
- Free-run de bi drift khi loi nho bi cong don nhieu buoc.
- Clip dong vai tro rang buoc mien gia tri hop ly theo du lieu train.

## 3. Vi sao 5,1,2 tot hon 2,2,1 trong bai nay

ARX(5,1,2) (voi bo feature mo rong) da dat can bang tot hon giua nho he thong va do tre input:
- na=5: mo hinh nho duoc dong hoc output dai hon, giam loi tich luy khi free-run.
- nb=1: tranh over-parameter hoa phan lag input khi da co nhieu bien dau vao da duoc xu ly.
- nk=2: phu hop hon voi do tre tac dong thuc te trong du lieu synthetic nay.

Noi gon:
- 2,2,1 qua ngan bo nho output cho che do free-run.
- 5,1,2 giam drift tot hon nen FIT_sim tang manh.

## 4. Cau hoi trong tam: Neu giu 2,2,1 roi ap dung cac phuong phap tren thi co tang khong?

Co, co tang. Nhung tang khong du manh bang 5,1,2.

Ablation cong bang (giu nguyen best_cfg = (2,2,1)):

1) fixed_221_raw_no_intercept_no_clip
- Val FIT_sim: 42.959
- Test FIT_sim: 43.875

2) fixed_221_raw_intercept
- Val FIT_sim: 42.486
- Test FIT_sim: 43.399
- Ket qua: intercept don le khong giup, con giam nhe.

3) fixed_221_augmented_intercept
- Val FIT_sim: 38.215
- Test FIT_sim: 40.676
- Ket qua: them nhieu feature ma khong clip de drift free-run xau hon.

4) fixed_221_augmented_intercept_clip
- Val FIT_sim: 48.368
- Test FIT_sim: 53.448
- Ket qua: co tang ro so voi baseline 2,2,1 (Test +9.573 diem), nhung van thap hon rat xa so voi 5,1,2.

So sanh truc tiep tren Test FIT_sim:
- 2,2,1 baseline: 43.875
- 2,2,1 + best enhancement: 53.448
- 5,1,2 + best enhancement: 66.414

=> 2,2,1 co the cai thien, nhung tran hieu nang cua no thap hon cau hinh 5,1,2 trong bai toan nay.

## 5. Tai sao co scenario val tot nhung test rat xau?

Da thay o scenario wide_order_raw_inputs:
- Val FIT_sim: 53.990
- Test FIT_sim: 31.342

Dieu nay la dau hieu overfit va/hoac mo hinh khong on dinh free-run khi gap regime khac.

Y nghia:
- Khong duoc ket luan chi dua tren val.
- Phai uu tien scenario co val tot va test cung tot (generalization).

## 6. Co che chinh tao ra muc tang lon

Muc tang +22.539 tren test den tu su ket hop, khong phai 1 trick don le:
- Feature regime (setpoint + season) -> mo hinh biet boi canh van hanh.
- na lon hon (5) -> giam cong don sai so khi tu hoi quy.
- nk=2 + nb=1 -> do tre input gon va dung hanh vi hon.
- Clip free-run -> cat drift phi thuc te.

Neu chi dung tung phuong phap rieng le, ket qua khong dat muc tang manh nhu tren.

## 7. Huong dan su dung ket qua trong notebook

Trong notebook [ARX_Model_Algo_Only.ipynb](ARX_Model_Algo_Only.ipynb):
- Cell 13: benchmark nhieu scenario cai thien
- Cell 14: tong hop gain baseline vs improved
- Cell 15: xuat artifact da cai thien
- Cell 16: ablation giu co dinh 2,2,1 de tra loi cau hoi cong bang

Artifact da cai thien:
- [arx_model_algo_improved.json](arx_model_algo_improved.json)

## 8. Ket luan ngan gon

- Ban dung: 2,2,1 khong phai khong cai thien duoc.
- Nhung dung: 2,2,1 co tran hieu nang thap hon trong free-run cua bai nay.
- De dat muc tang lon va on dinh tren test, cau hinh 5,1,2 + feature engineering + clip la lua chon tot hon ro ret.
