# Defense Notes

## Neu thay hoi: Data nay la gia lap, co dung khong?

Tra loi:

Data nay la protocol-grade simulation, dung de chung minh pipeline nhan dang he thong. Trong trien khai that, nhom se thu cung cac truong log tren nha kinh mini va retrain ARX. Gia lap khong duoc xem la thay the data that, ma la buoc kiem thu thuat toan va quy trinh.

## Neu thay hoi: Vi sao ban dau van can rule-based?

Tra loi:

Vi AI/MPC can model, model can data. Khi chua co data, phai dung rule-based safety de van hanh an toan va tao log ban dau. Sau khi co ARX da validate, MPC moi dung ARX de du doan va toi uu lenh bom.

## Neu thay hoi: Chen pulse co phai gian lan khong?

Tra loi:

Khong. Day la persistent excitation trong system identification. Pulse duoc lap lich truoc, nho, co gioi han an toan, va khong dung output tuong lai. Neu dat qua uot/kho thi safety supervisor chan hoac ghi de lenh.

## Neu thay hoi: Vi sao khong dung NARX neu no phi tuyen?

Tra loi:

NARX co the manh hon ve du bao phi tuyen, nhung MPC hien tai trong bao cao dua tren ARX tuyen tinh/RLS. Doi sang NARX la doi kien truc dieu khien sang NMPC hoac local linearization. Vi de tai chinh la ARX va nha kinh mini can giai thich/deploy duoc, ARX la lua chon chinh.

## Cau ket luan nen dung

Nhom uu tien ARX vi co cau truc ro rang, phu hop MPC tuyen tinh va de trien khai tren he nho. NARX duoc dung nhu doi chung de kiem tra phi tuyen; neu NARX khong vuot ro rang o free-run thi khong co ly do doi kien truc dieu khien.

