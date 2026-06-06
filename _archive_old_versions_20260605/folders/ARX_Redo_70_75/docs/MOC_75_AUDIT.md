# Audit moc 75

## Dieu kien sach

Moc 75 chi duoc cong nhan cho production neu:

- Khong dung `Soil_Moisture` tuong lai.
- Khong cat bo test.
- Khong chon model theo test.
- Input dung tai thoi diem du doan phai that su co san trong van hanh.

## Ket qua vong V75

Co hai track:

1. `causal_history`: dung ARX simulated trajectory va input qua khu. Day la track production-safe.
2. `diagnostic_future_actuator`: cho residual thay input tuong lai, gom ca actuator. Track nay chi la upper bound, khong production-safe neu actuator tuong lai chua biet truoc.

Ket qua chi tiet nam trong `results_v75/`.

## Ket luan ky thuat

Track production-safe cai thien tu test `FIT_sim = 70.294%` len khoang `71.3%`.

Track diagnostic vuot 75%, nhung ly do la future actuator logs chua thong tin feedback tu controller. Trong du lieu nay, `Drip`, `Mist`, `Fan` duoc kich hoat dua tren trang thai he thong, nen actuator tuong lai co the vo tinh tiet lo thong tin ve do am dat hien tai. Neu dung no de du doan backward/offline thi FIT cao, nhung neu can forecast online thi day la leakage theo nghia thuc te.

De dat 75% mot cach production-safe, can mot trong cac dieu kien sau:

- Co lich actuator tuong lai that su duoc biet truoc doc lap voi `Soil_Moisture` can du doan.
- Co module du bao actuator/controller tuong lai duoc huan luyen rieng va khong dung test label.
- Co them sensor/feature giai thich nhieu ngau nhien tich luy trong dat.
- Doi che do danh gia sang rolling forecast co sensor update dinh ky, neu day moi la workflow thuc te.

