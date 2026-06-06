# Nhật ký rà soát bản residual final

## 1. Lý do tạo folder này

Trước đó residual nằm ở nhiều chỗ:

- một folder extension;
- một số doc trong folder ARX thuần;
- kết quả so sánh nằm rải rác.

Điều này dễ làm người đọc không biết đâu là bản bảo vệ chính. Vì vậy tạo folder:

```text
ARX_PBL5_Residual_Final_Clean
```

## 2. Quyết định hiện tại

Trong folder này:

```text
Hybrid ARX residual correction là bản bảo vệ chính.
ARX thuần 16 input là backbone và nền so sánh.
```

## 3. Kết quả đã kiểm tra

```text
ARX backbone FIT_sim = 82.496
Hybrid residual FIT_sim = 83.433
Gain = 0.937 điểm
```

## 4. Ranh giới học thuật

Không nói:

```text
ARX thuần đạt 83.43.
```

Nói đúng:

```text
Hybrid ARX residual correction đạt 83.43, với ARX làm backbone.
```

## 5. Checklist trước khi nộp

- [x] Có code chạy lại.
- [x] Có kết quả `metrics.json`.
- [x] Có `leaderboard.csv` để kiểm tra candidate.
- [x] Có docs lý thuyết.
- [x] Có docs thu dữ liệu thật.
- [x] Có câu hỏi bảo vệ.
- [x] Có tự phản biện.
- [ ] Có dữ liệu phần cứng thật.
- [ ] Có kết quả test phần cứng thật.
