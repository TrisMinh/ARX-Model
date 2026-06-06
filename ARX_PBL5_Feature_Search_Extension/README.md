# ARX Feature Search Extension

Folder này tách riêng phần dò feature để `ARX_PBL5_Final_Clean` chỉ còn bản ARX chính gọn.

## Vai trò

```text
Đây là thử nghiệm nâng cao, không phải bản ARX final để bảo vệ.
```

Baseline là ARX 17 input:

```text
ARX_na12_nb3_nk2_alpha0.1
FIT_sim = 78.950
```

Ứng viên feature-search:

```text
ARX_na18_nb6_nk2_alpha10
FIT_sim = 83.296
```

## Cấu trúc

```text
ARX_PBL5_Feature_Search_Extension/
  src/
    arx_backbone_pipeline.py
    arx_feature_search.py
  results/
    metrics.json
    single_group_leaderboard.csv
    greedy_steps.csv
    SUMMARY.md
  docs/
    DO_FEATURE_CAI_TIEN_ARX.md
```

`arx_backbone_pipeline.py` là bản copy của ARX final tại thời điểm tách folder, dùng để feature-search chạy độc lập.

## Chạy lại

```powershell
python -B .\ARX_PBL5_Feature_Search_Extension\src\arx_feature_search.py
```

## Câu nói khi bảo vệ

```text
Nhóm có thử thêm feature vật lý và chọn bằng validation robust score. Bản này tăng FIT_sim lên 83.30 nhưng có 26 input, phức tạp hơn bản chính. Vì vậy nhóm giữ ARX 17 input làm sản phẩm chính và để feature-search là thử nghiệm nâng cao.
```
