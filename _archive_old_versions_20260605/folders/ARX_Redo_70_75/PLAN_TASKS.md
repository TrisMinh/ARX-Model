# Plan va task

## Muc tieu

Tao lai pipeline ARX du doan do am dat, co ket qua `FIT_sim` 70-75% tren danh gia khong leakage.

## Tieu chi dung

- Split theo thoi gian 60/20/20.
- Train: 2025-01-01 den 2025-08-07.
- Validation: 2025-08-08 den 2025-10-01.
- Test: 2025-10-01 den 2025-12-31.
- Khong shuffle.
- Khong loc/cat test.
- Feature scaling fit tren train.
- Chon model/shrink theo validation.
- Test chi dung de bao cao.

## Task

- [x] Doc project cu va xac dinh baseline.
- [x] Kiem tra data generator, cot du lieu, split, metric.
- [x] Thu ARX OLS/ridge/order/feature va xac dinh tran ARX thuan.
- [x] Thu residual correction khong leakage target.
- [x] Chon ban conservative: residual input lag tu `t-2` tro ve truoc.
- [x] Tao pipeline co the chay lai.
- [x] Bat dau vong V75: causal history residual va diagnostic future-actuator upper bound.
- [x] Xac dinh mốc 75 chi dat khi cho model thay actuator tuong lai; track nay khong production-safe neu actuator tuong lai chua co san.
- [ ] Neu dua vao san xuat: them job retrain dinh ky va monitor drift.

## Ket qua ky vong

- ARX backbone: test `FIT_sim` khoang 66-67%.
- Hybrid ARX residual conservative: validation/test `FIT_sim` tren 70%.
- Artifact day du trong `results/`.
