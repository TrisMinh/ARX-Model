# Huong Dan Lam Lai Notebook ARX (Chi Phan Thuat Toan)

Muc tieu tai lieu nay la giup ban tao lai notebook theo tung cell, chi giu phan mo hinh ARX (data -> estimate -> evaluate -> model search), bo het phan ve do thi va report narrative.

## 1) Nhung cell cu can bo qua

Trong notebook hien tai, bo qua toan bo cell phuc vu ve do thi va ghi chu report.

Ban chi can giu logic tuong duong cac nhom sau:
- Cau hinh va import
- Chay pipeline thuat toan
- Bang tham so va AR roots
- Bang metric du doan
- Bang residual diagnostics
- Bang model search va best candidate

## 2) Cau truc notebook moi (de xuat 10 cell)

Tao notebook moi, sau do dan dung thu tu 10 cell ben duoi.

---

## Cell 1 (Markdown)

```markdown
# ARX Algorithm-Only Notebook

Notebook nay chi chay phan thuat toan nhan dang mo hinh ARX:
1. Nap/sinh du lieu
2. Time-series split
3. Tao ma tran hoi quy ARX
4. Uoc luong OLS
5. Danh gia 1-step, n-step, free-run
6. Residual diagnostics
7. Tim cau truc ARX tot hon theo free-run
```

---

## Cell 2 (Code) - Import va cau hinh

```python
from pathlib import Path

import numpy as np
import pandas as pd

from arx_pipeline import (
    DataConfig,
    SplitConfig,
    ModelConfig,
    load_or_generate_data,
    split_time_series,
    build_regression_matrix,
    estimate_ols,
    build_true_theta,
    summarize_parameters,
    compute_ar_roots,
    evaluate_slice,
    model_selection_search,
    evaluate_candidate_order,
    artifact_payload,
)

# Data config
DATA_CONFIG = DataConfig(
    csv_path=Path("greenhouse_data.csv"),
    generator_script_path=Path("data_generator.py"),
    force_regenerate_from_script=False,
    auto_save_generated_csv=True,
    generated_days=365,
    generated_sampling_seconds=300,
    generated_seed=42,
    generated_start_date="2025-01-01",
)

# Split config
SPLIT_CONFIG = SplitConfig(train_ratio=0.60, val_ratio=0.20)

# Baseline ARX(2,2,1)
MODEL_CONFIG = ModelConfig(
    na=2,
    nb=2,
    nk=1,
    include_intercept=False,
)
```

---

## Cell 3 (Code) - Nap/sinh du lieu va chia tap

```python
df_full, true_params, data_source = load_or_generate_data(DATA_CONFIG)
df_train, df_val, df_test = split_time_series(df_full, SPLIT_CONFIG)

overview = {
    "data_source": data_source,
    "rows_full": len(df_full),
    "rows_train": len(df_train),
    "rows_val": len(df_val),
    "rows_test": len(df_test),
    "timestamp_start": str(df_full["Timestamp"].iloc[0]),
    "timestamp_end": str(df_full["Timestamp"].iloc[-1]),
    "months_present": sorted(int(m) for m in pd.Series(df_full["Month"]).dropna().unique()),
    "seasons_present": sorted(str(s) for s in pd.Series(df_full["Season"]).dropna().unique()),
}

pd.Series(overview)
```

---

## Cell 4 (Code) - Tao regression matrix va estimate OLS

```python
x_train, y_train = build_regression_matrix(df_train, MODEL_CONFIG)

theta_hat, cov_hat, sigma2_hat = estimate_ols(x_train, y_train)
true_theta = build_true_theta(true_params, MODEL_CONFIG)

train_matrix_info = {
    "x_train_shape": x_train.shape,
    "y_train_shape": y_train.shape,
    "rank_x_train": int(np.linalg.matrix_rank(x_train)),
    "n_params": len(MODEL_CONFIG.param_names),
    "cond_xtx": float(np.linalg.cond(x_train.T @ x_train)),
    "sigma2_hat": float(sigma2_hat),
}

pd.Series(train_matrix_info)
```

---

## Cell 5 (Code) - Bang tham so va AR roots

```python
params_df = pd.DataFrame(
    summarize_parameters(theta_hat, cov_hat, MODEL_CONFIG, true_params)
)
roots_df = pd.DataFrame(compute_ar_roots(theta_hat, MODEL_CONFIG))

params_display = params_df[[
    "name",
    "estimate",
    "std",
    "ci95_low",
    "ci95_high",
    "true_value",
    "delta_vs_true",
    "sign_ok",
]].copy()

sign_ok_count = int(params_df["sign_ok"].sum())
sign_total = int(len(params_df))

print(f"Sign recovery: {sign_ok_count}/{sign_total}")
print("\nParameter table:")
display(params_display.round(6))

print("AR roots:")
display(roots_df.round(6))
```

---

## Cell 6 (Code) - Danh gia Train / Validation / Test

```python
train_eval = evaluate_slice("Train", df_train, theta_hat, MODEL_CONFIG, true_theta=true_theta, n_step=12)
val_eval = evaluate_slice("Validation", df_val, theta_hat, MODEL_CONFIG, true_theta=true_theta, n_step=12)
test_eval = evaluate_slice("Test", df_test, theta_hat, MODEL_CONFIG, true_theta=true_theta, n_step=12)

metrics_table = pd.DataFrame([
    {
        "Split": "Train",
        "FIT_1step": train_eval["metrics_1step"]["FIT"],
        "FIT_12step": train_eval["metrics_n_step"]["FIT"],
        "FIT_sim": train_eval["metrics_sim"]["FIT"],
        "Theo_FIT_sim": train_eval.get("theoretical_max_free_run", {}).get("FIT", np.nan),
        "RMSE_1step": train_eval["metrics_1step"]["RMSE"],
        "RMSE_sim": train_eval["metrics_sim"]["RMSE"],
    },
    {
        "Split": "Validation",
        "FIT_1step": val_eval["metrics_1step"]["FIT"],
        "FIT_12step": val_eval["metrics_n_step"]["FIT"],
        "FIT_sim": val_eval["metrics_sim"]["FIT"],
        "Theo_FIT_sim": val_eval.get("theoretical_max_free_run", {}).get("FIT", np.nan),
        "RMSE_1step": val_eval["metrics_1step"]["RMSE"],
        "RMSE_sim": val_eval["metrics_sim"]["RMSE"],
    },
    {
        "Split": "Test",
        "FIT_1step": test_eval["metrics_1step"]["FIT"],
        "FIT_12step": test_eval["metrics_n_step"]["FIT"],
        "FIT_sim": test_eval["metrics_sim"]["FIT"],
        "Theo_FIT_sim": test_eval.get("theoretical_max_free_run", {}).get("FIT", np.nan),
        "RMSE_1step": test_eval["metrics_1step"]["RMSE"],
        "RMSE_sim": test_eval["metrics_sim"]["RMSE"],
    },
])

metrics_table.round(4)
```

---

## Cell 7 (Code) - Residual diagnostics (khong ve hinh)

```python
diag_table = pd.DataFrame([
    {
        "Split": "Validation",
        "mean": val_eval["residual_diagnostics"]["mean"],
        "std": val_eval["residual_diagnostics"]["std"],
        "shapiro_p": val_eval["residual_diagnostics"]["normality"]["shapiro_pvalue"],
        "dagostino_p": val_eval["residual_diagnostics"]["normality"]["dagostino_pvalue"],
        "ljung_box_pass": val_eval["residual_diagnostics"]["ljung_box"]["passes_all_lags"],
        "failed_lags": val_eval["residual_diagnostics"]["ljung_box"]["failed_lags"],
    },
    {
        "Split": "Test",
        "mean": test_eval["residual_diagnostics"]["mean"],
        "std": test_eval["residual_diagnostics"]["std"],
        "shapiro_p": test_eval["residual_diagnostics"]["normality"]["shapiro_pvalue"],
        "dagostino_p": test_eval["residual_diagnostics"]["normality"]["dagostino_pvalue"],
        "ljung_box_pass": test_eval["residual_diagnostics"]["ljung_box"]["passes_all_lags"],
        "failed_lags": test_eval["residual_diagnostics"]["ljung_box"]["failed_lags"],
    },
])

diag_table.round(4)
```

---

## Cell 8 (Code) - Model structure search (na, nb, nk)

```python
selection_df = model_selection_search(
    df_train=df_train,
    df_val=df_val,
    base_model_config=MODEL_CONFIG,
    na_list=[1, 2, 3],
    nb_list=[1, 2, 3],
    nk_list=[1, 2],
)

selection_df.head(10).round(4)
```

---

## Cell 9 (Code) - Danh gia best candidate tren Validation/Test

```python
if selection_df.empty:
    raise RuntimeError("Model selection rong, can kiem tra lai du lieu hoac config")

best_row = selection_df.iloc[0]
best_candidate = evaluate_candidate_order(
    df_train=df_train,
    df_val=df_val,
    df_test=df_test,
    base_model_config=MODEL_CONFIG,
    na=int(best_row["na"]),
    nb=int(best_row["nb"]),
    nk=int(best_row["nk"]),
)

best_cfg = best_candidate["model_config"]

comparison_df = pd.DataFrame([
    {
        "Model": f"Baseline ARX({MODEL_CONFIG.na},{MODEL_CONFIG.nb},{MODEL_CONFIG.nk})",
        "Val_FIT_sim": val_eval["metrics_sim"]["FIT"],
        "Test_FIT_sim": test_eval["metrics_sim"]["FIT"],
        "Val_RMSE_sim": val_eval["metrics_sim"]["RMSE"],
        "Test_RMSE_sim": test_eval["metrics_sim"]["RMSE"],
    },
    {
        "Model": f"Best ARX({best_cfg.na},{best_cfg.nb},{best_cfg.nk})",
        "Val_FIT_sim": best_candidate["val"]["metrics_sim"]["FIT"],
        "Test_FIT_sim": best_candidate["test"]["metrics_sim"]["FIT"],
        "Val_RMSE_sim": best_candidate["val"]["metrics_sim"]["RMSE"],
        "Test_RMSE_sim": best_candidate["test"]["metrics_sim"]["RMSE"],
    },
])

comparison_df.round(4)
```

---

## Cell 10 (Code) - Dong goi ket qua thuat toan thanh artifact JSON

```python
results_algo = {
    "data_source": data_source,
    "data_config": DATA_CONFIG,
    "split_config": SPLIT_CONFIG,
    "model_config": MODEL_CONFIG,
    "df_full": df_full,
    "df_train": df_train,
    "df_val": df_val,
    "df_test": df_test,
    "true_params": true_params,
    "dataset_overview": {
        "rows": int(len(df_full)),
        "timestamp_start": str(df_full["Timestamp"].iloc[0]),
        "timestamp_end": str(df_full["Timestamp"].iloc[-1]),
        "months_present": sorted(int(m) for m in pd.Series(df_full["Month"]).dropna().unique()),
        "seasons_present": sorted(str(s) for s in pd.Series(df_full["Season"]).dropna().unique()),
        "condition_number_xtx": float(np.linalg.cond(x_train.T @ x_train)),
        "rank_x_train": int(np.linalg.matrix_rank(x_train)),
        "n_params": int(len(MODEL_CONFIG.param_names)),
    },
    "theta_hat": theta_hat.tolist(),
    "sigma2": float(sigma2_hat),
    "ar_roots": roots_df.to_dict(orient="records"),
    "parameter_summary": params_df.to_dict(orient="records"),
    "train": train_eval,
    "val": val_eval,
    "test": test_eval,
    "model_selection": selection_df,
    "best_candidate": best_candidate,
}

payload = artifact_payload(results_algo)

with Path("arx_model_algo_only.json").open("w", encoding="utf-8") as f:
    import json
    json.dump(payload, f, indent=2)
    f.write("\n")

print("Saved: arx_model_algo_only.json")
```

---

## 3) Tieu chi dung de tu kiem tra sau khi lam xong

- Cell 4: x_train shape phai hop le va rank_x_train > 0.
- Cell 5: params_df co du ten tham so va co cot sign_ok.
- Cell 6: co du metric cho 3 che do 1-step, 12-step, free-run.
- Cell 7: co ket qua Ljung-Box va normality p-value.
- Cell 8-9: selection_df khong rong va lay duoc best candidate.
- Cell 10: sinh thanh cong file arx_model_algo_only.json.

## 4) Ghi chu de tranh loi khi copy

- Neu ban gap loi thieu cot (Month/Season), hay dam bao file greenhouse_data.csv dung bo sinh du lieu trong du an.
- Neu selection_df rong, thu giam tim kiem (vi du chi na=[1,2], nb=[1,2], nk=[1]) de debug nhanh.
- Neu cond_xtx qua lon, can can nhac standardize input trong phien ban tiep theo (nhung chua can cho baseline algorithm-only).
