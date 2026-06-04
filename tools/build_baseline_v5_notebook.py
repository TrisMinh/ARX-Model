from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path("ARX_Model_Version") / "ARX_Baseline_V5.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }

    nb.cells = [
        md(
            """
# ARX Baseline V5

V5 giữ nguyên V4 nhưng mở rộng input từ 6 biến gốc lên 16 feature:

- `ARX(2,2,1)`
- 6 input gốc + 10 feature engineer
- Ridge
- không Lasso
- không clip free-run
- z-score normalization bằng thống kê train
- `include_intercept=True`
"""
        ),
        md(
            """
## 1. Import và cấu hình

Notebook có thể chạy từ project root hoặc từ thư mục `ARX_Model_Version`.
"""
        ),
        code(
            """
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WORK_DIR = Path.cwd()
PROJECT_ROOT = WORK_DIR.parent if WORK_DIR.name == "ARX_Model_Version" else WORK_DIR
VERSION_DIR = PROJECT_ROOT / "ARX_Model_Version"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arx_pipeline import (
    DataConfig,
    SplitConfig,
    ModelConfig,
    load_or_generate_data,
    split_time_series,
    summarize_dataset_behavior,
    build_regression_matrix,
    estimate_ols,
    build_true_theta,
    evaluate_slice,
    summarize_parameters,
    compute_ar_roots,
    compute_metrics,
)

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 160)
"""
        ),
        md("## 2. Cấu hình V5"),
        code(
            """
DATA_CONFIG = DataConfig(
    csv_path=PROJECT_ROOT / "greenhouse_data.csv",
    generator_script_path=PROJECT_ROOT / "data_generator.py",
    force_regenerate_from_script=False,
    auto_save_generated_csv=True,
)

SPLIT_CONFIG = SplitConfig(train_ratio=0.60, val_ratio=0.20)

BASELINE_INPUT_COLS = (
    "Temperature",
    "Humidity",
    "Light",
    "Drip",
    "Mist",
    "Fan",
)

AUGMENTED_INPUT_COLS = (
    *BASELINE_INPUT_COLS,
    "Light_log",
    "Temp_x_Humi",
    "Temp_x_Light",
    "Humi_x_Light",
    "SP_Center",
    "SP_Width",
    "Month_sin",
    "Month_cos",
    "Season_sin",
    "Season_cos",
)

V5_CONFIG = ModelConfig(
    na=2,
    nb=2,
    nk=1,
    include_intercept=True,
    input_cols=AUGMENTED_INPUT_COLS,
    output_col="Soil_Moisture",
    simulation_clip=None,
)

SCALE_COLS = ("Soil_Moisture", *AUGMENTED_INPUT_COLS)
RIDGE_ALPHAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
V5_CONFIG
"""
        ),
        md("## 3. Load data và chia train/validation/test"),
        code(
            """
df_full, true_params, data_source = load_or_generate_data(DATA_CONFIG)
df_train, df_val, df_test = split_time_series(df_full, SPLIT_CONFIG)

overview = pd.DataFrame([
    summarize_dataset_behavior(df_full, "Full"),
    summarize_dataset_behavior(df_train, "Train"),
    summarize_dataset_behavior(df_val, "Validation"),
    summarize_dataset_behavior(df_test, "Test"),
])

print("Nguồn dữ liệu:", data_source)
print("Số dòng:", len(df_full), "| Train:", len(df_train), "| Validation:", len(df_val), "| Test:", len(df_test))
overview
"""
        ),
        md("## 4. Tạo augmented features"),
        code(
            """
def build_augmented_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["Light_log"] = np.log1p(df["Light"].clip(lower=0))
    df["Temp_x_Humi"] = df["Temperature"] * df["Humidity"]
    df["Temp_x_Light"] = df["Temperature"] * df["Light_log"]
    df["Humi_x_Light"] = df["Humidity"] * df["Light_log"]
    df["SP_Center"] = 0.5 * (df["Soil_Low_SP"] + df["Soil_High_SP"])
    df["SP_Width"] = df["Soil_High_SP"] - df["Soil_Low_SP"]
    df["Month_sin"] = np.sin(2.0 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2.0 * np.pi * df["Month"] / 12.0)

    season_map = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
    season_num = df["Season"].map(season_map).fillna(0).astype(float)
    df["Season_sin"] = np.sin(2.0 * np.pi * season_num / 4.0)
    df["Season_cos"] = np.cos(2.0 * np.pi * season_num / 4.0)
    return df


df_full_aug = build_augmented_df(df_full)
df_train_aug, df_val_aug, df_test_aug = split_time_series(df_full_aug, SPLIT_CONFIG)

display(df_train_aug[list(AUGMENTED_INPUT_COLS)].head())
display(df_train_aug[list(AUGMENTED_INPUT_COLS)].describe().T.round(4))
"""
        ),
        md(
            """
## 5. Tính z-score từ train

Mean/std chỉ lấy từ train để tránh leakage sang validation/test.
"""
        ),
        code(
            """
scale_stats = {}
for col in SCALE_COLS:
    mean = float(df_train_aug[col].astype(float).mean())
    std = float(df_train_aug[col].astype(float).std(ddof=0))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    scale_stats[col] = {"mean": mean, "std": std}

pd.DataFrame(scale_stats).T
"""
        ),
        code(
            """
def apply_zscore(df_in: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    df_out = df_in.copy()
    for col, st in stats.items():
        df_out[col] = (df_out[col].astype(float) - st["mean"]) / st["std"]
    return df_out


def inverse_y(values: np.ndarray) -> np.ndarray:
    st = scale_stats["Soil_Moisture"]
    return np.asarray(values, dtype=float) * st["std"] + st["mean"]


df_train_z = apply_zscore(df_train_aug, scale_stats)
df_val_z = apply_zscore(df_val_aug, scale_stats)
df_test_z = apply_zscore(df_test_aug, scale_stats)

display(df_train_z[["Soil_Moisture", *AUGMENTED_INPUT_COLS]].describe().T.round(4))
"""
        ),
        md("## 6. Tạo regression matrix trên dữ liệu đã normalize"),
        code(
            """
X_train_z, y_train_z = build_regression_matrix(df_train_z, V5_CONFIG)
theta_ols_z, cov_ols_z, sigma2_ols_z = estimate_ols(X_train_z, y_train_z)
true_theta = build_true_theta(true_params, V5_CONFIG)

matrix_info = pd.Series({
    "x_train_shape": X_train_z.shape,
    "y_train_shape": y_train_z.shape,
    "rank_x_train": int(np.linalg.matrix_rank(X_train_z)),
    "condition_number_xtx": float(np.linalg.cond(X_train_z.T @ X_train_z)),
    "sigma2_ols_scaled": float(sigma2_ols_z),
})

matrix_info
"""
        ),
        md("## 7. Ridge alpha search"),
        code(
            """
def estimate_ridge(x_mat: np.ndarray, y_vec: np.ndarray, alpha: float, penalize_intercept: bool = False) -> np.ndarray:
    n_params = x_mat.shape[1]
    penalty = np.eye(n_params, dtype=float)
    if not penalize_intercept and V5_CONFIG.include_intercept:
        penalty[-1, -1] = 0.0
    return np.linalg.pinv(x_mat.T @ x_mat + float(alpha) * penalty) @ x_mat.T @ y_vec


def ridge_sigma2(x_mat: np.ndarray, y_vec: np.ndarray, theta: np.ndarray) -> float:
    resid = y_vec - x_mat @ theta
    return float(np.dot(resid, resid) / max(1, len(y_vec) - len(theta)))


def real_unit_metrics_for_theta(theta: np.ndarray, split_name: str, df_z: pd.DataFrame, mode: str) -> dict[str, float]:
    ev = evaluate_slice(split_name, df_z, theta, V5_CONFIG, true_theta=true_theta, n_step=12)
    arrays = ev["arrays"]
    if mode == "1step":
        y_true = inverse_y(arrays["y_true_1step"])
        y_pred = inverse_y(arrays["y_pred_1step"])
    elif mode == "12step":
        y_true = inverse_y(arrays["y_true_sim"])
        y_pred = inverse_y(arrays["y_pred_n_step"])
    elif mode == "sim":
        y_true = inverse_y(arrays["y_true_sim"])
        y_pred = inverse_y(arrays["y_pred_sim"])
    else:
        raise ValueError(mode)
    return compute_metrics(y_true, y_pred, len(theta))


search_rows = []
theta_by_alpha = {}
for alpha in RIDGE_ALPHAS:
    theta = theta_ols_z if alpha == 0.0 else estimate_ridge(X_train_z, y_train_z, alpha)
    theta_by_alpha[float(alpha)] = theta
    train_sim = real_unit_metrics_for_theta(theta, "Train", df_train_z, "sim")
    val_1 = real_unit_metrics_for_theta(theta, "Validation", df_val_z, "1step")
    val_12 = real_unit_metrics_for_theta(theta, "Validation", df_val_z, "12step")
    val_sim = real_unit_metrics_for_theta(theta, "Validation", df_val_z, "sim")
    test_sim = real_unit_metrics_for_theta(theta, "Test", df_test_z, "sim")
    search_rows.append({
        "alpha": float(alpha),
        "train_FIT_sim": train_sim["FIT"],
        "val_FIT_1step": val_1["FIT"],
        "val_FIT_12step": val_12["FIT"],
        "val_FIT_sim": val_sim["FIT"],
        "val_RMSE_sim": val_sim["RMSE"],
        "test_FIT_sim": test_sim["FIT"],
        "test_RMSE_sim": test_sim["RMSE"],
        "theta_l2": float(np.linalg.norm(theta)),
    })

alpha_search_df = pd.DataFrame(search_rows).sort_values(
    ["val_FIT_sim", "val_FIT_12step", "val_FIT_1step"],
    ascending=[False, False, False],
).reset_index(drop=True)

best_alpha = float(alpha_search_df.iloc[0]["alpha"])
theta_hat_z = theta_by_alpha[best_alpha]
sigma2_hat_z = ridge_sigma2(X_train_z, y_train_z, theta_hat_z)

alpha_search_df.round(6)
"""
        ),
        md("## 8. Bảng tham số scaled của alpha tốt nhất"),
        code(
            """
best_summary_df = pd.DataFrame({
    "name": V5_CONFIG.param_names,
    "theta_scaled": theta_hat_z,
    "abs_theta_scaled": np.abs(theta_hat_z),
})
print("Best alpha:", best_alpha)
best_summary_df
"""
        ),
        md(
            """
## 9. Đánh giá Train / Validation / Test

FIT không đổi theo phép đổi đơn vị tuyến tính, nhưng RMSE được báo lại theo đơn vị `Soil_Moisture` thật.
"""
        ),
        code(
            """
train_eval_z = evaluate_slice("Train", df_train_z, theta_hat_z, V5_CONFIG, true_theta=true_theta, n_step=12)
val_eval_z = evaluate_slice("Validation", df_val_z, theta_hat_z, V5_CONFIG, true_theta=true_theta, n_step=12)
test_eval_z = evaluate_slice("Test", df_test_z, theta_hat_z, V5_CONFIG, true_theta=true_theta, n_step=12)


def real_unit_metrics(ev: dict, mode: str) -> dict[str, float]:
    arrays = ev["arrays"]
    if mode == "1step":
        y_true = inverse_y(arrays["y_true_1step"])
        y_pred = inverse_y(arrays["y_pred_1step"])
    elif mode == "12step":
        y_true = inverse_y(arrays["y_true_sim"])
        y_pred = inverse_y(arrays["y_pred_n_step"])
    elif mode == "sim":
        y_true = inverse_y(arrays["y_true_sim"])
        y_pred = inverse_y(arrays["y_pred_sim"])
    else:
        raise ValueError(mode)
    return compute_metrics(y_true, y_pred, len(theta_hat_z))


def metric_row(split_name: str, ev: dict) -> dict[str, float | str]:
    m1 = real_unit_metrics(ev, "1step")
    m12 = real_unit_metrics(ev, "12step")
    msim = real_unit_metrics(ev, "sim")
    return {
        "split": split_name,
        "FIT_1step": m1["FIT"],
        "RMSE_1step": m1["RMSE"],
        "FIT_12step": m12["FIT"],
        "RMSE_12step": m12["RMSE"],
        "FIT_sim": msim["FIT"],
        "RMSE_sim": msim["RMSE"],
        "Bias_sim": msim["Bias"],
    }


metrics_df = pd.DataFrame([
    metric_row("Train", train_eval_z),
    metric_row("Validation", val_eval_z),
    metric_row("Test", test_eval_z),
]).round(4)

metrics_df
"""
        ),
        md("## 10. So sánh với V1/V2/V3/V4 artifact"),
        code(
            """
v1_path_candidates = [
    VERSION_DIR / "arx_baseline_v1.json",
    PROJECT_ROOT / "arx_baseline_v1.json",
]
v2_path_candidates = [
    VERSION_DIR / "arx_baseline_v2.json",
    PROJECT_ROOT / "arx_baseline_v2.json",
]
v3_path_candidates = [
    VERSION_DIR / "arx_baseline_v3.json",
    PROJECT_ROOT / "arx_baseline_v3.json",
]
v4_path_candidates = [
    VERSION_DIR / "arx_baseline_v4.json",
    PROJECT_ROOT / "arx_baseline_v4.json",
]
v1_artifact = None
for path in v1_path_candidates:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            v1_artifact = json.load(f)
        break
v2_artifact = None
for path in v2_path_candidates:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            v2_artifact = json.load(f)
        break
v3_artifact = None
for path in v3_path_candidates:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            v3_artifact = json.load(f)
        break
v4_artifact = None
for path in v4_path_candidates:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            v4_artifact = json.load(f)
        break

if v1_artifact is None:
    comparison_df = pd.DataFrame()
else:
    rows = [
        {
            "version": "V1 raw OLS",
            "val_FIT_sim": v1_artifact["metrics"]["validation"]["fit_sim"],
            "test_FIT_sim": v1_artifact["metrics"]["test"]["fit_sim"],
        }
    ]
    if v2_artifact is not None:
        rows.append(
            {
                "version": "V2 z-score OLS",
                "val_FIT_sim": v2_artifact["metrics"]["validation"]["fit_sim"],
                "test_FIT_sim": v2_artifact["metrics"]["test"]["fit_sim"],
            }
        )
    if v3_artifact is not None:
        rows.append(
            {
                "version": "V3 z-score + intercept OLS",
                "val_FIT_sim": v3_artifact["metrics"]["validation"]["fit_sim"],
                "test_FIT_sim": v3_artifact["metrics"]["test"]["fit_sim"],
            }
        )
    if v4_artifact is not None:
        rows.append(
            {
                "version": f"V4 Ridge alpha={v4_artifact['regularization']['best_alpha']:g}",
                "val_FIT_sim": v4_artifact["metrics"]["validation"]["fit_sim"],
                "test_FIT_sim": v4_artifact["metrics"]["test"]["fit_sim"],
            }
        )
    rows.append(
        {
            "version": f"V5 augmented Ridge alpha={best_alpha:g}",
            "val_FIT_sim": float(metrics_df.loc[metrics_df["split"] == "Validation", "FIT_sim"].iloc[0]),
            "test_FIT_sim": float(metrics_df.loc[metrics_df["split"] == "Test", "FIT_sim"].iloc[0]),
        }
    )
    comparison_df = pd.DataFrame(rows)
    comparison_df["test_gain_vs_v1"] = comparison_df["test_FIT_sim"] - comparison_df.loc[0, "test_FIT_sim"]
    comparison_df["test_gain_vs_previous"] = comparison_df["test_FIT_sim"].diff()

comparison_df.round(4)
"""
        ),
        md("## 11. Vẽ free-run simulation"),
        code(
            """
def prediction_frame_real(ev: dict, split_name: str) -> pd.DataFrame:
    arrays = ev["arrays"]
    n = len(arrays["y_true_sim"])
    return pd.DataFrame({
        "split": split_name,
        "idx": np.arange(n),
        "y_true": inverse_y(arrays["y_true_sim"]),
        "y_pred_sim": inverse_y(arrays["y_pred_sim"]),
        "y_pred_12step": inverse_y(arrays["y_pred_n_step"]),
    })


pred_val = prediction_frame_real(val_eval_z, "Validation")
pred_test = prediction_frame_real(test_eval_z, "Test")

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
for ax, pred_df, title in [
    (axes[0], pred_val.iloc[: 7 * 24 * 12], "Validation - 7 ngày đầu"),
    (axes[1], pred_test.iloc[: 7 * 24 * 12], "Test - 7 ngày đầu"),
]:
    ax.plot(pred_df["idx"], pred_df["y_true"], label="Thực tế", linewidth=1.2)
    ax.plot(pred_df["idx"], pred_df["y_pred_sim"], label="Free-run V5", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("Soil_Moisture")
    ax.grid(True, alpha=0.25)
    ax.legend()
axes[-1].set_xlabel("Chỉ số mẫu cục bộ")
fig.tight_layout()
plt.show()
"""
        ),
        md("## 12. Stability roots"),
        code(
            """
ar_roots = compute_ar_roots(theta_hat_z, V5_CONFIG)
pd.DataFrame(ar_roots)
"""
        ),
        md("## 13. Lưu artifact V5"),
        code(
            """
def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


artifact = {
    "model_type": "ARX",
    "version": "v5_augmented_zscore_intercept_ridge",
    "notes": "ARX(2,2,1), 16 augmented inputs, Ridge alpha search, z-score normalization from train stats, include_intercept=True, no clip.",
    "data_source": data_source,
    "data_config": DATA_CONFIG.__dict__,
    "split_config": SPLIT_CONFIG.__dict__,
    "model_config": V5_CONFIG.__dict__,
    "feature_engineering": {
        "enabled": True,
        "raw_input_cols": list(BASELINE_INPUT_COLS),
        "augmented_input_cols": list(AUGMENTED_INPUT_COLS),
        "new_features": [
            "Light_log",
            "Temp_x_Humi",
            "Temp_x_Light",
            "Humi_x_Light",
            "SP_Center",
            "SP_Width",
            "Month_sin",
            "Month_cos",
            "Season_sin",
            "Season_cos",
        ],
    },
    "regularization": {
        "method": "ridge",
        "alpha_grid": RIDGE_ALPHAS,
        "best_alpha": best_alpha,
        "selected_by": "max validation FIT_sim",
        "penalize_intercept": False,
    },
    "normalization": {
        "method": "zscore",
        "fit_on": "train",
        "scaled_columns": list(SCALE_COLS),
        "stats": scale_stats,
        "target_inverse_transform": "Soil_Moisture = Soil_Moisture_scaled * train_std + train_mean",
    },
    "param_names": V5_CONFIG.param_names,
    "theta_hat_scaled": theta_hat_z.tolist(),
    "sigma2_scaled": sigma2_hat_z,
    "matrix_info": matrix_info.to_dict(),
    "alpha_search": alpha_search_df.to_dict(orient="records"),
    "ar_roots": ar_roots,
    "metrics": {
        "train": {
            "fit_1": real_unit_metrics(train_eval_z, "1step")["FIT"],
            "fit_12": real_unit_metrics(train_eval_z, "12step")["FIT"],
            "fit_sim": real_unit_metrics(train_eval_z, "sim")["FIT"],
            "rmse_sim": real_unit_metrics(train_eval_z, "sim")["RMSE"],
        },
        "validation": {
            "fit_1": real_unit_metrics(val_eval_z, "1step")["FIT"],
            "fit_12": real_unit_metrics(val_eval_z, "12step")["FIT"],
            "fit_sim": real_unit_metrics(val_eval_z, "sim")["FIT"],
            "rmse_sim": real_unit_metrics(val_eval_z, "sim")["RMSE"],
        },
        "test": {
            "fit_1": real_unit_metrics(test_eval_z, "1step")["FIT"],
            "fit_12": real_unit_metrics(test_eval_z, "12step")["FIT"],
            "fit_sim": real_unit_metrics(test_eval_z, "sim")["FIT"],
            "rmse_sim": real_unit_metrics(test_eval_z, "sim")["RMSE"],
        },
    },
    "comparison_vs_v1_v2_v3_v4": comparison_df.to_dict(orient="records") if not comparison_df.empty else [],
}

VERSION_DIR.mkdir(exist_ok=True)
output_path = VERSION_DIR / "arx_baseline_v5.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(json_ready(artifact), f, indent=2)
    f.write("\\n")

output_path
"""
        ),
        md(
            """
## 14. Kết luận V5

V5 chỉ trả lời một câu hỏi: thêm feature engineering vào `ARX(2,2,1)` có cải thiện free-run không khi chưa dùng clip.

Nếu V5 chưa cải thiện `FIT_sim`, bước tiếp theo hợp lý là thêm clip free-run để kiểm soát drift.
"""
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH.resolve()}")


if __name__ == "__main__":
    main()
