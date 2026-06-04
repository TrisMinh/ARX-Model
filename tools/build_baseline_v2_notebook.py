from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path("ARX_Model_Version") / "ARX_Baseline_V2.ipynb"


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
# ARX Baseline V2

V2 giữ nguyên cấu trúc baseline để kiểm tra riêng tác động của normalization:

- `ARX(2,2,1)`
- 6 input gốc
- OLS
- không feature engineering
- không Ridge/Lasso
- không clip free-run
- z-score normalization bằng thống kê train
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
        md("## 2. Cấu hình V2"),
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

V2_CONFIG = ModelConfig(
    na=2,
    nb=2,
    nk=1,
    include_intercept=False,
    input_cols=BASELINE_INPUT_COLS,
    output_col="Soil_Moisture",
    simulation_clip=None,
)

SCALE_COLS = ("Soil_Moisture", *BASELINE_INPUT_COLS)
V2_CONFIG
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
        md(
            """
## 4. Tính z-score từ train

Mean/std chỉ lấy từ train để tránh leakage sang validation/test.
"""
        ),
        code(
            """
scale_stats = {}
for col in SCALE_COLS:
    mean = float(df_train[col].astype(float).mean())
    std = float(df_train[col].astype(float).std(ddof=0))
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


df_train_z = apply_zscore(df_train, scale_stats)
df_val_z = apply_zscore(df_val, scale_stats)
df_test_z = apply_zscore(df_test, scale_stats)

display(df_train_z[["Soil_Moisture", *BASELINE_INPUT_COLS]].describe().T.round(4))
"""
        ),
        md("## 5. Fit OLS trên dữ liệu đã normalize"),
        code(
            """
X_train_z, y_train_z = build_regression_matrix(df_train_z, V2_CONFIG)
theta_hat_z, cov_hat_z, sigma2_hat_z = estimate_ols(X_train_z, y_train_z)
true_theta = build_true_theta(true_params, V2_CONFIG)

matrix_info = pd.Series({
    "x_train_shape": X_train_z.shape,
    "y_train_shape": y_train_z.shape,
    "rank_x_train": int(np.linalg.matrix_rank(X_train_z)),
    "condition_number_xtx": float(np.linalg.cond(X_train_z.T @ X_train_z)),
    "sigma2_hat_scaled": float(sigma2_hat_z),
})

matrix_info
"""
        ),
        md("## 6. Bảng tham số scaled"),
        code(
            """
param_df = pd.DataFrame(summarize_parameters(theta_hat_z, cov_hat_z, V2_CONFIG, true_params))
param_df
"""
        ),
        md(
            """
## 7. Đánh giá Train / Validation / Test

FIT không đổi theo phép đổi đơn vị tuyến tính, nhưng RMSE được báo lại theo đơn vị `Soil_Moisture` thật.
"""
        ),
        code(
            """
train_eval_z = evaluate_slice("Train", df_train_z, theta_hat_z, V2_CONFIG, true_theta=true_theta, n_step=12)
val_eval_z = evaluate_slice("Validation", df_val_z, theta_hat_z, V2_CONFIG, true_theta=true_theta, n_step=12)
test_eval_z = evaluate_slice("Test", df_test_z, theta_hat_z, V2_CONFIG, true_theta=true_theta, n_step=12)


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
        md("## 8. So sánh với V1 artifact"),
        code(
            """
v1_path_candidates = [
    VERSION_DIR / "arx_baseline_v1.json",
    PROJECT_ROOT / "arx_baseline_v1.json",
]
v1_artifact = None
for path in v1_path_candidates:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            v1_artifact = json.load(f)
        break

if v1_artifact is None:
    comparison_df = pd.DataFrame()
else:
    comparison_df = pd.DataFrame([
        {
            "version": "V1 raw OLS",
            "val_FIT_sim": v1_artifact["metrics"]["validation"]["fit_sim"],
            "test_FIT_sim": v1_artifact["metrics"]["test"]["fit_sim"],
        },
        {
            "version": "V2 z-score OLS",
            "val_FIT_sim": float(metrics_df.loc[metrics_df["split"] == "Validation", "FIT_sim"].iloc[0]),
            "test_FIT_sim": float(metrics_df.loc[metrics_df["split"] == "Test", "FIT_sim"].iloc[0]),
        },
    ])
    comparison_df["test_gain_vs_v1"] = comparison_df["test_FIT_sim"] - comparison_df.loc[0, "test_FIT_sim"]

comparison_df.round(4)
"""
        ),
        md("## 9. Vẽ free-run simulation"),
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
    ax.plot(pred_df["idx"], pred_df["y_pred_sim"], label="Free-run V2", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("Soil_Moisture")
    ax.grid(True, alpha=0.25)
    ax.legend()
axes[-1].set_xlabel("Chỉ số mẫu cục bộ")
fig.tight_layout()
plt.show()
"""
        ),
        md("## 10. Stability roots"),
        code(
            """
ar_roots = compute_ar_roots(theta_hat_z, V2_CONFIG)
pd.DataFrame(ar_roots)
"""
        ),
        md("## 11. Lưu artifact V2"),
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
    "version": "v2_zscore_ols",
    "notes": "ARX(2,2,1), 6 raw inputs, OLS, z-score normalization from train stats, no feature engineering, no clip, no Ridge/Lasso.",
    "data_source": data_source,
    "data_config": DATA_CONFIG.__dict__,
    "split_config": SPLIT_CONFIG.__dict__,
    "model_config": V2_CONFIG.__dict__,
    "normalization": {
        "method": "zscore",
        "fit_on": "train",
        "scaled_columns": list(SCALE_COLS),
        "stats": scale_stats,
        "target_inverse_transform": "Soil_Moisture = Soil_Moisture_scaled * train_std + train_mean",
    },
    "param_names": V2_CONFIG.param_names,
    "theta_hat_scaled": theta_hat_z.tolist(),
    "sigma2_scaled": sigma2_hat_z,
    "matrix_info": matrix_info.to_dict(),
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
    "comparison_vs_v1": comparison_df.to_dict(orient="records") if not comparison_df.empty else [],
}

VERSION_DIR.mkdir(exist_ok=True)
output_path = VERSION_DIR / "arx_baseline_v2.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(json_ready(artifact), f, indent=2)
    f.write("\\n")

output_path
"""
        ),
        md(
            """
## 12. Kết luận V2

V2 chỉ trả lời một câu hỏi: normalize bằng train z-score có cải thiện baseline OLS hay không.

Nếu V2 chưa cải thiện `FIT_sim`, đó là kết quả hợp lệ: normalization chủ yếu giúp điều kiện số và là nền cho Ridge/Lasso, không đảm bảo tự tăng chất lượng free-run khi vẫn dùng OLS.
"""
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH.resolve()}")


if __name__ == "__main__":
    main()
