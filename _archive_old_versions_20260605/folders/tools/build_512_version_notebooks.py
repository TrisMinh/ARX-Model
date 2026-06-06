from __future__ import annotations

from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    import json
    import types

    class _NotebookNode(dict):
        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value):
            self[name] = value

    class _V4:
        @staticmethod
        def new_markdown_cell(source: str):
            return {"cell_type": "markdown", "metadata": {}, "source": source}

        @staticmethod
        def new_code_cell(source: str):
            return {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }

        @staticmethod
        def new_notebook():
            return _NotebookNode({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})

    def _write(nb, out):
        Path(out).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    nbf = types.SimpleNamespace(v4=_V4, write=_write, NotebookNode=_NotebookNode)


VERSION_DIR = Path("ARX_Model_Version 512")


VERSIONS = [
    {
        "n": 1,
        "version": "v1_512_raw_ols",
        "title": "ARX 512 Baseline V1",
        "summary": "ARX(5,1,2), 6 raw inputs, OLS, no intercept, no normalization, no clip.",
        "use_augmented": False,
        "use_zscore": False,
        "include_intercept": False,
        "use_ridge": False,
        "use_clip": False,
    },
    {
        "n": 2,
        "version": "v2_512_raw_zscore_ols",
        "title": "ARX 512 Baseline V2",
        "summary": "ARX(5,1,2), 6 raw inputs, OLS, z-score normalization, no intercept, no clip.",
        "use_augmented": False,
        "use_zscore": True,
        "include_intercept": False,
        "use_ridge": False,
        "use_clip": False,
    },
    {
        "n": 3,
        "version": "v3_512_raw_zscore_intercept_ols",
        "title": "ARX 512 Baseline V3",
        "summary": "ARX(5,1,2), 6 raw inputs, OLS, z-score normalization, intercept, no clip.",
        "use_augmented": False,
        "use_zscore": True,
        "include_intercept": True,
        "use_ridge": False,
        "use_clip": False,
    },
    {
        "n": 4,
        "version": "v4_512_raw_zscore_intercept_ridge",
        "title": "ARX 512 Baseline V4",
        "summary": "ARX(5,1,2), 6 raw inputs, Ridge alpha search, z-score normalization, intercept, no clip.",
        "use_augmented": False,
        "use_zscore": True,
        "include_intercept": True,
        "use_ridge": True,
        "use_clip": False,
    },
    {
        "n": 5,
        "version": "v5_512_augmented_zscore_intercept_ridge",
        "title": "ARX 512 Baseline V5",
        "summary": "ARX(5,1,2), 16 augmented inputs, Ridge alpha search, z-score normalization, intercept, no clip.",
        "use_augmented": True,
        "use_zscore": True,
        "include_intercept": True,
        "use_ridge": True,
        "use_clip": False,
    },
    {
        "n": 6,
        "version": "v6_512_augmented_zscore_intercept_ridge_clip",
        "title": "ARX 512 Baseline V6",
        "summary": "ARX(5,1,2), 16 augmented inputs, Ridge alpha search, z-score normalization, intercept, free-run clip.",
        "use_augmented": True,
        "use_zscore": True,
        "include_intercept": True,
        "use_ridge": True,
        "use_clip": True,
    },
]


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def make_notebook(config: dict) -> nbf.NotebookNode:
    n = config["n"]
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }

    nb.cells = [
        md(
            f"""
# {config["title"]}

{config["summary"]}

Chuỗi này tương ứng với folder `ARX_Model_Version 221`, nhưng cố định order là `ARX(5,1,2)`.
"""
        ),
        md("## 1. Import và cấu hình"),
        code(
            f"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WORK_DIR = Path.cwd()
PROJECT_ROOT = WORK_DIR.parent if WORK_DIR.name.startswith("ARX_Model_Version") else WORK_DIR
VERSION_DIR = PROJECT_ROOT / "ARX_Model_Version 512"

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
    evaluate_slice,
    compute_ar_roots,
    compute_metrics,
)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

VERSION_NUMBER = {n}
VERSION_NAME = "{config["version"]}"
USE_AUGMENTED = {config["use_augmented"]}
USE_ZSCORE = {config["use_zscore"]}
USE_RIDGE = {config["use_ridge"]}
USE_CLIP = {config["use_clip"]}
INCLUDE_INTERCEPT = {config["include_intercept"]}
RIDGE_ALPHAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CLIP_QUANTILES = (0.01, 0.99)
"""
        ),
        md("## 2. Load data"),
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

df_full, true_params, data_source = load_or_generate_data(DATA_CONFIG)
df_train_raw, df_val_raw, df_test_raw = split_time_series(df_full, SPLIT_CONFIG)

overview = pd.DataFrame([
    summarize_dataset_behavior(df_full, "Full"),
    summarize_dataset_behavior(df_train_raw, "Train"),
    summarize_dataset_behavior(df_val_raw, "Validation"),
    summarize_dataset_behavior(df_test_raw, "Test"),
])

print("Nguồn dữ liệu:", data_source)
print("Số dòng:", len(df_full), "| Train:", len(df_train_raw), "| Validation:", len(df_val_raw), "| Test:", len(df_test_raw))
overview
"""
        ),
        md("## 3. Feature set"),
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


df_model = build_augmented_df(df_full) if USE_AUGMENTED else df_full.copy()
df_train, df_val, df_test = split_time_series(df_model, SPLIT_CONFIG)
INPUT_COLS = AUGMENTED_INPUT_COLS if USE_AUGMENTED else BASELINE_INPUT_COLS
SCALE_COLS = ("Soil_Moisture", *INPUT_COLS)

display(pd.Series({
    "use_augmented": USE_AUGMENTED,
    "n_inputs": len(INPUT_COLS),
    "inputs": list(INPUT_COLS),
}))
"""
        ),
        md("## 4. Normalization và clip"),
        code(
            """
scale_stats = {}
for col in SCALE_COLS:
    mean = float(df_train[col].astype(float).mean())
    std = float(df_train[col].astype(float).std(ddof=0))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    scale_stats[col] = {"mean": mean, "std": std}


def apply_zscore(df_in: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    df_out = df_in.copy()
    if not USE_ZSCORE:
        return df_out
    for col, st in stats.items():
        df_out[col] = (df_out[col].astype(float) - st["mean"]) / st["std"]
    return df_out


def inverse_y(values: np.ndarray) -> np.ndarray:
    if not USE_ZSCORE:
        return np.asarray(values, dtype=float)
    st = scale_stats["Soil_Moisture"]
    return np.asarray(values, dtype=float) * st["std"] + st["mean"]


clip_low_real = float(df_train["Soil_Moisture"].quantile(CLIP_QUANTILES[0]))
clip_high_real = float(df_train["Soil_Moisture"].quantile(CLIP_QUANTILES[1]))
if USE_ZSCORE:
    y_scale = scale_stats["Soil_Moisture"]
    clip_bounds_model = (
        float((clip_low_real - y_scale["mean"]) / y_scale["std"]),
        float((clip_high_real - y_scale["mean"]) / y_scale["std"]),
    )
else:
    clip_bounds_model = (clip_low_real, clip_high_real)

simulation_clip = clip_bounds_model if USE_CLIP else None

df_train_m = apply_zscore(df_train, scale_stats)
df_val_m = apply_zscore(df_val, scale_stats)
df_test_m = apply_zscore(df_test, scale_stats)

MODEL_CONFIG = ModelConfig(
    na=5,
    nb=1,
    nk=2,
    include_intercept=INCLUDE_INTERCEPT,
    input_cols=INPUT_COLS,
    output_col="Soil_Moisture",
    simulation_clip=simulation_clip,
)

display(pd.Series({
    "use_zscore": USE_ZSCORE,
    "use_clip": USE_CLIP,
    "clip_low_real": clip_low_real,
    "clip_high_real": clip_high_real,
    "clip_low_model": None if simulation_clip is None else simulation_clip[0],
    "clip_high_model": None if simulation_clip is None else simulation_clip[1],
}))
MODEL_CONFIG
"""
        ),
        md("## 5. Fit model"),
        code(
            """
X_train, y_train = build_regression_matrix(df_train_m, MODEL_CONFIG)
theta_ols, cov_ols, sigma2_ols = estimate_ols(X_train, y_train)


def estimate_ridge(x_mat: np.ndarray, y_vec: np.ndarray, alpha: float, penalize_intercept: bool = False) -> np.ndarray:
    n_params = x_mat.shape[1]
    penalty = np.eye(n_params, dtype=float)
    if not penalize_intercept and MODEL_CONFIG.include_intercept:
        penalty[-1, -1] = 0.0
    return np.linalg.pinv(x_mat.T @ x_mat + float(alpha) * penalty) @ x_mat.T @ y_vec


def sigma2_for_theta(x_mat: np.ndarray, y_vec: np.ndarray, theta: np.ndarray) -> float:
    resid = y_vec - x_mat @ theta
    return float(np.dot(resid, resid) / max(1, len(y_vec) - len(theta)))


def evaluate_real(split_name: str, df_split: pd.DataFrame, theta: np.ndarray) -> dict:
    ev = evaluate_slice(split_name, df_split, theta, MODEL_CONFIG, true_theta=None, n_step=12)
    arrays = ev["arrays"]
    return {
        "eval": ev,
        "metrics_1step_real": compute_metrics(inverse_y(arrays["y_true_1step"]), inverse_y(arrays["y_pred_1step"]), len(theta)),
        "metrics_12step_real": compute_metrics(inverse_y(arrays["y_true_sim"]), inverse_y(arrays["y_pred_n_step"]), len(theta)),
        "metrics_sim_real": compute_metrics(inverse_y(arrays["y_true_sim"]), inverse_y(arrays["y_pred_sim"]), len(theta)),
    }


theta_by_alpha = {}
search_rows = []
alphas = RIDGE_ALPHAS if USE_RIDGE else [0.0]
for alpha in alphas:
    theta = theta_ols if alpha == 0.0 else estimate_ridge(X_train, y_train, alpha)
    theta_by_alpha[float(alpha)] = theta
    train_result = evaluate_real("Train", df_train_m, theta)
    val_result = evaluate_real("Validation", df_val_m, theta)
    test_result = evaluate_real("Test", df_test_m, theta)
    search_rows.append({
        "alpha": float(alpha),
        "train_FIT_sim": train_result["metrics_sim_real"]["FIT"],
        "val_FIT_1step": val_result["metrics_1step_real"]["FIT"],
        "val_FIT_12step": val_result["metrics_12step_real"]["FIT"],
        "val_FIT_sim": val_result["metrics_sim_real"]["FIT"],
        "val_RMSE_sim": val_result["metrics_sim_real"]["RMSE"],
        "test_FIT_sim": test_result["metrics_sim_real"]["FIT"],
        "test_RMSE_sim": test_result["metrics_sim_real"]["RMSE"],
        "theta_l2": float(np.linalg.norm(theta)),
    })

alpha_search_df = pd.DataFrame(search_rows).sort_values(
    ["val_FIT_sim", "val_FIT_12step", "val_FIT_1step"],
    ascending=[False, False, False],
).reset_index(drop=True)

best_alpha = float(alpha_search_df.iloc[0]["alpha"])
theta_hat = theta_by_alpha[best_alpha]
sigma2_hat = sigma2_for_theta(X_train, y_train, theta_hat)

matrix_info = pd.Series({
    "x_train_shape": X_train.shape,
    "y_train_shape": y_train.shape,
    "rank_x_train": int(np.linalg.matrix_rank(X_train)),
    "condition_number_xtx": float(np.linalg.cond(X_train.T @ X_train)),
    "sigma2_ols": float(sigma2_ols),
    "sigma2_selected": float(sigma2_hat),
})

display(matrix_info)
alpha_search_df.round(6)
"""
        ),
        md("## 6. Metrics"),
        code(
            """
train_result = evaluate_real("Train", df_train_m, theta_hat)
val_result = evaluate_real("Validation", df_val_m, theta_hat)
test_result = evaluate_real("Test", df_test_m, theta_hat)

train_eval = train_result["eval"]
val_eval = val_result["eval"]
test_eval = test_result["eval"]


def metric_row(split_name: str, result: dict) -> dict:
    m1 = result["metrics_1step_real"]
    m12 = result["metrics_12step_real"]
    msim = result["metrics_sim_real"]
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
    metric_row("Train", train_result),
    metric_row("Validation", val_result),
    metric_row("Test", test_result),
]).round(4)

best_summary_df = pd.DataFrame({
    "name": MODEL_CONFIG.param_names,
    "theta": theta_hat,
    "abs_theta": np.abs(theta_hat),
}).sort_values("abs_theta", ascending=False)

print("Best alpha:", best_alpha)
display(metrics_df)
best_summary_df
"""
        ),
        md("## 7. So sánh với version trước trong folder 512"),
        code(
            """
comparison_rows = []
for prev_n in range(1, VERSION_NUMBER):
    path = VERSION_DIR / f"arx_512_v{prev_n}.json"
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        prev = json.load(f)
    comparison_rows.append({
        "version": f"512 V{prev_n}",
        "val_FIT_sim": prev["metrics"]["validation"]["fit_sim"],
        "test_FIT_sim": prev["metrics"]["test"]["fit_sim"],
    })

comparison_rows.append({
    "version": f"512 V{VERSION_NUMBER}",
    "val_FIT_sim": float(metrics_df.loc[metrics_df["split"] == "Validation", "FIT_sim"].iloc[0]),
    "test_FIT_sim": float(metrics_df.loc[metrics_df["split"] == "Test", "FIT_sim"].iloc[0]),
})

comparison_df = pd.DataFrame(comparison_rows)
if not comparison_df.empty:
    comparison_df["test_gain_vs_v1"] = comparison_df["test_FIT_sim"] - comparison_df.loc[0, "test_FIT_sim"]
    comparison_df["test_gain_vs_previous"] = comparison_df["test_FIT_sim"].diff()

comparison_df.round(4)
"""
        ),
        md("## 8. Bộ biểu đồ báo cáo"),
        code(
            """
figure_catalog = pd.DataFrame([
    {"figure": "Fig. 1", "name": "Phân chia dữ liệu theo thời gian", "report_section": "Dữ liệu và thiết kế thí nghiệm"},
    {"figure": "Fig. 2", "name": "Phân bố biến chính theo split", "report_section": "Phân tích dữ liệu đầu vào"},
    {"figure": "Fig. 3", "name": "So sánh FIT/RMSE giữa Train-Val-Test", "report_section": "Đánh giá định lượng"},
    {"figure": "Fig. 4", "name": "So sánh các version ARX(5,1,2)", "report_section": "Ablation/cải tiến mô hình"},
    {"figure": "Fig. 5", "name": "Ridge alpha search", "report_section": "Lựa chọn siêu tham số"},
    {"figure": "Fig. 6", "name": "Độ lớn tham số ước lượng", "report_section": "Diễn giải mô hình"},
    {"figure": "Fig. 7", "name": "Actual vs predicted", "report_section": "Khả năng dự báo"},
    {"figure": "Fig. 8", "name": "Residual diagnostics", "report_section": "Kiểm định sai số"},
    {"figure": "Fig. 9", "name": "Impulse response", "report_section": "Động học hệ thống"},
])
figure_catalog
"""
        ),
        code(
            """
split_frames = [
    ("Train", df_train_raw),
    ("Validation", df_val_raw),
    ("Test", df_test_raw),
]

fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=False)
for ax, (split_name, split_df) in zip(axes, split_frames):
    plot_df = split_df.iloc[::12].copy()
    ax.plot(plot_df["Timestamp"], plot_df["Soil_Moisture"], label="Soil moisture", linewidth=1.2)
    if {"Soil_Low_SP", "Soil_High_SP"}.issubset(plot_df.columns):
        ax.plot(plot_df["Timestamp"], plot_df["Soil_Low_SP"], label="Low SP", linewidth=0.9, alpha=0.75)
        ax.plot(plot_df["Timestamp"], plot_df["Soil_High_SP"], label="High SP", linewidth=0.9, alpha=0.75)
    ax.set_title(f"Fig. 1 - {split_name}: soil moisture và setpoint")
    ax.set_ylabel("Soil_Moisture")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncols=3)
axes[-1].set_xlabel("Thời gian")
fig.tight_layout()
plt.show()
"""
        ),
        code(
            """
dist_cols = ["Soil_Moisture", "Temperature", "Humidity", "Light", "Drip", "Mist", "Fan"]
dist_df = pd.concat(
    [frame.assign(split=name) for name, frame in split_frames],
    ignore_index=True,
)

fig, axes = plt.subplots(2, 4, figsize=(17, 8))
axes = axes.ravel()
for ax, col in zip(axes, dist_cols):
    for split_name in ["Train", "Validation", "Test"]:
        values = dist_df.loc[dist_df["split"] == split_name, col].astype(float)
        ax.hist(values, bins=35, density=True, alpha=0.35, label=split_name)
    ax.set_title(f"Fig. 2 - {col}")
    ax.grid(True, alpha=0.2)
axes[len(dist_cols)].axis("off")
axes[0].legend()
fig.tight_layout()
plt.show()
"""
        ),
        code(
            """
metrics_plot = metrics_df.melt(
    id_vars="split",
    value_vars=["FIT_1step", "FIT_12step", "FIT_sim", "RMSE_1step", "RMSE_12step", "RMSE_sim"],
    var_name="metric",
    value_name="value",
)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fit_plot = metrics_plot[metrics_plot["metric"].str.startswith("FIT")]
rmse_plot = metrics_plot[metrics_plot["metric"].str.startswith("RMSE")]

for ax, plot_data, title, ylabel in [
    (axes[0], fit_plot, "Fig. 3a - FIT theo kiểu dự báo", "FIT (%)"),
    (axes[1], rmse_plot, "Fig. 3b - RMSE theo kiểu dự báo", "RMSE"),
]:
    pivot = plot_data.pivot(index="split", columns="metric", values="value")
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="")
fig.tight_layout()
plt.show()
"""
        ),
        code(
            """
if not comparison_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    comparison_df.plot(x="version", y="val_FIT_sim", kind="bar", ax=axes[0], legend=False, color="#4c78a8")
    comparison_df.plot(x="version", y="test_FIT_sim", kind="bar", ax=axes[1], legend=False, color="#f58518")
    axes[0].set_title("Fig. 4a - Validation FIT_sim qua các version")
    axes[1].set_title("Fig. 4b - Test FIT_sim qua các version")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("FIT_sim (%)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    plt.show()
else:
    print("Không có dữ liệu comparison_df để vẽ Fig. 4.")
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
alpha_plot = alpha_search_df.copy()
alpha_plot["alpha_label"] = alpha_plot["alpha"].map(lambda x: f"{x:g}")
axes[0].plot(alpha_plot["alpha_label"], alpha_plot["val_FIT_sim"], marker="o", label="Validation")
axes[0].plot(alpha_plot["alpha_label"], alpha_plot["test_FIT_sim"], marker="o", label="Test")
axes[0].axvline(alpha_plot.index[alpha_plot["alpha"].eq(best_alpha)][0], color="black", linestyle="--", alpha=0.7)
axes[0].set_title("Fig. 5a - FIT_sim theo alpha")
axes[0].set_xlabel("alpha")
axes[0].set_ylabel("FIT_sim (%)")
axes[0].grid(True, alpha=0.25)
axes[0].legend()

axes[1].plot(alpha_plot["alpha_label"], alpha_plot["theta_l2"], marker="o", color="#54a24b")
axes[1].set_title("Fig. 5b - Độ lớn vector tham số theo alpha")
axes[1].set_xlabel("alpha")
axes[1].set_ylabel("||theta||2")
axes[1].grid(True, alpha=0.25)
for ax in axes:
    ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
plt.show()
"""
        ),
        code(
            """
coef_df = best_summary_df.copy()
top_coef = coef_df.sort_values("abs_theta", ascending=False).head(25).sort_values("abs_theta")
colors = np.where(top_coef["theta"] >= 0, "#4c78a8", "#e45756")

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(top_coef["name"], top_coef["theta"], color=colors, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.9)
ax.set_title("Fig. 6 - Top 25 tham số có độ lớn lớn nhất")
ax.set_xlabel("Hệ số trong không gian model" + (" z-score" if USE_ZSCORE else " raw"))
ax.grid(True, axis="x", alpha=0.25)
fig.tight_layout()
plt.show()
"""
        ),
        md("## 9. Free-run và dự báo"),
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


pred_train = prediction_frame_real(train_eval, "Train")
pred_val = prediction_frame_real(val_eval, "Validation")
pred_test = prediction_frame_real(test_eval, "Test")

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
for ax, pred_df, title in [
    (axes[0], pred_val.iloc[: 7 * 24 * 12], "Validation - 7 ngày đầu"),
    (axes[1], pred_test.iloc[: 7 * 24 * 12], "Test - 7 ngày đầu"),
]:
    ax.plot(pred_df["idx"], pred_df["y_true"], label="Thực tế", linewidth=1.2)
    ax.plot(pred_df["idx"], pred_df["y_pred_sim"], label=f"Free-run 512 V{VERSION_NUMBER}", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("Soil_Moisture")
    ax.grid(True, alpha=0.25)
    ax.legend()
axes[-1].set_xlabel("Chỉ số mẫu cục bộ")
fig.tight_layout()
plt.show()
"""
        ),
        code(
            """
prediction_all = pd.concat([pred_train, pred_val, pred_test], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pred_df, title in [
    (axes[0], pred_val, "Fig. 7a - Validation actual vs free-run"),
    (axes[1], pred_test, "Fig. 7b - Test actual vs free-run"),
]:
    sample_df = pred_df.iloc[::6].copy()
    ax.scatter(sample_df["y_true"], sample_df["y_pred_sim"], s=8, alpha=0.35)
    lo = min(sample_df["y_true"].min(), sample_df["y_pred_sim"].min())
    hi = max(sample_df["y_true"].max(), sample_df["y_pred_sim"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Thực tế")
    ax.set_ylabel("Dự báo free-run")
    ax.grid(True, alpha=0.25)
fig.tight_layout()
plt.show()
"""
        ),
        code(
            """
residual_frames = []
for pred_df in [pred_train, pred_val, pred_test]:
    tmp = pred_df.copy()
    tmp["residual_sim"] = tmp["y_true"] - tmp["y_pred_sim"]
    tmp["residual_12step"] = tmp["y_true"] - tmp["y_pred_12step"]
    residual_frames.append(tmp)
residual_df = pd.concat(residual_frames, ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for split_name, color in [("Validation", "#4c78a8"), ("Test", "#f58518")]:
    tmp = residual_df[residual_df["split"] == split_name]
    axes[0, 0].plot(tmp["idx"], tmp["residual_sim"], linewidth=0.8, alpha=0.75, label=split_name)
    axes[0, 1].hist(tmp["residual_sim"], bins=45, alpha=0.45, density=True, label=split_name, color=color)
    axes[1, 0].scatter(tmp["y_pred_sim"].iloc[::6], tmp["residual_sim"].iloc[::6], s=8, alpha=0.35, label=split_name)

test_resid = residual_df[residual_df["split"] == "Test"]["residual_sim"].to_numpy(dtype=float)
max_lag = 48
acf = []
for lag in range(max_lag + 1):
    if lag == 0:
        acf.append(1.0)
    else:
        acf.append(float(np.corrcoef(test_resid[:-lag], test_resid[lag:])[0, 1]))

axes[0, 0].axhline(0, color="black", linewidth=0.9)
axes[0, 0].set_title("Fig. 8a - Residual free-run theo thời gian")
axes[0, 0].set_xlabel("Chỉ số mẫu cục bộ")
axes[0, 0].set_ylabel("Residual")
axes[0, 0].grid(True, alpha=0.25)
axes[0, 0].legend()

axes[0, 1].set_title("Fig. 8b - Phân bố residual free-run")
axes[0, 1].set_xlabel("Residual")
axes[0, 1].grid(True, alpha=0.25)
axes[0, 1].legend()

axes[1, 0].axhline(0, color="black", linewidth=0.9)
axes[1, 0].set_title("Fig. 8c - Residual theo giá trị dự báo")
axes[1, 0].set_xlabel("Dự báo free-run")
axes[1, 0].set_ylabel("Residual")
axes[1, 0].grid(True, alpha=0.25)
axes[1, 0].legend()

axes[1, 1].stem(range(max_lag + 1), acf)
axes[1, 1].set_title("Fig. 8d - ACF residual Test")
axes[1, 1].set_xlabel("Lag")
axes[1, 1].set_ylabel("ACF")
axes[1, 1].grid(True, alpha=0.25)
fig.tight_layout()
plt.show()
"""
        ),
        code(
            """
def impulse_response_df(theta: np.ndarray, config: ModelConfig, horizon: int = 48) -> pd.DataFrame:
    a = np.asarray(theta[: config.na], dtype=float)
    start = config.na
    rows = []
    for input_idx, input_name in enumerate(config.input_cols):
        b_start = start + input_idx * config.nb
        b = np.asarray(theta[b_start : b_start + config.nb], dtype=float)
        y = np.zeros(horizon, dtype=float)
        for t in range(horizon):
            ar_part = 0.0
            for lag in range(1, config.na + 1):
                if t - lag >= 0:
                    ar_part += a[lag - 1] * y[t - lag]
            input_part = 0.0
            for j in range(config.nb):
                lag = config.nk + j
                if t - lag == 0:
                    input_part += b[j]
            y[t] = ar_part + input_part
            rows.append({"input": input_name, "step": t, "response": y[t]})
    return pd.DataFrame(rows)


irf_df = impulse_response_df(theta_hat, MODEL_CONFIG, horizon=48)
irf_summary = (
    irf_df.groupby("input", as_index=False)
    .agg(cumulative_48=("response", "sum"), peak_abs=("response", lambda s: float(np.max(np.abs(s)))))
    .sort_values("peak_abs", ascending=False)
)
top_inputs = irf_summary.head(8)["input"].tolist()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for input_name in top_inputs:
    tmp = irf_df[irf_df["input"] == input_name]
    axes[0].plot(tmp["step"], tmp["response"], linewidth=1.2, label=input_name)
axes[0].axhline(0, color="black", linewidth=0.9)
axes[0].set_title("Fig. 9a - Impulse response top 8 input")
axes[0].set_xlabel("Bước dự báo")
axes[0].set_ylabel("Phản ứng Soil_Moisture" + (" scaled" if USE_ZSCORE else ""))
axes[0].grid(True, alpha=0.25)
axes[0].legend(ncols=2, fontsize=9)

plot_irf = irf_summary.sort_values("cumulative_48")
colors = np.where(plot_irf["cumulative_48"] >= 0, "#4c78a8", "#e45756")
axes[1].barh(plot_irf["input"], plot_irf["cumulative_48"], color=colors, alpha=0.85)
axes[1].axvline(0, color="black", linewidth=0.9)
axes[1].set_title("Fig. 9b - Tổng đáp ứng 48 bước")
axes[1].set_xlabel("Cumulative response")
axes[1].grid(True, axis="x", alpha=0.25)
fig.tight_layout()
plt.show()

irf_summary.round(5)
"""
        ),
        md("## 10. Lưu artifact"),
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


ar_roots = compute_ar_roots(theta_hat, MODEL_CONFIG)
artifact = {
    "model_type": "ARX",
    "version": VERSION_NAME,
    "version_number": VERSION_NUMBER,
    "notes": "{summary}",
    "data_source": data_source,
    "data_config": DATA_CONFIG.__dict__,
    "split_config": SPLIT_CONFIG.__dict__,
    "model_config": MODEL_CONFIG.__dict__,
    "feature_engineering": {
        "enabled": USE_AUGMENTED,
        "raw_input_cols": list(BASELINE_INPUT_COLS),
        "input_cols": list(INPUT_COLS),
        "new_features": list(INPUT_COLS[len(BASELINE_INPUT_COLS):]) if USE_AUGMENTED else [],
    },
    "normalization": {
        "enabled": USE_ZSCORE,
        "method": "zscore" if USE_ZSCORE else None,
        "fit_on": "train",
        "scaled_columns": list(SCALE_COLS) if USE_ZSCORE else [],
        "stats": scale_stats if USE_ZSCORE else {},
    },
    "regularization": {
        "method": "ridge" if USE_RIDGE else "ols",
        "alpha_grid": RIDGE_ALPHAS if USE_RIDGE else [0.0],
        "best_alpha": best_alpha,
        "selected_by": "max validation FIT_sim" if USE_RIDGE else "ols",
        "penalize_intercept": False,
    },
    "simulation_clip": {
        "enabled": USE_CLIP,
        "source": "train Soil_Moisture quantiles",
        "quantiles": list(CLIP_QUANTILES),
        "bounds_real": [clip_low_real, clip_high_real],
        "bounds_model": list(clip_bounds_model),
    },
    "param_names": MODEL_CONFIG.param_names,
    "theta_hat": theta_hat.tolist(),
    "sigma2": sigma2_hat,
    "matrix_info": matrix_info.to_dict(),
    "alpha_search": alpha_search_df.to_dict(orient="records"),
    "ar_roots": ar_roots,
    "metrics": {
        "train": {
            "fit_1": train_result["metrics_1step_real"]["FIT"],
            "fit_12": train_result["metrics_12step_real"]["FIT"],
            "fit_sim": train_result["metrics_sim_real"]["FIT"],
            "rmse_sim": train_result["metrics_sim_real"]["RMSE"],
        },
        "validation": {
            "fit_1": val_result["metrics_1step_real"]["FIT"],
            "fit_12": val_result["metrics_12step_real"]["FIT"],
            "fit_sim": val_result["metrics_sim_real"]["FIT"],
            "rmse_sim": val_result["metrics_sim_real"]["RMSE"],
        },
        "test": {
            "fit_1": test_result["metrics_1step_real"]["FIT"],
            "fit_12": test_result["metrics_12step_real"]["FIT"],
            "fit_sim": test_result["metrics_sim_real"]["FIT"],
            "rmse_sim": test_result["metrics_sim_real"]["RMSE"],
        },
    },
    "comparison_512": comparison_df.to_dict(orient="records") if not comparison_df.empty else [],
}

VERSION_DIR.mkdir(exist_ok=True)
output_path = VERSION_DIR / f"arx_512_v{VERSION_NUMBER}.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(json_ready(artifact), f, indent=2)
    f.write("\\n")

output_path
""".replace("{summary}", config["summary"])
        ),
    ]
    return nb


def main() -> None:
    VERSION_DIR.mkdir(exist_ok=True)
    for config in VERSIONS:
        nb = make_notebook(config)
        out = VERSION_DIR / f"ARX_512_V{config['n']}.ipynb"
        nbf.write(nb, out)
        print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
