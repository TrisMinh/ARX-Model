from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT_DIR = Path("NARX")
NOTEBOOK_PATH = OUT_DIR / "NARX_V4.ipynb"


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
# NARX V4

V4 thử cải thiện free-run bằng **closed-loop retraining**:

1. Train MLP base bằng 1-step teacher forcing.
2. Chạy base model free-run trên train.
3. Tạo training matrix thứ hai, thay `y(t-1)` bằng `y_sim(t-1)`.
4. Train MLP final trên dữ liệu gộp: true-lag + simulated-lag.

Mục tiêu là giảm mismatch giữa lúc train dùng `y_true` và lúc free-run dùng `y_pred`.
"""
        ),
        md("## 1. Import và data"),
        code(
            """
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

WORK_DIR = Path.cwd()
PROJECT_ROOT = WORK_DIR.parent if WORK_DIR.name == "NARX" else WORK_DIR
OUT_DIR = PROJECT_ROOT / "NARX"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arx_pipeline import DataConfig, SplitConfig, load_or_generate_data, split_time_series
from narx_pipeline import (
    NarxConfig,
    build_augmented_df,
    fit_zscore_stats,
    apply_zscore,
    inverse_zscore_y,
    scaled_clip_bounds,
    build_narx_matrix,
    simulate_narx,
    compute_metrics,
)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 200)

NA, NB, NK = 1, 5, 2
CLIP_QUANTILES = (0.01, 0.99)
RANDOM_STATE = 42
HIDDEN = (32, 16)
ACTIVATION = "relu"
ALPHA = 1e-5

DATA_CONFIG = DataConfig(
    csv_path=PROJECT_ROOT / "greenhouse_data.csv",
    generator_script_path=PROJECT_ROOT / "data_generator.py",
    force_regenerate_from_script=False,
    auto_save_generated_csv=True,
)
SPLIT_CONFIG = SplitConfig(train_ratio=0.60, val_ratio=0.20)

BASELINE_INPUT_COLS = ("Temperature", "Humidity", "Light", "Drip", "Mist", "Fan")
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
df_aug = build_augmented_df(df_full)
df_train, df_val, df_test = split_time_series(df_aug, SPLIT_CONFIG)
SCALE_COLS = ("Soil_Moisture", *AUGMENTED_INPUT_COLS)
scale_stats = fit_zscore_stats(df_train, SCALE_COLS)
clip_bounds_real, clip_bounds_scaled = scaled_clip_bounds(df_train, scale_stats, CLIP_QUANTILES)
df_train_z = apply_zscore(df_train, scale_stats)
df_val_z = apply_zscore(df_val, scale_stats)
df_test_z = apply_zscore(df_test, scale_stats)

CONFIG = NarxConfig(
    na=NA,
    nb=NB,
    nk=NK,
    input_cols=AUGMENTED_INPUT_COLS,
    simulation_clip=clip_bounds_scaled,
)

X_train, y_train = build_narx_matrix(df_train_z, CONFIG)
X_val, y_val = build_narx_matrix(df_val_z, CONFIG)
X_test, y_test = build_narx_matrix(df_test_z, CONFIG)

pd.Series({
    "data_source": data_source,
    "X_train": X_train.shape,
    "clip_real": clip_bounds_real,
    "clip_scaled": clip_bounds_scaled,
})
"""
        ),
        md("## 2. Helper train/evaluate"),
        code(
            """
def make_mlp(seed: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation=ACTIVATION,
        solver="adam",
        alpha=ALPHA,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=250,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=seed,
        verbose=False,
    )


def one_step_metrics(model: MLPRegressor, x_mat: np.ndarray, y_vec: np.ndarray) -> dict[str, float]:
    pred = model.predict(x_mat)
    return compute_metrics(
        inverse_zscore_y(y_vec, scale_stats),
        inverse_zscore_y(pred, scale_stats),
        x_mat.shape[1],
    )


def sim_metrics(model: MLPRegressor, df_z: pd.DataFrame) -> dict[str, float]:
    y_pred, y_true = simulate_narx(df_z, model, CONFIG)
    return compute_metrics(
        inverse_zscore_y(y_true, scale_stats),
        inverse_zscore_y(y_pred, scale_stats),
        CONFIG.n_features,
    )


def metric_row(name: str, model: MLPRegressor) -> dict:
    if name == "Train":
        x_mat, y_vec, df_z = X_train, y_train, df_train_z
    elif name == "Validation":
        x_mat, y_vec, df_z = X_val, y_val, df_val_z
    elif name == "Test":
        x_mat, y_vec, df_z = X_test, y_test, df_test_z
    else:
        raise ValueError(name)
    m1 = one_step_metrics(model, x_mat, y_vec)
    ms = sim_metrics(model, df_z)
    return {
        "split": name,
        "FIT_1step": m1["FIT"],
        "RMSE_1step": m1["RMSE"],
        "FIT_sim": ms["FIT"],
        "RMSE_sim": ms["RMSE"],
        "Bias_sim": ms["Bias"],
    }
"""
        ),
        md("## 3. Train base model và tạo closed-loop matrix"),
        code(
            """
start = time.time()
base_model = make_mlp(RANDOM_STATE)
base_model.fit(X_train, y_train)
base_train_seconds = time.time() - start

y_train_sim, y_train_true_sim = simulate_narx(df_train_z, base_model, CONFIG)
lag = CONFIG.max_lag

X_train_simlag = X_train.copy()
# Với NA=1, cột đầu tiên là y(t-1). Hàng row_idx ứng với t=row_idx+lag.
# Dùng y_sim(t-1), tức y_train_sim[row_idx-1] nếu t-1 >= lag; những hàng đầu giữ y_true.
for row_idx in range(1, len(X_train_simlag)):
    X_train_simlag[row_idx, 0] = y_train_sim[row_idx - 1]

X_aug = np.vstack([X_train, X_train_simlag])
y_aug = np.concatenate([y_train, y_train])

pd.Series({
    "base_n_iter": base_model.n_iter_,
    "base_loss": base_model.loss_,
    "base_train_seconds": base_train_seconds,
    "X_train": X_train.shape,
    "X_train_simlag": X_train_simlag.shape,
    "X_aug": X_aug.shape,
})
"""
        ),
        md("## 4. Train final closed-loop model"),
        code(
            """
start = time.time()
final_model = make_mlp(RANDOM_STATE + 100)
final_model.fit(X_aug, y_aug)
final_train_seconds = time.time() - start

base_metrics = pd.DataFrame([metric_row(split, base_model) for split in ["Train", "Validation", "Test"]]).round(4)
final_metrics = pd.DataFrame([metric_row(split, final_model) for split in ["Train", "Validation", "Test"]]).round(4)

print("Base model")
display(base_metrics)
print("Closed-loop retrained model")
display(final_metrics)
pd.Series({
    "final_n_iter": final_model.n_iter_,
    "final_loss": final_model.loss_,
    "final_train_seconds": final_train_seconds,
})
"""
        ),
        md("## 5. So sánh với NARX cũ và ARX"),
        code(
            """
comparison_rows = []
for label, path in [
    ("ARX Search V1 best", PROJECT_ROOT / "ARX_Model_VersionSearch" / "arx_order_search_v1.json"),
    ("NARX V1", OUT_DIR / "narx_v1.json"),
    ("NARX V2", OUT_DIR / "narx_v2.json"),
    ("NARX V3", OUT_DIR / "narx_v3.json"),
]:
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        artifact = json.load(f)
    if label.startswith("ARX"):
        best = artifact["best_by_validation"]
        comparison_rows.append({"model": label, "val_FIT_sim": best["val_FIT_sim"], "test_FIT_sim": best["test_FIT_sim"], "test_RMSE_sim": best["test_RMSE_sim"]})
    elif label == "NARX V3":
        best = artifact["best_candidate"]
        comparison_rows.append({"model": label, "val_FIT_sim": best["val_FIT_sim"], "test_FIT_sim": best["test_FIT_sim"], "test_RMSE_sim": best["test_RMSE_sim"]})
    else:
        comparison_rows.append({"model": label, "val_FIT_sim": artifact["metrics"]["validation"]["fit_sim"], "test_FIT_sim": artifact["metrics"]["test"]["fit_sim"], "test_RMSE_sim": artifact["metrics"]["test"]["rmse_sim"]})

comparison_rows.append({
    "model": "NARX V4 closed-loop",
    "val_FIT_sim": float(final_metrics.loc[final_metrics["split"] == "Validation", "FIT_sim"].iloc[0]),
    "test_FIT_sim": float(final_metrics.loc[final_metrics["split"] == "Test", "FIT_sim"].iloc[0]),
    "test_RMSE_sim": float(final_metrics.loc[final_metrics["split"] == "Test", "RMSE_sim"].iloc[0]),
})
comparison_df = pd.DataFrame(comparison_rows)
comparison_df["test_gain_vs_arx_search_best"] = comparison_df["test_FIT_sim"] - comparison_df.loc[0, "test_FIT_sim"]
comparison_df.round(4)
"""
        ),
        md("## 6. Lưu artifact"),
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
    "model_type": "NARX",
    "version": "narx_v4_closed_loop_retraining",
    "order": {"na": NA, "nb": NB, "nk": NK},
    "input_cols": list(AUGMENTED_INPUT_COLS),
    "normalization": {"method": "zscore", "fit_on": "train", "stats": scale_stats},
    "simulation_clip": {"enabled": True, "bounds_real": list(clip_bounds_real), "bounds_scaled": list(clip_bounds_scaled)},
    "mlp_config": {"hidden": list(HIDDEN), "activation": ACTIVATION, "alpha": ALPHA},
    "closed_loop_training": {
        "enabled": True,
        "base_train_rows": int(len(X_train)),
        "simlag_train_rows": int(len(X_train_simlag)),
        "final_train_rows": int(len(X_aug)),
        "base_n_iter": int(base_model.n_iter_),
        "base_loss": float(base_model.loss_),
        "final_n_iter": int(final_model.n_iter_),
        "final_loss": float(final_model.loss_),
        "base_train_seconds": float(base_train_seconds),
        "final_train_seconds": float(final_train_seconds),
    },
    "base_metrics_table": base_metrics.to_dict(orient="records"),
    "metrics_table": final_metrics.to_dict(orient="records"),
    "metrics": {
        "train": {
            "fit_1": float(final_metrics.loc[final_metrics["split"] == "Train", "FIT_1step"].iloc[0]),
            "fit_sim": float(final_metrics.loc[final_metrics["split"] == "Train", "FIT_sim"].iloc[0]),
            "rmse_sim": float(final_metrics.loc[final_metrics["split"] == "Train", "RMSE_sim"].iloc[0]),
        },
        "validation": {
            "fit_1": float(final_metrics.loc[final_metrics["split"] == "Validation", "FIT_1step"].iloc[0]),
            "fit_sim": float(final_metrics.loc[final_metrics["split"] == "Validation", "FIT_sim"].iloc[0]),
            "rmse_sim": float(final_metrics.loc[final_metrics["split"] == "Validation", "RMSE_sim"].iloc[0]),
        },
        "test": {
            "fit_1": float(final_metrics.loc[final_metrics["split"] == "Test", "FIT_1step"].iloc[0]),
            "fit_sim": float(final_metrics.loc[final_metrics["split"] == "Test", "FIT_sim"].iloc[0]),
            "rmse_sim": float(final_metrics.loc[final_metrics["split"] == "Test", "RMSE_sim"].iloc[0]),
        },
    },
    "comparison": comparison_df.to_dict(orient="records"),
}

OUT_DIR.mkdir(exist_ok=True)
out_path = OUT_DIR / "narx_v4.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(json_ready(artifact), f, indent=2)
    f.write("\\n")
out_path
"""
        ),
    ]

    OUT_DIR.mkdir(exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH.resolve()}")


if __name__ == "__main__":
    main()
