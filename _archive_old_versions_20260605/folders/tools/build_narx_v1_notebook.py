from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT_DIR = Path("NARX")
NOTEBOOK_PATH = OUT_DIR / "NARX_V1.ipynb"


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
# NARX V1

NARX V1 dùng cấu hình mạnh nhất đang có từ ARX search làm mốc đầu tiên:

- Order: `NARX(1,5,2)` theo best ARX order search
- 16 augmented features
- z-score theo train
- MLP nhỏ `[32, 16]`
- early stopping
- clip free-run Q1%-Q99% train

Mục tiêu: kiểm tra NARX phi tuyến có vượt ARX best hiện tại trên `FIT_sim` không.
"""
        ),
        md("## 1. Import và cấu hình"),
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

from arx_pipeline import (
    DataConfig,
    SplitConfig,
    load_or_generate_data,
    split_time_series,
    compute_metrics,
)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 200)

NA = 1
NB = 5
NK = 2
N_STEP = 12
CLIP_QUANTILES = (0.01, 0.99)
RANDOM_STATE = 42
"""
        ),
        md("## 2. Load data và tạo augmented features"),
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


df_full, true_params, data_source = load_or_generate_data(DATA_CONFIG)
df_aug = build_augmented_df(df_full)
df_train, df_val, df_test = split_time_series(df_aug, SPLIT_CONFIG)

print("Data source:", data_source)
print("Rows:", len(df_full), len(df_train), len(df_val), len(df_test))
display(df_train[list(AUGMENTED_INPUT_COLS)].head())
"""
        ),
        md("## 3. Z-score và clip"),
        code(
            """
SCALE_COLS = ("Soil_Moisture", *AUGMENTED_INPUT_COLS)
scale_stats = {}
for col in SCALE_COLS:
    mean = float(df_train[col].astype(float).mean())
    std = float(df_train[col].astype(float).std(ddof=0))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    scale_stats[col] = {"mean": mean, "std": std}


def apply_zscore(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    for col, st in scale_stats.items():
        df_out[col] = (df_out[col].astype(float) - st["mean"]) / st["std"]
    return df_out


def inverse_y(values: np.ndarray) -> np.ndarray:
    st = scale_stats["Soil_Moisture"]
    return np.asarray(values, dtype=float) * st["std"] + st["mean"]


clip_low_real = float(df_train["Soil_Moisture"].quantile(CLIP_QUANTILES[0]))
clip_high_real = float(df_train["Soil_Moisture"].quantile(CLIP_QUANTILES[1]))
y_scale = scale_stats["Soil_Moisture"]
CLIP_BOUNDS_SCALED = (
    float((clip_low_real - y_scale["mean"]) / y_scale["std"]),
    float((clip_high_real - y_scale["mean"]) / y_scale["std"]),
)
CLIP_BOUNDS_REAL = (clip_low_real, clip_high_real)

df_train_z = apply_zscore(df_train)
df_val_z = apply_zscore(df_val)
df_test_z = apply_zscore(df_test)

pd.Series({
    "clip_low_real": CLIP_BOUNDS_REAL[0],
    "clip_high_real": CLIP_BOUNDS_REAL[1],
    "clip_low_scaled": CLIP_BOUNDS_SCALED[0],
    "clip_high_scaled": CLIP_BOUNDS_SCALED[1],
})
"""
        ),
        md("## 4. NARX regression matrix"),
        code(
            """
def max_lag(na: int, nb: int, nk: int) -> int:
    return max(na, nb + nk - 1)


def build_narx_matrix(df: pd.DataFrame, na: int, nb: int, nk: int, input_cols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    y = df["Soil_Moisture"].astype(float).to_numpy()
    inputs = [df[col].astype(float).to_numpy() for col in input_cols]
    lag = max_lag(na, nb, nk)
    n_eff = len(y) - lag
    if n_eff <= 0:
        raise ValueError("Not enough rows")
    n_features = na + len(input_cols) * nb
    x = np.zeros((n_eff, n_features), dtype=float)
    target = np.zeros(n_eff, dtype=float)
    for row_idx in range(n_eff):
        t = row_idx + lag
        row = []
        for y_lag in range(1, na + 1):
            row.append(float(y[t - y_lag]))
        for u in inputs:
            for u_lag in range(nk, nk + nb):
                row.append(float(u[t - u_lag]))
        x[row_idx] = row
        target[row_idx] = float(y[t])
    return x, target


X_train, y_train = build_narx_matrix(df_train_z, NA, NB, NK, AUGMENTED_INPUT_COLS)
X_val, y_val = build_narx_matrix(df_val_z, NA, NB, NK, AUGMENTED_INPUT_COLS)
X_test, y_test = build_narx_matrix(df_test_z, NA, NB, NK, AUGMENTED_INPUT_COLS)

pd.Series({
    "X_train_shape": X_train.shape,
    "X_val_shape": X_val.shape,
    "X_test_shape": X_test.shape,
})
"""
        ),
        md("## 5. Train MLP NARX"),
        code(
            """
start = time.time()
model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=512,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=12,
    random_state=RANDOM_STATE,
    verbose=False,
)
model.fit(X_train, y_train)
train_seconds = time.time() - start

pd.Series({
    "train_seconds": train_seconds,
    "n_iter": model.n_iter_,
    "loss": model.loss_,
    "best_validation_score": getattr(model, "best_validation_score_", np.nan),
})
"""
        ),
        md("## 6. Evaluate 1-step / 12-step / free-run"),
        code(
            """
def make_row_from_state(y_source: np.ndarray, inputs: list[np.ndarray], t: int, na: int, nb: int, nk: int) -> np.ndarray:
    row = []
    for y_lag in range(1, na + 1):
        row.append(float(y_source[t - y_lag]))
    for u in inputs:
        for u_lag in range(nk, nk + nb):
            row.append(float(u[t - u_lag]))
    return np.asarray(row, dtype=float).reshape(1, -1)


def simulate_narx(df: pd.DataFrame, model: MLPRegressor, na: int, nb: int, nk: int, input_cols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    y = df["Soil_Moisture"].astype(float).to_numpy().copy()
    inputs = [df[col].astype(float).to_numpy() for col in input_cols]
    lag = max_lag(na, nb, nk)
    y_sim = y.copy()
    for t in range(lag, len(y)):
        row = make_row_from_state(y_sim, inputs, t, na, nb, nk)
        y_next = float(model.predict(row)[0])
        y_next = float(np.clip(y_next, CLIP_BOUNDS_SCALED[0], CLIP_BOUNDS_SCALED[1]))
        y_sim[t] = y_next
    return y_sim[lag:], y[lag:]


def simulate_narx_n_step(df: pd.DataFrame, model: MLPRegressor, n_steps: int, na: int, nb: int, nk: int, input_cols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if na != 1:
        raise NotImplementedError("NARX V1 vectorized n-step evaluator currently supports na=1")
    y = df["Soil_Moisture"].astype(float).to_numpy().copy()
    inputs = [df[col].astype(float).to_numpy() for col in input_cols]
    lag = max_lag(na, nb, nk)
    target_t = np.arange(lag, len(y))
    origins = np.maximum(lag - 1, target_t - n_steps)
    horizons = target_t - origins
    current_y = y[origins].copy()
    predicted = np.full(len(target_t), np.nan, dtype=float)

    for h in range(1, n_steps + 1):
        active = horizons >= h
        if not np.any(active):
            continue
        step_t = origins[active] + h
        rows = [current_y[active]]
        for u in inputs:
            for u_lag in range(nk, nk + nb):
                rows.append(u[step_t - u_lag])
        x_step = np.column_stack(rows)
        y_next = model.predict(x_step)
        y_next = np.clip(y_next, CLIP_BOUNDS_SCALED[0], CLIP_BOUNDS_SCALED[1])
        active_idx = np.where(active)[0]
        current_y[active_idx] = y_next
        done_idx = active_idx[horizons[active_idx] == h]
        if len(done_idx):
            predicted[done_idx] = current_y[done_idx]

    return predicted, y[lag:]


def evaluate_split(name: str, df_z: pd.DataFrame, x_mat: np.ndarray, y_vec: np.ndarray) -> dict:
    y_pred_1 = model.predict(x_mat)
    y_pred_sim, y_true_sim = simulate_narx(df_z, model, NA, NB, NK, AUGMENTED_INPUT_COLS)
    y_pred_12, _ = simulate_narx_n_step(df_z, model, N_STEP, NA, NB, NK, AUGMENTED_INPUT_COLS)
    return {
        "name": name,
        "metrics_1step": compute_metrics(inverse_y(y_vec), inverse_y(y_pred_1), X_train.shape[1]),
        "metrics_12step": compute_metrics(inverse_y(y_true_sim), inverse_y(y_pred_12), X_train.shape[1]),
        "metrics_sim": compute_metrics(inverse_y(y_true_sim), inverse_y(y_pred_sim), X_train.shape[1]),
        "arrays": {
            "y_true_sim": y_true_sim,
            "y_pred_sim": y_pred_sim,
            "y_pred_12step": y_pred_12,
        },
    }


train_eval = evaluate_split("Train", df_train_z, X_train, y_train)
val_eval = evaluate_split("Validation", df_val_z, X_val, y_val)
test_eval = evaluate_split("Test", df_test_z, X_test, y_test)


def row(ev: dict) -> dict:
    return {
        "split": ev["name"],
        "FIT_1step": ev["metrics_1step"]["FIT"],
        "RMSE_1step": ev["metrics_1step"]["RMSE"],
        "FIT_12step": ev["metrics_12step"]["FIT"],
        "RMSE_12step": ev["metrics_12step"]["RMSE"],
        "FIT_sim": ev["metrics_sim"]["FIT"],
        "RMSE_sim": ev["metrics_sim"]["RMSE"],
        "Bias_sim": ev["metrics_sim"]["Bias"],
    }


metrics_df = pd.DataFrame([row(train_eval), row(val_eval), row(test_eval)]).round(4)
metrics_df
"""
        ),
        md("## 7. So sánh với ARX best"),
        code(
            """
comparison_rows = []
for label, path in [
    ("ARX Search V1 best", PROJECT_ROOT / "ARX_Model_VersionSearch" / "arx_order_search_v1.json"),
    ("ARX 512 V6", PROJECT_ROOT / "ARX_Model_Version 512" / "arx_512_v6.json"),
]:
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        artifact = json.load(f)
    if label == "ARX Search V1 best":
        best = artifact["best_by_validation"]
        comparison_rows.append({
            "model": label,
            "order": best["order"],
            "val_FIT_sim": best["val_FIT_sim"],
            "test_FIT_sim": best["test_FIT_sim"],
            "test_RMSE_sim": best["test_RMSE_sim"],
        })
    else:
        comparison_rows.append({
            "model": label,
            "order": f"({artifact['model_config']['na']},{artifact['model_config']['nb']},{artifact['model_config']['nk']})",
            "val_FIT_sim": artifact["metrics"]["validation"]["fit_sim"],
            "test_FIT_sim": artifact["metrics"]["test"]["fit_sim"],
            "test_RMSE_sim": artifact["metrics"]["test"]["rmse_sim"],
        })

comparison_rows.append({
    "model": "NARX V1 MLP",
    "order": f"({NA},{NB},{NK})",
    "val_FIT_sim": float(metrics_df.loc[metrics_df["split"] == "Validation", "FIT_sim"].iloc[0]),
    "test_FIT_sim": float(metrics_df.loc[metrics_df["split"] == "Test", "FIT_sim"].iloc[0]),
    "test_RMSE_sim": float(metrics_df.loc[metrics_df["split"] == "Test", "RMSE_sim"].iloc[0]),
})

comparison_df = pd.DataFrame(comparison_rows)
comparison_df["test_gain_vs_arx_search_best"] = comparison_df["test_FIT_sim"] - comparison_df.loc[0, "test_FIT_sim"]
comparison_df.round(4)
"""
        ),
        md("## 8. Lưu artifact"),
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
    "version": "narx_v1_mlp_152_augmented_clip",
    "order": {"na": NA, "nb": NB, "nk": NK},
    "notes": "NARX MLP using ARX search best order (1,5,2), 16 augmented features, z-score train normalization, output clip.",
    "data_source": data_source,
    "input_cols": list(AUGMENTED_INPUT_COLS),
    "normalization": {
        "method": "zscore",
        "fit_on": "train",
        "scaled_columns": list(SCALE_COLS),
        "stats": scale_stats,
    },
    "simulation_clip": {
        "enabled": True,
        "quantiles": list(CLIP_QUANTILES),
        "bounds_real": list(CLIP_BOUNDS_REAL),
        "bounds_scaled": list(CLIP_BOUNDS_SCALED),
    },
    "mlp_config": {
        "hidden_layer_sizes": [32, 16],
        "activation": "relu",
        "alpha": 1e-4,
        "batch_size": 512,
        "learning_rate_init": 1e-3,
        "max_iter": 200,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 12,
        "random_state": RANDOM_STATE,
        "n_iter": int(model.n_iter_),
        "loss": float(model.loss_),
        "best_validation_score": float(getattr(model, "best_validation_score_", np.nan)),
        "train_seconds": float(train_seconds),
    },
    "matrix_info": {
        "x_train_shape": list(X_train.shape),
        "x_val_shape": list(X_val.shape),
        "x_test_shape": list(X_test.shape),
    },
    "metrics": {
        "train": {
            "fit_1": train_eval["metrics_1step"]["FIT"],
            "fit_12": train_eval["metrics_12step"]["FIT"],
            "fit_sim": train_eval["metrics_sim"]["FIT"],
            "rmse_sim": train_eval["metrics_sim"]["RMSE"],
        },
        "validation": {
            "fit_1": val_eval["metrics_1step"]["FIT"],
            "fit_12": val_eval["metrics_12step"]["FIT"],
            "fit_sim": val_eval["metrics_sim"]["FIT"],
            "rmse_sim": val_eval["metrics_sim"]["RMSE"],
        },
        "test": {
            "fit_1": test_eval["metrics_1step"]["FIT"],
            "fit_12": test_eval["metrics_12step"]["FIT"],
            "fit_sim": test_eval["metrics_sim"]["FIT"],
            "rmse_sim": test_eval["metrics_sim"]["RMSE"],
        },
    },
    "comparison": comparison_df.to_dict(orient="records"),
}

OUT_DIR.mkdir(exist_ok=True)
out_path = OUT_DIR / "narx_v1.json"
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
