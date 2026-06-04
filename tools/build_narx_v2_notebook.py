from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT_DIR = Path("NARX")
NOTEBOOK_PATH = OUT_DIR / "NARX_V2.ipynb"


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
# NARX V2

V2 search hyperparameter MLP trên order `NARX(1,5,2)`:

- 16 augmented features
- z-score train
- clip free-run
- chọn theo validation `FIT_sim`

V2 dùng `narx_pipeline.py` riêng, không dùng fit/simulate của ARX.
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

HIDDEN_OPTIONS = [(16,), (32,), (32, 16), (64, 32)]
ACTIVATIONS = ["tanh", "relu"]
ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2]
"""
        ),
        md("## 2. Data và NARX matrix"),
        code(
            """
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
    "X_val": X_val.shape,
    "X_test": X_test.shape,
    "clip_real": clip_bounds_real,
    "clip_scaled": clip_bounds_scaled,
})
"""
        ),
        md("## 3. Helper đánh giá"),
        code(
            """
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
        X_train.shape[1],
    )


def train_candidate(hidden: tuple[int, ...], activation: str, alpha: float, seed: int) -> tuple[MLPRegressor, dict]:
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=activation,
        solver="adam",
        alpha=alpha,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=250,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=seed,
        verbose=False,
    )
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    val_1 = one_step_metrics(model, X_val, y_val)
    val_sim = sim_metrics(model, df_val_z)
    test_sim = sim_metrics(model, df_test_z)
    return model, {
        "hidden": list(hidden),
        "activation": activation,
        "alpha": float(alpha),
        "seed": int(seed),
        "n_iter": int(model.n_iter_),
        "loss": float(model.loss_),
        "best_validation_score": float(getattr(model, "best_validation_score_", np.nan)),
        "train_seconds": float(elapsed),
        "val_FIT_1step": val_1["FIT"],
        "val_RMSE_1step": val_1["RMSE"],
        "val_FIT_sim": val_sim["FIT"],
        "val_RMSE_sim": val_sim["RMSE"],
        "test_FIT_sim": test_sim["FIT"],
        "test_RMSE_sim": test_sim["RMSE"],
    }
"""
        ),
        md("## 4. Hyperparameter search"),
        code(
            """
rows = []
models = {}
candidate_id = 0
total = len(HIDDEN_OPTIONS) * len(ACTIVATIONS) * len(ALPHAS)

for hidden in HIDDEN_OPTIONS:
    for activation in ACTIVATIONS:
        for alpha in ALPHAS:
            candidate_id += 1
            print(f"{candidate_id}/{total}: hidden={hidden}, activation={activation}, alpha={alpha}")
            model, row = train_candidate(hidden, activation, alpha, RANDOM_STATE)
            row["candidate_id"] = candidate_id
            rows.append(row)
            models[candidate_id] = model

search_df = pd.DataFrame(rows).sort_values(
    ["val_FIT_sim", "val_FIT_1step"],
    ascending=[False, False],
).reset_index(drop=True)

best_row = search_df.iloc[0].to_dict()
best_model = models[int(best_row["candidate_id"])]
search_df.round(4)
"""
        ),
        md("## 5. Final metrics và so sánh"),
        code(
            """
train_1 = one_step_metrics(best_model, X_train, y_train)
val_1 = one_step_metrics(best_model, X_val, y_val)
test_1 = one_step_metrics(best_model, X_test, y_test)
train_sim = sim_metrics(best_model, df_train_z)
val_sim = sim_metrics(best_model, df_val_z)
test_sim = sim_metrics(best_model, df_test_z)

metrics_df = pd.DataFrame([
    {"split": "Train", "FIT_1step": train_1["FIT"], "RMSE_1step": train_1["RMSE"], "FIT_sim": train_sim["FIT"], "RMSE_sim": train_sim["RMSE"], "Bias_sim": train_sim["Bias"]},
    {"split": "Validation", "FIT_1step": val_1["FIT"], "RMSE_1step": val_1["RMSE"], "FIT_sim": val_sim["FIT"], "RMSE_sim": val_sim["RMSE"], "Bias_sim": val_sim["Bias"]},
    {"split": "Test", "FIT_1step": test_1["FIT"], "RMSE_1step": test_1["RMSE"], "FIT_sim": test_sim["FIT"], "RMSE_sim": test_sim["RMSE"], "Bias_sim": test_sim["Bias"]},
]).round(4)

comparison_rows = []
for label, path in [
    ("ARX Search V1 best", PROJECT_ROOT / "ARX_Model_VersionSearch" / "arx_order_search_v1.json"),
    ("NARX V1", OUT_DIR / "narx_v1.json"),
]:
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        artifact = json.load(f)
    if label.startswith("ARX"):
        best = artifact["best_by_validation"]
        comparison_rows.append({"model": label, "val_FIT_sim": best["val_FIT_sim"], "test_FIT_sim": best["test_FIT_sim"], "test_RMSE_sim": best["test_RMSE_sim"]})
    else:
        comparison_rows.append({"model": label, "val_FIT_sim": artifact["metrics"]["validation"]["fit_sim"], "test_FIT_sim": artifact["metrics"]["test"]["fit_sim"], "test_RMSE_sim": artifact["metrics"]["test"]["rmse_sim"]})

comparison_rows.append({"model": "NARX V2 best", "val_FIT_sim": val_sim["FIT"], "test_FIT_sim": test_sim["FIT"], "test_RMSE_sim": test_sim["RMSE"]})
comparison_df = pd.DataFrame(comparison_rows)
comparison_df["test_gain_vs_arx_search_best"] = comparison_df["test_FIT_sim"] - comparison_df.loc[0, "test_FIT_sim"]

display(metrics_df)
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
    "version": "narx_v2_mlp_hyperparameter_search",
    "order": {"na": NA, "nb": NB, "nk": NK},
    "input_cols": list(AUGMENTED_INPUT_COLS),
    "normalization": {"method": "zscore", "fit_on": "train", "stats": scale_stats},
    "simulation_clip": {"enabled": True, "bounds_real": list(clip_bounds_real), "bounds_scaled": list(clip_bounds_scaled)},
    "search_space": {"hidden_options": [list(v) for v in HIDDEN_OPTIONS], "activations": ACTIVATIONS, "alphas": ALPHAS},
    "best_candidate": best_row,
    "all_results": search_df.to_dict(orient="records"),
    "metrics": {
        "train": {"fit_1": train_1["FIT"], "fit_sim": train_sim["FIT"], "rmse_sim": train_sim["RMSE"]},
        "validation": {"fit_1": val_1["FIT"], "fit_sim": val_sim["FIT"], "rmse_sim": val_sim["RMSE"]},
        "test": {"fit_1": test_1["FIT"], "fit_sim": test_sim["FIT"], "rmse_sim": test_sim["RMSE"]},
    },
    "comparison": comparison_df.to_dict(orient="records"),
}

OUT_DIR.mkdir(exist_ok=True)
out_path = OUT_DIR / "narx_v2.json"
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
