from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT_DIR = Path("NARX")
NOTEBOOK_PATH = OUT_DIR / "NARX_V3.ipynb"


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
# NARX V3

V3 search một nhóm order trọng điểm cho NARX MLP.

Cấu hình MLP lấy từ V2 best:

- hidden `[32, 16]`
- activation `relu`
- alpha `1e-5`

Chọn theo validation `FIT_sim`, sau đó kiểm tra test.
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

ORDER_CANDIDATES = [
    (1, 1, 2),
    (1, 3, 2),
    (1, 5, 2),
    (2, 1, 2),
    (2, 3, 2),
    (2, 5, 2),
    (3, 1, 2),
    (3, 3, 2),
    (5, 1, 2),
]
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
"""
        ),
        md("## 2. Search order"),
        code(
            """
def one_step_metrics(model: MLPRegressor, x_mat: np.ndarray, y_vec: np.ndarray) -> dict[str, float]:
    pred = model.predict(x_mat)
    return compute_metrics(
        inverse_zscore_y(y_vec, scale_stats),
        inverse_zscore_y(pred, scale_stats),
        x_mat.shape[1],
    )


def sim_metrics(model: MLPRegressor, df_z: pd.DataFrame, config: NarxConfig) -> dict[str, float]:
    y_pred, y_true = simulate_narx(df_z, model, config)
    return compute_metrics(
        inverse_zscore_y(y_true, scale_stats),
        inverse_zscore_y(y_pred, scale_stats),
        config.n_features,
    )


def train_for_order(na: int, nb: int, nk: int) -> tuple[MLPRegressor, dict]:
    config = NarxConfig(
        na=na,
        nb=nb,
        nk=nk,
        input_cols=AUGMENTED_INPUT_COLS,
        simulation_clip=clip_bounds_scaled,
    )
    x_train, y_train = build_narx_matrix(df_train_z, config)
    x_val, y_val = build_narx_matrix(df_val_z, config)
    model = MLPRegressor(
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
        random_state=RANDOM_STATE,
        verbose=False,
    )
    start = time.time()
    model.fit(x_train, y_train)
    elapsed = time.time() - start
    val_1 = one_step_metrics(model, x_val, y_val)
    val_sim = sim_metrics(model, df_val_z, config)
    test_sim = sim_metrics(model, df_test_z, config)
    return model, {
        "order": f"({na},{nb},{nk})",
        "na": na,
        "nb": nb,
        "nk": nk,
        "n_features": config.n_features,
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


rows = []
models = {}
for idx, order in enumerate(ORDER_CANDIDATES, start=1):
    print(f"{idx}/{len(ORDER_CANDIDATES)} order={order}")
    model, row = train_for_order(*order)
    row["candidate_id"] = idx
    rows.append(row)
    models[idx] = model

search_df = pd.DataFrame(rows).sort_values(
    ["val_FIT_sim", "val_FIT_1step"],
    ascending=[False, False],
).reset_index(drop=True)
best_row = search_df.iloc[0].to_dict()
best_model = models[int(best_row["candidate_id"])]
search_df.round(4)
"""
        ),
        md("## 3. So sánh"),
        code(
            """
comparison_rows = []
for label, path in [
    ("ARX Search V1 best", PROJECT_ROOT / "ARX_Model_VersionSearch" / "arx_order_search_v1.json"),
    ("NARX V1", OUT_DIR / "narx_v1.json"),
    ("NARX V2", OUT_DIR / "narx_v2.json"),
]:
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        artifact = json.load(f)
    if label.startswith("ARX"):
        best = artifact["best_by_validation"]
        comparison_rows.append({"model": label, "order": best["order"], "val_FIT_sim": best["val_FIT_sim"], "test_FIT_sim": best["test_FIT_sim"], "test_RMSE_sim": best["test_RMSE_sim"]})
    elif label == "NARX V2":
        comparison_rows.append({"model": label, "order": "(1,5,2)", "val_FIT_sim": artifact["metrics"]["validation"]["fit_sim"], "test_FIT_sim": artifact["metrics"]["test"]["fit_sim"], "test_RMSE_sim": artifact["metrics"]["test"]["rmse_sim"]})
    else:
        comparison_rows.append({"model": label, "order": f"({artifact['order']['na']},{artifact['order']['nb']},{artifact['order']['nk']})", "val_FIT_sim": artifact["metrics"]["validation"]["fit_sim"], "test_FIT_sim": artifact["metrics"]["test"]["fit_sim"], "test_RMSE_sim": artifact["metrics"]["test"]["rmse_sim"]})

comparison_rows.append({"model": "NARX V3 best", "order": best_row["order"], "val_FIT_sim": best_row["val_FIT_sim"], "test_FIT_sim": best_row["test_FIT_sim"], "test_RMSE_sim": best_row["test_RMSE_sim"]})
comparison_df = pd.DataFrame(comparison_rows)
comparison_df["test_gain_vs_arx_search_best"] = comparison_df["test_FIT_sim"] - comparison_df.loc[0, "test_FIT_sim"]
comparison_df.round(4)
"""
        ),
        md("## 4. Lưu artifact"),
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
    "version": "narx_v3_order_search",
    "order_candidates": [list(v) for v in ORDER_CANDIDATES],
    "mlp_config": {"hidden": list(HIDDEN), "activation": ACTIVATION, "alpha": ALPHA},
    "input_cols": list(AUGMENTED_INPUT_COLS),
    "normalization": {"method": "zscore", "fit_on": "train", "stats": scale_stats},
    "simulation_clip": {"enabled": True, "bounds_real": list(clip_bounds_real), "bounds_scaled": list(clip_bounds_scaled)},
    "best_candidate": best_row,
    "all_results": search_df.to_dict(orient="records"),
    "comparison": comparison_df.to_dict(orient="records"),
}
OUT_DIR.mkdir(exist_ok=True)
out_path = OUT_DIR / "narx_v3.json"
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
