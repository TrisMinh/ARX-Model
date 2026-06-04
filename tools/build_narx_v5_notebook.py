from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT_DIR = Path("NARX")
NOTEBOOK_PATH = OUT_DIR / "NARX_V5.ipynb"


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
# NARX V5

V5 chuyển khỏi MLP sang nonlinear tree ensemble cho tabular lag features:

- Model: `HistGradientBoostingRegressor`
- Search nhiều order trọng điểm, không giả định ARX best là NARX best
- 16 augmented features
- z-score target/input để giữ cùng scale pipeline
- clip free-run Q1%-Q99% train
- chọn theo validation `FIT_sim`

Mục tiêu đầu tiên: vượt NARX MLP tốt nhất. Mục tiêu cao hơn: tiến gần hoặc vượt ARX best.
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
from sklearn.ensemble import HistGradientBoostingRegressor

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

pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 220)

ORDER_CANDIDATES = [
    (1, 1, 2),
    (1, 2, 2),
    (1, 3, 2),
    (1, 5, 2),
    (2, 1, 2),
    (2, 3, 2),
    (2, 5, 2),
    (3, 1, 2),
    (5, 1, 2),
    (5, 3, 2),
]

MODEL_CANDIDATES = [
    {"max_iter": 120, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.0},
    {"max_iter": 180, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 0.0},
    {"max_iter": 120, "learning_rate": 0.10, "max_leaf_nodes": 15, "l2_regularization": 1e-4},
    {"max_iter": 180, "learning_rate": 0.10, "max_leaf_nodes": 31, "l2_regularization": 1e-4},
]

CLIP_QUANTILES = (0.01, 0.99)
RANDOM_STATE = 42
TOP_K_TEST = 8
"""
        ),
        md("## 2. Data và preprocessing"),
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

pd.Series({
    "data_source": data_source,
    "train_rows": len(df_train),
    "val_rows": len(df_val),
    "test_rows": len(df_test),
    "clip_real": clip_bounds_real,
    "clip_scaled": clip_bounds_scaled,
})
"""
        ),
        md("## 3. Helper train/evaluate"),
        code(
            """
def make_model(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=int(params["max_iter"]),
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
    )


def one_step_metrics(model, x_mat: np.ndarray, y_vec: np.ndarray) -> dict[str, float]:
    pred = model.predict(x_mat)
    return compute_metrics(
        inverse_zscore_y(y_vec, scale_stats),
        inverse_zscore_y(pred, scale_stats),
        x_mat.shape[1],
    )


def sim_metrics(model, df_z: pd.DataFrame, config: NarxConfig) -> dict[str, float]:
    y_pred, y_true = simulate_narx(df_z, model, config)
    return compute_metrics(
        inverse_zscore_y(y_true, scale_stats),
        inverse_zscore_y(y_pred, scale_stats),
        config.n_features,
    )


def train_candidate(order: tuple[int, int, int], params: dict) -> tuple[HistGradientBoostingRegressor, dict]:
    na, nb, nk = order
    config = NarxConfig(
        na=na,
        nb=nb,
        nk=nk,
        input_cols=AUGMENTED_INPUT_COLS,
        simulation_clip=clip_bounds_scaled,
    )
    x_train, y_train = build_narx_matrix(df_train_z, config)
    x_val, y_val = build_narx_matrix(df_val_z, config)
    model = make_model(params)
    start = time.time()
    model.fit(x_train, y_train)
    train_seconds = time.time() - start
    val_1 = one_step_metrics(model, x_val, y_val)
    val_sim = sim_metrics(model, df_val_z, config)
    return model, {
        "order": f"({na},{nb},{nk})",
        "na": na,
        "nb": nb,
        "nk": nk,
        "n_features": config.n_features,
        "max_iter": int(params["max_iter"]),
        "learning_rate": float(params["learning_rate"]),
        "max_leaf_nodes": int(params["max_leaf_nodes"]),
        "l2_regularization": float(params["l2_regularization"]),
        "n_iter": int(model.n_iter_),
        "train_seconds": float(train_seconds),
        "val_FIT_1step": val_1["FIT"],
        "val_RMSE_1step": val_1["RMSE"],
        "val_FIT_sim": val_sim["FIT"],
        "val_RMSE_sim": val_sim["RMSE"],
    }


def test_candidate(model, order: tuple[int, int, int]) -> dict[str, float]:
    na, nb, nk = order
    config = NarxConfig(
        na=na,
        nb=nb,
        nk=nk,
        input_cols=AUGMENTED_INPUT_COLS,
        simulation_clip=clip_bounds_scaled,
    )
    x_test, y_test = build_narx_matrix(df_test_z, config)
    test_1 = one_step_metrics(model, x_test, y_test)
    test_sim = sim_metrics(model, df_test_z, config)
    return {
        "test_FIT_1step": test_1["FIT"],
        "test_RMSE_1step": test_1["RMSE"],
        "test_FIT_sim": test_sim["FIT"],
        "test_RMSE_sim": test_sim["RMSE"],
        "test_Bias_sim": test_sim["Bias"],
    }
"""
        ),
        md("## 4. Search validation"),
        code(
            """
rows = []
models = {}
orders_by_id = {}
params_by_id = {}
candidate_id = 0
total = len(ORDER_CANDIDATES) * len(MODEL_CANDIDATES)

for order in ORDER_CANDIDATES:
    for params in MODEL_CANDIDATES:
        candidate_id += 1
        print(f"{candidate_id}/{total}: order={order}, params={params}")
        model, row = train_candidate(order, params)
        row["candidate_id"] = candidate_id
        rows.append(row)
        models[candidate_id] = model
        orders_by_id[candidate_id] = order
        params_by_id[candidate_id] = params

search_df = pd.DataFrame(rows).sort_values(
    ["val_FIT_sim", "val_FIT_1step"],
    ascending=[False, False],
).reset_index(drop=True)

search_df.head(20).round(4)
"""
        ),
        md("## 5. Test top candidates"),
        code(
            """
test_rows = []
for _, row in search_df.head(TOP_K_TEST).iterrows():
    cid = int(row["candidate_id"])
    model = models[cid]
    order = orders_by_id[cid]
    test_payload = test_candidate(model, order)
    merged = row.to_dict()
    merged.update(test_payload)
    test_rows.append(merged)

tested_df = pd.DataFrame(test_rows).sort_values(
    ["val_FIT_sim", "test_FIT_sim"],
    ascending=[False, False],
).reset_index(drop=True)

best_by_val = tested_df.iloc[0].to_dict()
best_by_test = tested_df.sort_values(["test_FIT_sim", "val_FIT_sim"], ascending=[False, False]).iloc[0].to_dict()

display(tested_df.round(4))
print("Best by validation:", best_by_val["order"], best_by_val["val_FIT_sim"], best_by_val["test_FIT_sim"])
print("Best by test among tested:", best_by_test["order"], best_by_test["val_FIT_sim"], best_by_test["test_FIT_sim"])
"""
        ),
        md("## 6. So sánh"),
        code(
            """
comparison_rows = []
for label, path in [
    ("ARX Search V1 best", PROJECT_ROOT / "ARX_Model_VersionSearch" / "arx_order_search_v1.json"),
    ("NARX V1", OUT_DIR / "narx_v1.json"),
    ("NARX V2", OUT_DIR / "narx_v2.json"),
    ("NARX V3", OUT_DIR / "narx_v3.json"),
    ("NARX V4", OUT_DIR / "narx_v4.json"),
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
    "model": "NARX V5 HGBR best-by-val",
    "val_FIT_sim": best_by_val["val_FIT_sim"],
    "test_FIT_sim": best_by_val["test_FIT_sim"],
    "test_RMSE_sim": best_by_val["test_RMSE_sim"],
})
comparison_rows.append({
    "model": "NARX V5 HGBR best-tested",
    "val_FIT_sim": best_by_test["val_FIT_sim"],
    "test_FIT_sim": best_by_test["test_FIT_sim"],
    "test_RMSE_sim": best_by_test["test_RMSE_sim"],
})

comparison_df = pd.DataFrame(comparison_rows)
comparison_df["test_gain_vs_arx_search_best"] = comparison_df["test_FIT_sim"] - comparison_df.loc[0, "test_FIT_sim"]
comparison_df.round(4)
"""
        ),
        md("## 7. Lưu artifact"),
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
    "version": "narx_v5_hist_gradient_boosting_order_search",
    "estimator": "HistGradientBoostingRegressor",
    "order_candidates": [list(v) for v in ORDER_CANDIDATES],
    "model_candidates": MODEL_CANDIDATES,
    "input_cols": list(AUGMENTED_INPUT_COLS),
    "normalization": {"method": "zscore", "fit_on": "train", "stats": scale_stats},
    "simulation_clip": {"enabled": True, "bounds_real": list(clip_bounds_real), "bounds_scaled": list(clip_bounds_scaled)},
    "best_by_validation": best_by_val,
    "best_by_test_among_tested": best_by_test,
    "validation_search_results": search_df.to_dict(orient="records"),
    "tested_top_candidates": tested_df.to_dict(orient="records"),
    "comparison": comparison_df.to_dict(orient="records"),
}

OUT_DIR.mkdir(exist_ok=True)
out_path = OUT_DIR / "narx_v5.json"
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
