from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT_DIR = Path("Hybrid_ARX_NARX")
NOTEBOOK_PATH = OUT_DIR / "Hybrid_Residual_V2.ipynb"


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
# Hybrid Residual V2

V1 chỉ dùng ARX backbone `(1,5,2)`. V2 search nhiều backbone ARX trọng điểm rồi train residual correction cho từng backbone:

- Backbone lấy từ top ARX order search.
- Residual models: Ridge, HGBR, ExtraTrees, RandomForest.
- Chọn final bằng validation `FIT_sim`.
"""
        ),
        code(
            """
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WORK_DIR = Path.cwd()
PROJECT_ROOT = WORK_DIR.parent if WORK_DIR.name == "Hybrid_ARX_NARX" else WORK_DIR
OUT_DIR = PROJECT_ROOT / "Hybrid_ARX_NARX"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arx_pipeline import (
    DataConfig,
    SplitConfig,
    ModelConfig,
    load_or_generate_data,
    split_time_series,
    build_regression_matrix,
    estimate_ols,
    simulate_arx,
    compute_metrics,
)
from narx_pipeline import (
    build_augmented_df,
    fit_zscore_stats,
    apply_zscore,
    inverse_zscore_y,
    scaled_clip_bounds,
)

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
SCALE_COLS = ("Soil_Moisture", *AUGMENTED_INPUT_COLS)
SHRINK_CANDIDATES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

BACKBONE_ORDERS = [
    (1, 5, 2),
    (5, 1, 2),
    (2, 5, 2),
    (5, 3, 2),
    (1, 1, 2),
    (1, 3, 2),
    (3, 1, 2),
    (2, 1, 2),
]
"""
        ),
        md("## 1. Data"),
        code(
            """
df_full, _, data_source = load_or_generate_data(
    DataConfig(
        csv_path=PROJECT_ROOT / "greenhouse_data.csv",
        generator_script_path=PROJECT_ROOT / "data_generator.py",
        force_regenerate_from_script=False,
        auto_save_generated_csv=True,
    )
)
df_aug = build_augmented_df(df_full)
df_train, df_val, df_test = split_time_series(df_aug, SplitConfig(train_ratio=0.60, val_ratio=0.20))

scale_stats = fit_zscore_stats(df_train, SCALE_COLS)
clip_bounds_real, clip_bounds_scaled = scaled_clip_bounds(df_train, scale_stats, (0.01, 0.99))
df_train_z = apply_zscore(df_train, scale_stats)
df_val_z = apply_zscore(df_val, scale_stats)
df_test_z = apply_zscore(df_test, scale_stats)
"""
        ),
        md("## 2. Helpers"),
        code(
            """
def build_residual_features(df_z: pd.DataFrame, y_arx_sim: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    lag = max(cfg.na, cfg.nb + cfg.nk - 1)
    y_arx_full = df_z[cfg.output_col].astype(float).to_numpy().copy()
    y_arx_full[lag:] = y_arx_sim
    rows = []
    for idx, t in enumerate(range(lag, len(df_z))):
        row = [float(y_arx_sim[idx])]
        for y_lag in range(1, 4):
            if t - y_lag >= 0:
                row.append(float(y_arx_full[t - y_lag]))
            else:
                row.append(float(y_arx_full[0]))
        for col in AUGMENTED_INPUT_COLS:
            values = df_z[col].astype(float).to_numpy()
            row.append(float(values[t]))
            row.append(float(values[max(0, t - 1)]))
            row.append(float(values[max(0, t - 2)]))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def metric_real(y_true_z: np.ndarray, y_pred_z: np.ndarray, n_params: int = 0) -> dict[str, float]:
    return compute_metrics(
        inverse_zscore_y(y_true_z, scale_stats),
        inverse_zscore_y(y_pred_z, scale_stats),
        n_params,
    )


def evaluate_hybrid(y_true_z: np.ndarray, y_arx_z: np.ndarray, correction_z: np.ndarray, shrink: float) -> dict[str, float]:
    y_hybrid_z = y_arx_z + shrink * correction_z
    y_hybrid_z = np.clip(y_hybrid_z, clip_bounds_scaled[0], clip_bounds_scaled[1])
    return metric_real(y_true_z, y_hybrid_z)


def residual_models(seed: int):
    return [
        ("ridge_1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("hgb_31", HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        )),
        ("extra_trees_leaf5", ExtraTreesRegressor(
            n_estimators=120,
            max_features=0.7,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed + 1,
        )),
        ("random_forest_leaf5", RandomForestRegressor(
            n_estimators=120,
            max_features=0.7,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed + 2,
        )),
    ]
"""
        ),
        md("## 3. Backbone + residual search"),
        code(
            """
rows = []
backbone_rows = []
candidate_id = 0

for backbone_idx, (na, nb, nk) in enumerate(BACKBONE_ORDERS, start=1):
    cfg = ModelConfig(
        na=na,
        nb=nb,
        nk=nk,
        include_intercept=True,
        input_cols=AUGMENTED_INPUT_COLS,
        simulation_clip=clip_bounds_scaled,
    )
    x_arx_train, y_arx_train_target = build_regression_matrix(df_train_z, cfg)
    theta_arx, _, sigma2_arx = estimate_ols(x_arx_train, y_arx_train_target)

    y_arx_train, y_true_train = simulate_arx(df_train_z, theta_arx, cfg)
    y_arx_val, y_true_val = simulate_arx(df_val_z, theta_arx, cfg)
    y_arx_test, y_true_test = simulate_arx(df_test_z, theta_arx, cfg)

    arx_val = metric_real(y_true_val, y_arx_val, x_arx_train.shape[1])
    arx_test = metric_real(y_true_test, y_arx_test, x_arx_train.shape[1])
    order_label = f"({na},{nb},{nk})"
    backbone_rows.append({
        "order": order_label,
        "na": na,
        "nb": nb,
        "nk": nk,
        "arx_val_FIT_sim": arx_val["FIT"],
        "arx_test_FIT_sim": arx_test["FIT"],
        "arx_test_RMSE_sim": arx_test["RMSE"],
    })

    x_res_train = build_residual_features(df_train_z, y_arx_train, cfg)
    x_res_val = build_residual_features(df_val_z, y_arx_val, cfg)
    x_res_test = build_residual_features(df_test_z, y_arx_test, cfg)
    y_res_train = y_true_train - y_arx_train

    for model_name, model in residual_models(seed=100 + backbone_idx * 10):
        candidate_id += 1
        start = time.time()
        model.fit(x_res_train, y_res_train)
        train_seconds = time.time() - start
        corr_val = model.predict(x_res_val)
        corr_test = model.predict(x_res_test)

        for shrink in SHRINK_CANDIDATES:
            val_hybrid = evaluate_hybrid(y_true_val, y_arx_val, corr_val, shrink)
            test_hybrid = evaluate_hybrid(y_true_test, y_arx_test, corr_test, shrink)
            rows.append({
                "candidate_id": candidate_id,
                "order": order_label,
                "na": na,
                "nb": nb,
                "nk": nk,
                "residual_model": model_name,
                "shrink": shrink,
                "train_seconds": train_seconds,
                "arx_val_FIT_sim": arx_val["FIT"],
                "arx_test_FIT_sim": arx_test["FIT"],
                "val_FIT_sim": val_hybrid["FIT"],
                "val_RMSE_sim": val_hybrid["RMSE"],
                "test_FIT_sim": test_hybrid["FIT"],
                "test_RMSE_sim": test_hybrid["RMSE"],
                "test_Bias_sim": test_hybrid["Bias"],
            })
    print(f"done backbone {order_label}")

backbone_df = pd.DataFrame(backbone_rows).sort_values("arx_val_FIT_sim", ascending=False).reset_index(drop=True)
search_df = pd.DataFrame(rows).sort_values(["val_FIT_sim", "test_FIT_sim"], ascending=[False, False]).reset_index(drop=True)
best_by_validation = search_df.iloc[0].to_dict()

display(backbone_df.round(4))
search_df.head(25).round(4)
"""
        ),
        md("## 4. Comparison"),
        code(
            """
comparison_rows = []

arx_best_path = PROJECT_ROOT / "ARX_Model_VersionSearch" / "arx_order_search_v1.json"
if arx_best_path.exists():
    with arx_best_path.open("r", encoding="utf-8") as f:
        arx_search = json.load(f)["best_by_validation"]
    arx_best_test = arx_search["test_FIT_sim"]
    comparison_rows.append({
        "model": "ARX Search V1 best",
        "val_FIT_sim": arx_search["val_FIT_sim"],
        "test_FIT_sim": arx_search["test_FIT_sim"],
        "test_RMSE_sim": arx_search["test_RMSE_sim"],
        "test_gain_vs_arx": 0.0,
    })
else:
    arx_best_test = float(backbone_df.iloc[0]["arx_test_FIT_sim"])

v1_path = OUT_DIR / "hybrid_residual_v1.json"
if v1_path.exists():
    with v1_path.open("r", encoding="utf-8") as f:
        v1 = json.load(f)["best_by_validation"]
    comparison_rows.append({
        "model": "Hybrid Residual V1",
        "val_FIT_sim": v1["val_FIT_sim"],
        "test_FIT_sim": v1["test_FIT_sim"],
        "test_RMSE_sim": v1["test_RMSE_sim"],
        "test_gain_vs_arx": v1["test_FIT_sim"] - arx_best_test,
    })

comparison_rows.append({
    "model": "Hybrid Residual V2 best-by-val",
    "val_FIT_sim": best_by_validation["val_FIT_sim"],
    "test_FIT_sim": best_by_validation["test_FIT_sim"],
    "test_RMSE_sim": best_by_validation["test_RMSE_sim"],
    "test_gain_vs_arx": best_by_validation["test_FIT_sim"] - arx_best_test,
})

narx_v6_path = PROJECT_ROOT / "NARX" / "narx_v6.json"
if narx_v6_path.exists():
    with narx_v6_path.open("r", encoding="utf-8") as f:
        narx_v6 = json.load(f)["selected_candidate"]
    comparison_rows.append({
        "model": "NARX V6 selected",
        "val_FIT_sim": narx_v6["val_FIT_sim"],
        "test_FIT_sim": narx_v6["test_FIT_sim"],
        "test_RMSE_sim": narx_v6["test_RMSE_sim"],
        "test_gain_vs_arx": narx_v6["test_FIT_sim"] - arx_best_test,
    })

comparison_df = pd.DataFrame(comparison_rows).sort_values("test_FIT_sim", ascending=False).reset_index(drop=True)
comparison_df.round(4)
"""
        ),
        md("## 5. Save"),
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


def df_to_markdown(df: pd.DataFrame) -> str:
    df_str = df.astype(str)
    headers = list(df_str.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df_str.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\\n".join(lines) + "\\n"


artifact = {
    "model_type": "Hybrid_ARX_Residual_NARX",
    "version": "hybrid_residual_v2_backbone_search",
    "backbone_orders": [list(v) for v in BACKBONE_ORDERS],
    "selection_metric": "validation FIT_sim",
    "best_by_validation": best_by_validation,
    "backbone_results": backbone_df.to_dict(orient="records"),
    "search_results": search_df.to_dict(orient="records"),
    "comparison": comparison_df.to_dict(orient="records"),
}

OUT_DIR.mkdir(exist_ok=True)
json_path = OUT_DIR / "hybrid_residual_v2.json"
search_csv_path = OUT_DIR / "hybrid_residual_v2_search.csv"
comparison_csv_path = OUT_DIR / "hybrid_residual_v2_comparison.csv"
comparison_md_path = OUT_DIR / "hybrid_residual_v2_comparison.md"
backbone_csv_path = OUT_DIR / "hybrid_residual_v2_backbones.csv"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(json_ready(artifact), f, indent=2)
    f.write("\\n")

search_df.to_csv(search_csv_path, index=False)
backbone_df.to_csv(backbone_csv_path, index=False)
comparison_df.to_csv(comparison_csv_path, index=False)
comparison_md_path.write_text(df_to_markdown(comparison_df.round(4)), encoding="utf-8")

json_path, search_csv_path, comparison_md_path
"""
        ),
    ]

    OUT_DIR.mkdir(exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH.resolve()}")


if __name__ == "__main__":
    main()
