from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT_DIR = Path("ARX_Model_VersionSearch")
NOTEBOOK_PATH = OUT_DIR / "ARX_Order_Search_V1.ipynb"


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
# ARX Order Search V1

Search này duyệt các order trọng điểm `ARX(na, nb, nk)` với:

- `na = 1..5`
- `nb = 1..5`
- `nk = 1..5`
- 16 augmented features
- z-score normalization theo train
- intercept
- clip free-run theo Q1%-Q99% train
- OLS cho search V1

Mục tiêu là tìm order tốt theo validation `FIT_sim`, sau đó xem test để kiểm tra generalization.
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

WORK_DIR = Path.cwd()
PROJECT_ROOT = WORK_DIR.parent if WORK_DIR.name.startswith("ARX_Model_Version") else WORK_DIR
OUT_DIR = PROJECT_ROOT / "ARX_Model_VersionSearch"

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
    compute_metrics,
    simulate_arx,
    simulate_arx_n_step,
)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 200)

NA_LIST = [1, 2, 3, 4, 5]
NB_LIST = [1, 2, 3, 4, 5]
NK_LIST = [1, 2, 3, 4, 5]
CLIP_QUANTILES = (0.01, 0.99)
N_STEP = 12
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
        md("## 3. Z-score và clip theo train"),
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
clip_bounds_scaled = (
    float((clip_low_real - y_scale["mean"]) / y_scale["std"]),
    float((clip_high_real - y_scale["mean"]) / y_scale["std"]),
)

df_train_z = apply_zscore(df_train)
df_val_z = apply_zscore(df_val)
df_test_z = apply_zscore(df_test)

pd.Series({
    "clip_low_real": clip_low_real,
    "clip_high_real": clip_high_real,
    "clip_low_scaled": clip_bounds_scaled[0],
    "clip_high_scaled": clip_bounds_scaled[1],
})
"""
        ),
        md("## 4. Lightweight order evaluation"),
        code(
            """
def metrics_for_split(df_split_z: pd.DataFrame, theta: np.ndarray, cfg: ModelConfig) -> dict[str, float]:
    x_mat, y_vec = build_regression_matrix(df_split_z, cfg)
    y_pred_1 = x_mat @ theta
    y_pred_sim, y_true_sim = simulate_arx(df_split_z, theta, cfg)
    y_pred_12, _ = simulate_arx_n_step(df_split_z, theta, n_steps=N_STEP, model_config=cfg)

    m1 = compute_metrics(inverse_y(y_vec), inverse_y(y_pred_1), len(theta))
    m12 = compute_metrics(inverse_y(y_true_sim), inverse_y(y_pred_12), len(theta))
    msim = compute_metrics(inverse_y(y_true_sim), inverse_y(y_pred_sim), len(theta))
    return {
        "FIT_1step": m1["FIT"],
        "RMSE_1step": m1["RMSE"],
        "FIT_12step": m12["FIT"],
        "RMSE_12step": m12["RMSE"],
        "FIT_sim": msim["FIT"],
        "RMSE_sim": msim["RMSE"],
        "Bias_sim": msim["Bias"],
    }


def evaluate_order(na: int, nb: int, nk: int) -> dict:
    cfg = ModelConfig(
        na=na,
        nb=nb,
        nk=nk,
        include_intercept=True,
        input_cols=AUGMENTED_INPUT_COLS,
        output_col="Soil_Moisture",
        simulation_clip=clip_bounds_scaled,
    )
    x_train, y_train = build_regression_matrix(df_train_z, cfg)
    theta, _, sigma2 = estimate_ols(x_train, y_train)
    val = metrics_for_split(df_val_z, theta, cfg)
    test = metrics_for_split(df_test_z, theta, cfg)
    return {
        "na": na,
        "nb": nb,
        "nk": nk,
        "order": f"({na},{nb},{nk})",
        "n_params": len(theta),
        "rank_x_train": int(np.linalg.matrix_rank(x_train)),
        "sigma2_train": float(sigma2),
        "val_FIT_1step": val["FIT_1step"],
        "val_FIT_12step": val["FIT_12step"],
        "val_FIT_sim": val["FIT_sim"],
        "val_RMSE_sim": val["RMSE_sim"],
        "test_FIT_1step": test["FIT_1step"],
        "test_FIT_12step": test["FIT_12step"],
        "test_FIT_sim": test["FIT_sim"],
        "test_RMSE_sim": test["RMSE_sim"],
        "test_Bias_sim": test["Bias_sim"],
    }
"""
        ),
        md("## 5. Chạy grid search"),
        code(
            """
start = time.time()
rows = []
errors = []
total = len(NA_LIST) * len(NB_LIST) * len(NK_LIST)
done = 0

for na in NA_LIST:
    for nb in NB_LIST:
        for nk in NK_LIST:
            done += 1
            try:
                rows.append(evaluate_order(na, nb, nk))
            except Exception as exc:
                errors.append({"na": na, "nb": nb, "nk": nk, "error": str(exc)})
            if done % 10 == 0 or done == total:
                print(f"{done}/{total} done")

search_df = pd.DataFrame(rows).sort_values(
    ["val_FIT_sim", "val_FIT_12step", "val_FIT_1step", "n_params"],
    ascending=[False, False, False, True],
).reset_index(drop=True)

elapsed_seconds = time.time() - start
print("Elapsed seconds:", round(elapsed_seconds, 2))
print("Errors:", len(errors))
search_df.round(4)
"""
        ),
        md("## 6. Bảng so sánh với các version hiện có"),
        code(
            """
version_rows = []

version_sources = [
    ("221 V1", PROJECT_ROOT / "ARX_Model_Version 221" / "arx_baseline_v1.json"),
    ("221 V6", PROJECT_ROOT / "ARX_Model_Version 221" / "arx_baseline_v6.json"),
    ("512 V1", PROJECT_ROOT / "ARX_Model_Version 512" / "arx_512_v1.json"),
    ("512 V6", PROJECT_ROOT / "ARX_Model_Version 512" / "arx_512_v6.json"),
]

for label, path in version_sources:
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        artifact = json.load(f)
    version_rows.append({
        "model": label,
        "order": f"({artifact['model_config']['na']},{artifact['model_config']['nb']},{artifact['model_config']['nk']})",
        "n_inputs": len(artifact["model_config"]["input_cols"]),
        "clip": artifact.get("simulation_clip", {}).get("enabled", artifact["model_config"].get("simulation_clip") is not None),
        "val_FIT_sim": artifact["metrics"]["validation"]["fit_sim"],
        "test_FIT_sim": artifact["metrics"]["test"]["fit_sim"],
    })

best = search_df.iloc[0]
version_rows.append({
    "model": "Search V1 best by val",
    "order": best["order"],
    "n_inputs": len(AUGMENTED_INPUT_COLS),
    "clip": True,
    "val_FIT_sim": float(best["val_FIT_sim"]),
    "test_FIT_sim": float(best["test_FIT_sim"]),
})

comparison_df = pd.DataFrame(version_rows)
comparison_df["test_gain_vs_221_v1"] = comparison_df["test_FIT_sim"] - comparison_df.loc[0, "test_FIT_sim"]
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


best = search_df.iloc[0].to_dict()
artifact = {
    "version": "order_search_v1",
    "search_space": {
        "na": NA_LIST,
        "nb": NB_LIST,
        "nk": NK_LIST,
        "n_candidates_requested": len(NA_LIST) * len(NB_LIST) * len(NK_LIST),
        "n_candidates_success": int(len(search_df)),
        "n_errors": int(len(errors)),
        "errors": errors,
    },
    "fixed_config": {
        "input_cols": list(AUGMENTED_INPUT_COLS),
        "include_intercept": True,
        "normalization": "zscore fitted on train",
        "simulation_clip_scaled": list(clip_bounds_scaled),
        "simulation_clip_real": [clip_low_real, clip_high_real],
        "estimator": "OLS",
        "selection_metric": "max validation FIT_sim",
    },
    "best_by_validation": best,
    "top_20": search_df.head(20).to_dict(orient="records"),
    "all_results": search_df.to_dict(orient="records"),
    "comparison_table": comparison_df.to_dict(orient="records"),
    "elapsed_seconds": elapsed_seconds,
}

OUT_DIR.mkdir(exist_ok=True)
out_path = OUT_DIR / "arx_order_search_v1.json"
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
