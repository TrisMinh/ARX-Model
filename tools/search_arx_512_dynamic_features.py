from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arx_pipeline import (  # noqa: E402
    DataConfig,
    SplitConfig,
    ModelConfig,
    build_regression_matrix,
    compute_metrics,
    estimate_ols,
    load_or_generate_data,
    split_time_series,
)


OUT_DIR = PROJECT_ROOT / "ARX_Model_Version 512"
BASELINE_INPUT_COLS = ("Temperature", "Humidity", "Light", "Drip", "Mist", "Fan")
CORE_ENGINEERED = (
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
ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
CLIP_QUANTILES = (0.01, 0.99)


def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["Light_log"] = np.log1p(df["Light"].clip(lower=0))
    df["Temp_x_Humi"] = df["Temperature"] * df["Humidity"]
    df["Temp_x_Light"] = df["Temperature"] * df["Light_log"]
    df["Humi_x_Light"] = df["Humidity"] * df["Light_log"]
    df["SP_Center"] = 0.5 * (df["Soil_Low_SP"] + df["Soil_High_SP"])
    df["SP_Width"] = df["Soil_High_SP"] - df["Soil_Low_SP"]
    df["Soil_Deficit"] = df["SP_Center"] - df["Soil_Moisture"]
    df["Abs_Soil_Deficit"] = np.abs(df["Soil_Deficit"])
    df["Dry_Below_SP"] = np.maximum(df["Soil_Low_SP"] - df["Soil_Moisture"], 0.0)

    # Water memory features. With ARX lags these are used from past timestamps.
    for hours, window in [("1h", 12), ("3h", 36), ("6h", 72), ("12h", 144)]:
        df[f"Drip_sum_{hours}"] = df["Drip"].rolling(window, min_periods=1).sum()
        df[f"Mist_sum_{hours}"] = df["Mist"].rolling(window, min_periods=1).sum()
    df["Drip_ewm_fast"] = df["Drip"].ewm(alpha=0.25, adjust=False).mean()
    df["Drip_ewm_slow"] = df["Drip"].ewm(alpha=0.05, adjust=False).mean()
    df["Mist_ewm"] = df["Mist"].ewm(alpha=0.12, adjust=False).mean()

    last_drip = -10_000
    since = np.zeros(len(df), dtype=float)
    for i, value in enumerate(df["Drip"].to_numpy(dtype=float)):
        if value > 0.5:
            last_drip = i
            since[i] = 0.0
        else:
            since[i] = min(i - last_drip, 288.0)
    df["Time_since_drip"] = since

    df["Drip_x_Deficit"] = df["Drip"] * df["Soil_Deficit"]
    df["DripMemory_x_Deficit"] = df["Drip_ewm_slow"] * df["Soil_Deficit"]
    df["DryingDemand"] = df["Light_log"] * df["Temperature"] * (100.0 - df["Humidity"]) / 100.0
    df["Light_x_Dryness"] = df["Light_log"] * (100.0 - df["Humidity"]) / 100.0
    df["Fan_x_DryingDemand"] = df["Fan"] * df["DryingDemand"]

    df["Hour"] = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0
    df["Hour_sin"] = np.sin(2.0 * np.pi * df["Hour"] / 24.0)
    df["Hour_cos"] = np.cos(2.0 * np.pi * df["Hour"] / 24.0)
    df["Month_sin"] = np.sin(2.0 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2.0 * np.pi * df["Month"] / 12.0)
    season_map = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
    season_num = df["Season"].map(season_map).fillna(0).astype(float)
    df["Season_sin"] = np.sin(2.0 * np.pi * season_num / 4.0)
    df["Season_cos"] = np.cos(2.0 * np.pi * season_num / 4.0)
    return df


def fit_stats(df_train: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, dict[str, float]]:
    stats = {}
    for col in cols:
        mean = float(df_train[col].astype(float).mean())
        std = float(df_train[col].astype(float).std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_zscore(df: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, st in stats.items():
        out[col] = (out[col].astype(float) - st["mean"]) / st["std"]
    return out


def inverse_y(values: np.ndarray, stats: dict[str, dict[str, float]]) -> np.ndarray:
    st = stats["Soil_Moisture"]
    return np.asarray(values, dtype=float) * st["std"] + st["mean"]


def estimate_ridge(x: np.ndarray, y: np.ndarray, alpha: float, include_intercept: bool) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=float)
    if include_intercept:
        penalty[-1, -1] = 0.0
    return np.linalg.pinv(x.T @ x + alpha * penalty) @ x.T @ y


Y_DERIVED_FEATURES = {
    "Soil_Deficit",
    "Abs_Soil_Deficit",
    "Dry_Below_SP",
    "Drip_x_Deficit",
    "DripMemory_x_Deficit",
}


def _dynamic_input_value(arrays: dict[str, np.ndarray], y_state: np.ndarray, col: str, idx: int) -> float:
    if col == "Soil_Deficit":
        return float(arrays["SP_Center"][idx] - y_state[idx])
    if col == "Abs_Soil_Deficit":
        return float(abs(arrays["SP_Center"][idx] - y_state[idx]))
    if col == "Dry_Below_SP":
        return float(max(arrays["Soil_Low_SP"][idx] - y_state[idx], 0.0))
    if col == "Drip_x_Deficit":
        deficit = float(arrays["SP_Center"][idx] - y_state[idx])
        return float(arrays["Drip"][idx] * deficit)
    if col == "DripMemory_x_Deficit":
        deficit = float(arrays["SP_Center"][idx] - y_state[idx])
        return float(arrays["Drip_ewm_slow"][idx] * deficit)
    return float(arrays[col][idx])


def simulate_arx_no_leak(df_sim: pd.DataFrame, theta: np.ndarray, config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Free-run simulation that recomputes y-derived engineered inputs from y_sim."""
    y = df_sim[config.output_col].astype(float).to_numpy().copy()
    y_sim = y.copy()
    max_lag = max(config.na, config.nb + config.nk - 1)
    fixed_inputs = {col: df_sim[col].astype(float).to_numpy() for col in config.input_cols if col not in Y_DERIVED_FEATURES}
    arrays = {
        "SP_Center": df_sim["SP_Center"].astype(float).to_numpy(),
        "Soil_Low_SP": df_sim["Soil_Low_SP"].astype(float).to_numpy(),
        "Drip": df_sim["Drip"].astype(float).to_numpy(),
        "Drip_ewm_slow": df_sim["Drip_ewm_slow"].astype(float).to_numpy() if "Drip_ewm_slow" in df_sim else np.zeros(len(df_sim)),
    }

    for t in range(max_lag, len(y)):
        row: list[float] = []
        for lag in range(1, config.na + 1):
            row.append(float(y_sim[t - lag]))
        for col in config.input_cols:
            arr = fixed_inputs.get(col)
            for lag in range(config.nk, config.nk + config.nb):
                idx = t - lag
                if arr is None:
                    row.append(_dynamic_input_value(arrays, y_sim, col, idx))
                else:
                    row.append(float(arr[idx]))
        if config.include_intercept:
            row.append(1.0)
        y_next = float(np.dot(row, theta))
        if config.simulation_clip is not None:
            y_next = float(np.clip(y_next, config.simulation_clip[0], config.simulation_clip[1]))
        y_sim[t] = y_next
    return y_sim[max_lag:], y[max_lag:]


def score_config(
    label: str,
    input_cols: tuple[str, ...],
    na: int,
    nb: int,
    nk: int,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame | None = None,
) -> dict:
    scale_cols = ("Soil_Moisture", *input_cols)
    stats = fit_stats(df_train, scale_cols)
    train_z = apply_zscore(df_train, stats)
    val_z = apply_zscore(df_val, stats)
    test_z = apply_zscore(df_test, stats) if df_test is not None else None

    y_scale = stats["Soil_Moisture"]
    lo = float(df_train["Soil_Moisture"].quantile(CLIP_QUANTILES[0]))
    hi = float(df_train["Soil_Moisture"].quantile(CLIP_QUANTILES[1]))
    clip = ((lo - y_scale["mean"]) / y_scale["std"], (hi - y_scale["mean"]) / y_scale["std"])
    config = ModelConfig(
        na=na,
        nb=nb,
        nk=nk,
        include_intercept=True,
        input_cols=input_cols,
        output_col="Soil_Moisture",
        simulation_clip=(float(clip[0]), float(clip[1])),
    )

    x_train, y_train = build_regression_matrix(train_z, config)
    theta_ols, _, _ = estimate_ols(x_train, y_train)

    def fast_eval(df_z: pd.DataFrame, theta: np.ndarray) -> dict[str, float]:
        x_mat, y_vec = build_regression_matrix(df_z, config)
        y_pred_1 = x_mat @ theta
        y_pred_sim, y_true_sim = simulate_arx_no_leak(df_z, theta, config)
        m1 = compute_metrics(inverse_y(y_vec, stats), inverse_y(y_pred_1, stats), len(theta))
        msim = compute_metrics(inverse_y(y_true_sim, stats), inverse_y(y_pred_sim, stats), len(theta))
        return {
            "FIT_1step": m1["FIT"],
            "FIT_sim": msim["FIT"],
            "RMSE_sim": msim["RMSE"],
            "MAE_sim": msim["MAE"],
            "Bias_sim": msim["Bias"],
        }

    best = None
    for alpha in ALPHAS:
        theta = theta_ols if alpha == 0.0 else estimate_ridge(x_train, y_train, alpha, True)
        val_metrics = fast_eval(val_z, theta)
        row = {
            "label": label,
            "na": na,
            "nb": nb,
            "nk": nk,
            "n_inputs": len(input_cols),
            "n_params": len(config.param_names),
            "alpha": float(alpha),
            "val_FIT_1step": val_metrics["FIT_1step"],
            "val_FIT_sim": val_metrics["FIT_sim"],
            "val_RMSE_sim": val_metrics["RMSE_sim"],
            "condition_number_xtx": float(np.linalg.cond(x_train.T @ x_train)),
        }
        if best is None or row["val_FIT_sim"] > best["val_FIT_sim"]:
            best = row
            best["_theta"] = theta
            best["_config"] = config
            best["_stats"] = stats
            best["_test_z"] = test_z
    assert best is not None
    if test_z is not None:
        test_metrics = fast_eval(test_z, best["_theta"])
        best.update(
            {
                "test_FIT_1step": test_metrics["FIT_1step"],
                "test_FIT_sim": test_metrics["FIT_sim"],
                "test_RMSE_sim": test_metrics["RMSE_sim"],
                "test_MAE_sim": test_metrics["MAE_sim"],
                "test_Bias_sim": test_metrics["Bias_sim"],
            }
        )
    return {k: v for k, v in best.items() if not k.startswith("_")}


def main() -> None:
    data_config = DataConfig(
        csv_path=PROJECT_ROOT / "greenhouse_data.csv",
        generator_script_path=PROJECT_ROOT / "data_generator.py",
        force_regenerate_from_script=False,
        auto_save_generated_csv=True,
    )
    split_config = SplitConfig(train_ratio=0.75, val_ratio=0.15)
    df_raw, _, source = load_or_generate_data(data_config)
    df = build_features(df_raw)
    df_train, df_val, df_test = split_time_series(df, split_config)
    print(f"Data: {source}; rows train/val/test = {len(df_train)}/{len(df_val)}/{len(df_test)}")

    water_memory = (
        "Drip_sum_1h",
        "Drip_sum_3h",
        "Drip_sum_6h",
        "Drip_sum_12h",
        "Drip_ewm_fast",
        "Drip_ewm_slow",
        "Time_since_drip",
    )
    deficit = ("Soil_Deficit", "Abs_Soil_Deficit", "Dry_Below_SP", "Drip_x_Deficit", "DripMemory_x_Deficit")
    drying = ("DryingDemand", "Light_x_Dryness", "Fan_x_DryingDemand", "Hour_sin", "Hour_cos")

    feature_sets = {
        "V6_core": tuple(BASELINE_INPUT_COLS + CORE_ENGINEERED),
        "Core_plus_deficit": tuple(BASELINE_INPUT_COLS + CORE_ENGINEERED + deficit),
        "Core_plus_water_deficit": tuple(BASELINE_INPUT_COLS + CORE_ENGINEERED + water_memory + deficit),
        "SP_water_deficit_minimal": ("Temperature", "Humidity", "Light", "Drip", "Mist", "Fan", "SP_Center", *water_memory, *deficit),
    }
    orders = [
        (5, 1, 2),
        (5, 3, 1),
        (5, 5, 1),
        (8, 3, 1),
    ]

    rows = []
    for label, cols in feature_sets.items():
        for na, nb, nk in orders:
            print(f"Scoring {label} ARX({na},{nb},{nk}) ...", flush=True)
            try:
                rows.append(score_config(label, cols, na, nb, nk, df_train, df_val, None))
            except Exception as exc:
                rows.append({"label": label, "na": na, "nb": nb, "nk": nk, "error": repr(exc)})
    val_df = pd.DataFrame(rows).sort_values("val_FIT_sim", ascending=False)
    val_path = OUT_DIR / "arx_512_dynamic_feature_search_validation_noleak.csv"
    val_df.to_csv(val_path, index=False)
    print(f"Wrote {val_path}")

    top_rows = val_df.dropna(subset=["val_FIT_sim"]).head(12)
    final_rows = []
    for _, row in top_rows.iterrows():
        label = str(row["label"])
        cols = feature_sets[label]
        na, nb, nk = int(row["na"]), int(row["nb"]), int(row["nk"])
        print(f"Testing {label} ARX({na},{nb},{nk}) ...", flush=True)
        final_rows.append(score_config(label, cols, na, nb, nk, df_train, df_val, df_test))
    final_df = pd.DataFrame(final_rows).sort_values(["test_FIT_sim", "val_FIT_sim"], ascending=False)
    final_path = OUT_DIR / "arx_512_dynamic_feature_search_test_top_noleak.csv"
    final_df.to_csv(final_path, index=False)
    print(f"Wrote {final_path}")
    print(final_df[["label", "na", "nb", "nk", "alpha", "val_FIT_sim", "test_FIT_sim", "test_RMSE_sim"]].head(12).round(4))


if __name__ == "__main__":
    main()
