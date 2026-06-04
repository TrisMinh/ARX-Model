from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

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
    simulate_arx,
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


def score(df_z: pd.DataFrame, theta: np.ndarray, config: ModelConfig, stats: dict[str, dict[str, float]]) -> dict:
    x, y = build_regression_matrix(df_z, config)
    pred_1 = x @ theta
    pred_sim, true_sim = simulate_arx(df_z, theta, config)
    m1 = compute_metrics(inverse_y(y, stats), inverse_y(pred_1, stats), len(theta))
    msim = compute_metrics(inverse_y(true_sim, stats), inverse_y(pred_sim, stats), len(theta))
    return {
        "FIT_1step": m1["FIT"],
        "RMSE_1step": m1["RMSE"],
        "FIT_sim": msim["FIT"],
        "RMSE_sim": msim["RMSE"],
        "MAE_sim": msim["MAE"],
        "Bias_sim": msim["Bias"],
    }


def main() -> None:
    data_config = DataConfig(
        csv_path=PROJECT_ROOT / "greenhouse_data.csv",
        generator_script_path=PROJECT_ROOT / "data_generator.py",
        force_regenerate_from_script=False,
        auto_save_generated_csv=True,
    )
    df_raw, _, _ = load_or_generate_data(data_config)
    df = build_augmented_df(df_raw)
    df_train, df_val, df_test = split_time_series(df, SplitConfig(train_ratio=0.75, val_ratio=0.15))
    input_cols = tuple(BASELINE_INPUT_COLS + CORE_ENGINEERED)
    stats = fit_stats(df_train, ("Soil_Moisture", *input_cols))
    train_z = apply_zscore(df_train, stats)
    val_z = apply_zscore(df_val, stats)
    test_z = apply_zscore(df_test, stats)
    y_scale = stats["Soil_Moisture"]
    lo = float(df_train["Soil_Moisture"].quantile(0.01))
    hi = float(df_train["Soil_Moisture"].quantile(0.99))
    clip = ((lo - y_scale["mean"]) / y_scale["std"], (hi - y_scale["mean"]) / y_scale["std"])
    config = ModelConfig(na=5, nb=1, nk=2, include_intercept=True, input_cols=input_cols, output_col="Soil_Moisture", simulation_clip=clip)

    x_train, y_train = build_regression_matrix(train_z, config)
    theta_ols, _, _ = estimate_ols(x_train, y_train)
    max_lag = max(config.na, config.nb + config.nk - 1)
    val_y = val_z["Soil_Moisture"].astype(float).to_numpy()[max_lag:]

    rows = []
    rows.append({"method": "OLS", "lambda": np.nan, "nfev": 0, **{f"val_{k}": v for k, v in score(val_z, theta_ols, config, stats).items()}, **{f"test_{k}": v for k, v in score(test_z, theta_ols, config, stats).items()}})

    for lam in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        print(f"Optimizing simulation error lambda={lam}...", flush=True)

        def residual(theta: np.ndarray) -> np.ndarray:
            pred, _ = simulate_arx(val_z, theta, config)
            err = pred - val_y
            if lam > 0:
                err = np.r_[err, np.sqrt(lam) * (theta - theta_ols)]
            return err

        res = least_squares(
            residual,
            theta_ols,
            method="trf",
            max_nfev=80,
            x_scale="jac",
            loss="soft_l1",
            f_scale=0.5,
            verbose=0,
        )
        theta = res.x
        val_metrics = score(val_z, theta, config, stats)
        test_metrics = score(test_z, theta, config, stats)
        rows.append({
            "method": "simulation_error_refine",
            "lambda": lam,
            "nfev": res.nfev,
            "cost": float(res.cost),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })

    out = pd.DataFrame(rows).sort_values(["test_FIT_sim", "val_FIT_sim"], ascending=False)
    path = OUT_DIR / "arx_512_simulation_error_refine.csv"
    out.to_csv(path, index=False)
    print(out[["method", "lambda", "nfev", "val_FIT_sim", "test_FIT_sim", "test_RMSE_sim"]].round(4))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
