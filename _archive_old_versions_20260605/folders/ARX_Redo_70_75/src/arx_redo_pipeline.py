from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = OUT_DIR / "results"


RAW_INPUT_COLS = (
    "Temperature",
    "Humidity",
    "Light",
    "Drip",
    "Mist",
    "Fan",
)

AUGMENTED_INPUT_COLS = (
    *RAW_INPUT_COLS,
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


@dataclass(frozen=True)
class PipelineConfig:
    csv_path: str = str(PROJECT_ROOT / "greenhouse_data.csv")
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    na: int = 5
    nb: int = 3
    nk: int = 2
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    residual_y_lags: tuple[int, ...] = (1, 2, 3, 6, 12)
    residual_input_lags: tuple[int, ...] = (2, 3, 6, 12, 24)
    shrink_grid: tuple[float, ...] = (0.8, 0.9, 1.0)
    random_state: int = 906


def json_ready(value: Any) -> Any:
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


def add_augmented_features(df_in: pd.DataFrame) -> pd.DataFrame:
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


def split_time(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test_start = n_train + n_val
    return (
        df.iloc[:n_train].reset_index(drop=True),
        df.iloc[n_train:n_test_start].reset_index(drop=True),
        df.iloc[n_test_start:].reset_index(drop=True),
    )


def fit_scale_stats(df_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in SCALE_COLS:
        mean = float(df_train[col].mean())
        std = float(df_train[col].std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        stats[col] = (mean, std)
    return stats


def apply_scale(df_in: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    df = df_in.copy()
    for col, (mean, std) in stats.items():
        df[col] = (df[col].astype(float) - mean) / std
    return df


def inverse_y(values_z: np.ndarray, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    mean, std = stats["Soil_Moisture"]
    return np.asarray(values_z, dtype=float) * std + mean


def max_lag(na: int, nb: int, nk: int) -> int:
    return max(na, nb + nk - 1)


def build_arx_matrix(df: pd.DataFrame, na: int, nb: int, nk: int) -> tuple[np.ndarray, np.ndarray]:
    y = df["Soil_Moisture"].to_numpy(dtype=float)
    lag = max_lag(na, nb, nk)
    cols: list[np.ndarray] = []

    for y_lag in range(1, na + 1):
        cols.append(y[lag - y_lag : len(y) - y_lag])

    for col in AUGMENTED_INPUT_COLS:
        u = df[col].to_numpy(dtype=float)
        for u_lag in range(nk, nk + nb):
            cols.append(u[lag - u_lag : len(u) - u_lag])

    cols.append(np.ones(len(y) - lag))
    return np.vstack(cols).T, y[lag:]


def fit_arx_ols(df_train_z: pd.DataFrame, cfg: PipelineConfig) -> np.ndarray:
    x_train, y_train = build_arx_matrix(df_train_z, cfg.na, cfg.nb, cfg.nk)
    theta, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return theta


def simulate_arx(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    cfg: PipelineConfig,
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    inputs = [df_z[col].to_numpy(dtype=float) for col in AUGMENTED_INPUT_COLS]
    lag = max_lag(cfg.na, cfg.nb, cfg.nk)
    y_sim = y.copy()

    for t in range(lag, len(y)):
        idx = 0
        y_next = 0.0
        for y_lag in range(1, cfg.na + 1):
            y_next += theta[idx] * y_sim[t - y_lag]
            idx += 1
        for u in inputs:
            for u_lag in range(cfg.nk, cfg.nk + cfg.nb):
                y_next += theta[idx] * u[t - u_lag]
                idx += 1
        y_next += theta[idx]
        y_sim[t] = float(np.clip(y_next, clip_bounds[0], clip_bounds[1]))

    return y_sim[lag:], y[lag:]


def residual_features(
    df_z: pd.DataFrame,
    y_arx_sim: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    lag = max_lag(cfg.na, cfg.nb, cfg.nk)
    y_full = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_full[lag:] = y_arx_sim
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in AUGMENTED_INPUT_COLS]

    n_rows = len(df_z) - lag
    n_cols = 1 + len(cfg.residual_y_lags) + len(AUGMENTED_INPUT_COLS) * len(cfg.residual_input_lags)
    x_res = np.empty((n_rows, n_cols), dtype=float)

    for row_idx, t in enumerate(range(lag, len(df_z))):
        row: list[float] = [float(y_arx_sim[row_idx])]
        row.extend(float(y_full[max(0, t - y_lag)]) for y_lag in cfg.residual_y_lags)
        for values in input_arrays:
            row.extend(float(values[max(0, t - u_lag)]) for u_lag in cfg.residual_input_lags)
        x_res[row_idx] = row

    return x_res


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    fit = 100.0 * (1.0 - np.linalg.norm(resid) / np.linalg.norm(y_true - y_true.mean()))
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y_true - y_true.mean(), y_true - y_true.mean()))
    r2 = float(1.0 - ss_res / ss_tot)
    return {
        "FIT": float(fit),
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "R2": r2,
    }


def evaluate_real(y_true_z: np.ndarray, y_pred_z: np.ndarray, stats: dict[str, tuple[float, float]]) -> dict[str, float]:
    return compute_metrics(inverse_y(y_true_z, stats), inverse_y(y_pred_z, stats))


def slice_info(name: str, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "name": name,
        "rows": int(len(df)),
        "start": str(df["Timestamp"].iloc[0]),
        "end": str(df["Timestamp"].iloc[-1]),
        "soil_mean": float(df["Soil_Moisture"].mean()),
        "soil_std": float(df["Soil_Moisture"].std(ddof=0)),
        "drip_on_pct": float(100.0 * df["Drip"].mean()),
        "mist_on_pct": float(100.0 * df["Mist"].mean()),
        "fan_on_pct": float(100.0 * df["Fan"].mean()),
    }


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    df = pd.read_csv(cfg.csv_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df_aug = add_augmented_features(df)
    df_train, df_val, df_test = split_time(df_aug, cfg.train_ratio, cfg.val_ratio)

    scale_stats = fit_scale_stats(df_train)
    df_train_z = apply_scale(df_train, scale_stats)
    df_val_z = apply_scale(df_val, scale_stats)
    df_test_z = apply_scale(df_test, scale_stats)

    clip_bounds = tuple(float(v) for v in np.quantile(df_train_z["Soil_Moisture"], cfg.clip_quantiles))
    theta_arx = fit_arx_ols(df_train_z, cfg)

    y_arx_train, y_true_train = simulate_arx(df_train_z, theta_arx, cfg, clip_bounds)
    y_arx_val, y_true_val = simulate_arx(df_val_z, theta_arx, cfg, clip_bounds)
    y_arx_test, y_true_test = simulate_arx(df_test_z, theta_arx, cfg, clip_bounds)

    x_res_train = residual_features(df_train_z, y_arx_train, cfg)
    y_res_train = y_true_train - y_arx_train

    residual_model = HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=63,
        l2_regularization=0.01,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=cfg.random_state,
    )
    residual_model.fit(x_res_train, y_res_train)

    corr_val = residual_model.predict(residual_features(df_val_z, y_arx_val, cfg))
    corr_test = residual_model.predict(residual_features(df_test_z, y_arx_test, cfg))

    leaderboard: list[dict[str, Any]] = []
    for shrink in cfg.shrink_grid:
        y_hybrid_val = np.clip(y_arx_val + shrink * corr_val, clip_bounds[0], clip_bounds[1])
        y_hybrid_test = np.clip(y_arx_test + shrink * corr_test, clip_bounds[0], clip_bounds[1])
        leaderboard.append(
            {
                "model": "Conservative Hybrid ARX Residual",
                "shrink": float(shrink),
                "val_FIT_sim": evaluate_real(y_true_val, y_hybrid_val, scale_stats)["FIT"],
                "val_RMSE_sim": evaluate_real(y_true_val, y_hybrid_val, scale_stats)["RMSE"],
                "test_FIT_sim": evaluate_real(y_true_test, y_hybrid_test, scale_stats)["FIT"],
                "test_RMSE_sim": evaluate_real(y_true_test, y_hybrid_test, scale_stats)["RMSE"],
                "test_Bias_sim": evaluate_real(y_true_test, y_hybrid_test, scale_stats)["Bias"],
            }
        )

    best = sorted(leaderboard, key=lambda row: row["val_FIT_sim"], reverse=True)[0]
    best_shrink = float(best["shrink"])
    y_hybrid_val_best = np.clip(y_arx_val + best_shrink * corr_val, clip_bounds[0], clip_bounds[1])
    y_hybrid_test_best = np.clip(y_arx_test + best_shrink * corr_test, clip_bounds[0], clip_bounds[1])

    results = {
        "config": asdict(cfg),
        "data": {
            "rows": int(len(df_aug)),
            "start": str(df_aug["Timestamp"].iloc[0]),
            "end": str(df_aug["Timestamp"].iloc[-1]),
            "slices": {
                "train": slice_info("train", df_train),
                "validation": slice_info("validation", df_val),
                "test": slice_info("test", df_test),
            },
        },
        "model": {
            "backbone": "ARX",
            "order": {"na": cfg.na, "nb": cfg.nb, "nk": cfg.nk},
            "input_cols": list(AUGMENTED_INPUT_COLS),
            "residual_model": "HistGradientBoostingRegressor",
            "residual_policy": {
                "uses_true_future_soil_moisture": False,
                "residual_y_lags_source": "ARX simulated trajectory only after initial conditions",
                "input_lags": list(cfg.residual_input_lags),
                "note": "Input lags start at t-2 to keep the residual conservative and aligned with ARX nk=2.",
            },
            "clip_bounds_scaled": list(clip_bounds),
            "selected_by": "maximum validation FIT_sim over shrink_grid",
            "selected_shrink": best_shrink,
        },
        "metrics": {
            "arx": {
                "train": evaluate_real(y_true_train, y_arx_train, scale_stats),
                "validation": evaluate_real(y_true_val, y_arx_val, scale_stats),
                "test": evaluate_real(y_true_test, y_arx_test, scale_stats),
            },
            "hybrid_selected": {
                "validation": evaluate_real(y_true_val, y_hybrid_val_best, scale_stats),
                "test": evaluate_real(y_true_test, y_hybrid_test_best, scale_stats),
            },
        },
        "leaderboard": leaderboard,
    }

    lag = max_lag(cfg.na, cfg.nb, cfg.nk)
    test_predictions = pd.DataFrame(
        {
            "Timestamp": df_test["Timestamp"].iloc[lag:].to_numpy(),
            "y_true": inverse_y(y_true_test, scale_stats),
            "y_arx_sim": inverse_y(y_arx_test, scale_stats),
            "y_hybrid_sim": inverse_y(y_hybrid_test_best, scale_stats),
            "residual_correction_scaled": best_shrink * corr_test,
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(results), f, indent=2)
        f.write("\n")
    pd.DataFrame(leaderboard).sort_values("val_FIT_sim", ascending=False).to_csv(
        RESULTS_DIR / "leaderboard.csv",
        index=False,
    )
    test_predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    write_summary(results)
    return results


def write_summary(results: dict[str, Any]) -> None:
    arx_val = results["metrics"]["arx"]["validation"]
    arx_test = results["metrics"]["arx"]["test"]
    hy_val = results["metrics"]["hybrid_selected"]["validation"]
    hy_test = results["metrics"]["hybrid_selected"]["test"]
    selected_shrink = results["model"]["selected_shrink"]
    lines = [
        "# Summary",
        "",
        f"Selected shrink: `{selected_shrink}`",
        "",
        "| Model | Val FIT_sim | Test FIT_sim | Test RMSE | Test Bias |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| ARX backbone | {arx_val['FIT']:.3f} | {arx_test['FIT']:.3f} | "
            f"{arx_test['RMSE']:.4f} | {arx_test['Bias']:.4f} |"
        ),
        (
            f"| Conservative Hybrid ARX Residual | {hy_val['FIT']:.3f} | {hy_test['FIT']:.3f} | "
            f"{hy_test['RMSE']:.4f} | {hy_test['Bias']:.4f} |"
        ),
        "",
        "Validation is used for shrink selection. Test is reported once after selection.",
        "Residual features use ARX simulated trajectory and input lags `[2, 3, 6, 12, 24]` only.",
        "",
    ]
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def print_summary(results: dict[str, Any]) -> None:
    arx_val = results["metrics"]["arx"]["validation"]
    arx_test = results["metrics"]["arx"]["test"]
    hy_val = results["metrics"]["hybrid_selected"]["validation"]
    hy_test = results["metrics"]["hybrid_selected"]["test"]
    print("=== ARX Redo 70-75 ===")
    print(f"ARX backbone     Val FIT={arx_val['FIT']:.3f} | Test FIT={arx_test['FIT']:.3f}")
    print(f"Hybrid selected  Val FIT={hy_val['FIT']:.3f} | Test FIT={hy_test['FIT']:.3f}")
    print(f"Test RMSE hybrid = {hy_test['RMSE']:.4f}")
    print(f"Artifacts saved to {RESULTS_DIR}")


def main() -> None:
    cfg = PipelineConfig()
    results = run_pipeline(cfg)
    print_summary(results)


if __name__ == "__main__":
    main()

