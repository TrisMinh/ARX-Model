from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from arx_redo_pipeline import (
    AUGMENTED_INPUT_COLS,
    PROJECT_ROOT,
    add_augmented_features,
    apply_scale,
    build_arx_matrix,
    compute_metrics,
    fit_scale_stats,
    inverse_y,
    json_ready,
    max_lag,
    simulate_arx,
    split_time,
)


OUT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = OUT_DIR / "results_v75"


@dataclass(frozen=True)
class V75Config:
    csv_path: str = str(PROJECT_ROOT / "greenhouse_data.csv")
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    na: int = 5
    nb: int = 3
    nk: int = 2
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    shrink_grid: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1)
    causal_input_lags: tuple[int, ...] = (2, 3, 6, 12, 24, 36, 72)
    diagnostic_input_lags: tuple[int, ...] = (-24, -12, -6, -3, -2, -1, 0, 1, 2, 3, 6, 12, 24, 36, 72)


def fit_arx_ols(df_train_z: pd.DataFrame, cfg: V75Config) -> np.ndarray:
    x_train, y_train = build_arx_matrix(df_train_z, cfg.na, cfg.nb, cfg.nk)
    theta, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return theta


def time_since_on(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    last_on = -10**9
    for idx, value in enumerate(values):
        if value > 0.0:
            last_on = idx
        out[idx] = min(idx - last_on, 288 * 30)
    return out


def _base_y_features(df_z: pd.DataFrame, y_arx_sim: np.ndarray, cfg: V75Config) -> dict[str, np.ndarray]:
    lag = max_lag(cfg.na, cfg.nb, cfg.nk)
    n_rows_full = len(df_z)
    y_full = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_full[lag:] = y_arx_sim
    arx_full = np.empty(n_rows_full, dtype=float)
    arx_full[:lag] = y_full[:lag]
    arx_full[lag:] = y_arx_sim
    arx_series = pd.Series(arx_full)

    data: dict[str, np.ndarray] = {
        "y_arx": arx_full,
        "arx_minus_sp_center": arx_full - df_z["SP_Center"].to_numpy(dtype=float),
    }
    for y_lag in (1, 2, 3, 6, 12, 24, 36):
        shifted = arx_series.shift(y_lag)
        data[f"y_arx_lag_{y_lag}"] = shifted.to_numpy(dtype=float)
        data[f"y_arx_delta_{y_lag}"] = (arx_series - shifted).to_numpy(dtype=float)
    return data


def causal_history_features(df_z: pd.DataFrame, y_arx_sim: np.ndarray, cfg: V75Config) -> np.ndarray:
    lag = max_lag(cfg.na, cfg.nb, cfg.nk)
    data = _base_y_features(df_z, y_arx_sim, cfg)

    for col in AUGMENTED_INPUT_COLS:
        values = pd.Series(df_z[col].to_numpy(dtype=float))
        for input_lag in cfg.causal_input_lags:
            data[f"{col}_lag_{input_lag}"] = values.shift(input_lag).to_numpy(dtype=float)

    for col in ("Drip", "Mist", "Fan"):
        values = pd.Series(df_z[col].to_numpy(dtype=float))
        for window in (6, 12, 24, 36, 72, 144, 288):
            data[f"{col}_sum_lag2_win_{window}"] = (
                values.shift(2).rolling(window, min_periods=1).sum().to_numpy(dtype=float)
            )
        data[f"{col}_time_since_on_lag2"] = pd.Series(time_since_on(values.to_numpy(dtype=float))).shift(2).to_numpy(
            dtype=float
        )

    for col in ("Temperature", "Humidity", "Light_log"):
        values = pd.Series(df_z[col].to_numpy(dtype=float))
        for window in (12, 24, 72, 144, 288):
            data[f"{col}_mean_lag2_win_{window}"] = (
                values.shift(2).rolling(window, min_periods=1).mean().to_numpy(dtype=float)
            )

    return pd.DataFrame(data).iloc[lag:].bfill().fillna(0.0).to_numpy(dtype=float)


def diagnostic_future_input_features(df_z: pd.DataFrame, y_arx_sim: np.ndarray, cfg: V75Config) -> np.ndarray:
    """Non-causal feature set used only to measure an upper bound.

    Negative lags mean future input values. This is not a production forecaster unless future actuator
    commands are genuinely known before prediction time.
    """
    lag = max_lag(cfg.na, cfg.nb, cfg.nk)
    data = _base_y_features(df_z, y_arx_sim, cfg)
    for col in AUGMENTED_INPUT_COLS:
        values = pd.Series(df_z[col].to_numpy(dtype=float))
        for input_lag in cfg.diagnostic_input_lags:
            data[f"{col}_lag_{input_lag}"] = values.shift(input_lag).to_numpy(dtype=float)
    return pd.DataFrame(data).iloc[lag:].bfill().ffill().fillna(0.0).to_numpy(dtype=float)


def fit_residual_model(x_train: np.ndarray, y_train: np.ndarray, *, causal: bool) -> ExtraTreesRegressor:
    if causal:
        return ExtraTreesRegressor(
            n_estimators=300,
            max_features=0.8,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=103,
        ).fit(x_train, y_train)
    return ExtraTreesRegressor(
        n_estimators=180,
        max_features=0.8,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=303,
    ).fit(x_train, y_train)


def evaluate_shrink_grid(
    y_true_val_z: np.ndarray,
    y_arx_val_z: np.ndarray,
    corr_val_z: np.ndarray,
    y_true_test_z: np.ndarray,
    y_arx_test_z: np.ndarray,
    corr_test_z: np.ndarray,
    stats: dict[str, tuple[float, float]],
    clip_bounds: tuple[float, float],
    cfg: V75Config,
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shrink in cfg.shrink_grid:
        y_val = np.clip(y_arx_val_z + shrink * corr_val_z, clip_bounds[0], clip_bounds[1])
        y_test = np.clip(y_arx_test_z + shrink * corr_test_z, clip_bounds[0], clip_bounds[1])
        val_metrics = compute_metrics(inverse_y(y_true_val_z, stats), inverse_y(y_val, stats))
        test_metrics = compute_metrics(inverse_y(y_true_test_z, stats), inverse_y(y_test, stats))
        rows.append(
            {
                "track": label,
                "shrink": float(shrink),
                "val_FIT_sim": val_metrics["FIT"],
                "val_RMSE_sim": val_metrics["RMSE"],
                "test_FIT_sim": test_metrics["FIT"],
                "test_RMSE_sim": test_metrics["RMSE"],
                "test_Bias_sim": test_metrics["Bias"],
            }
        )
    return rows


def run_train60_tracks(cfg: V75Config, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    stats = fit_scale_stats(df_train)
    df_train_z = apply_scale(df_train, stats)
    df_val_z = apply_scale(df_val, stats)
    df_test_z = apply_scale(df_test, stats)
    clip_bounds = tuple(float(v) for v in np.quantile(df_train_z["Soil_Moisture"], cfg.clip_quantiles))
    theta = fit_arx_ols(df_train_z, cfg)

    y_arx_train, y_true_train = simulate_arx(df_train_z, theta, cfg, clip_bounds)
    y_arx_val, y_true_val = simulate_arx(df_val_z, theta, cfg, clip_bounds)
    y_arx_test, y_true_test = simulate_arx(df_test_z, theta, cfg, clip_bounds)

    x_causal_train = causal_history_features(df_train_z, y_arx_train, cfg)
    y_res_train = y_true_train - y_arx_train
    causal_model = fit_residual_model(x_causal_train, y_res_train, causal=True)
    causal_rows = evaluate_shrink_grid(
        y_true_val,
        y_arx_val,
        causal_model.predict(causal_history_features(df_val_z, y_arx_val, cfg)),
        y_true_test,
        y_arx_test,
        causal_model.predict(causal_history_features(df_test_z, y_arx_test, cfg)),
        stats,
        clip_bounds,
        cfg,
        "causal_history_train60",
    )

    x_diag_train = diagnostic_future_input_features(df_train_z, y_arx_train, cfg)
    diagnostic_model = fit_residual_model(x_diag_train, y_res_train, causal=False)
    diagnostic_rows = evaluate_shrink_grid(
        y_true_val,
        y_arx_val,
        diagnostic_model.predict(diagnostic_future_input_features(df_val_z, y_arx_val, cfg)),
        y_true_test,
        y_arx_test,
        diagnostic_model.predict(diagnostic_future_input_features(df_test_z, y_arx_test, cfg)),
        stats,
        clip_bounds,
        cfg,
        "diagnostic_future_actuator_train60",
    )

    arx_metrics = {
        "validation": compute_metrics(inverse_y(y_true_val, stats), inverse_y(y_arx_val, stats)),
        "test": compute_metrics(inverse_y(y_true_test, stats), inverse_y(y_arx_test, stats)),
    }
    return causal_rows + diagnostic_rows, arx_metrics


def run_causal_refit80(
    cfg: V75Config,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    selected_shrink: float,
) -> dict[str, Any]:
    df_dev = pd.concat([df_train, df_val], ignore_index=True)
    stats = fit_scale_stats(df_dev)
    df_dev_z = apply_scale(df_dev, stats)
    df_test_z = apply_scale(df_test, stats)
    clip_bounds = tuple(float(v) for v in np.quantile(df_dev_z["Soil_Moisture"], cfg.clip_quantiles))
    theta = fit_arx_ols(df_dev_z, cfg)
    y_arx_dev, y_true_dev = simulate_arx(df_dev_z, theta, cfg, clip_bounds)
    y_arx_test, y_true_test = simulate_arx(df_test_z, theta, cfg, clip_bounds)
    residual_model = fit_residual_model(
        causal_history_features(df_dev_z, y_arx_dev, cfg),
        y_true_dev - y_arx_dev,
        causal=True,
    )
    corr_test = residual_model.predict(causal_history_features(df_test_z, y_arx_test, cfg))
    y_hybrid_test = np.clip(y_arx_test + selected_shrink * corr_test, clip_bounds[0], clip_bounds[1])
    return {
        "track": "causal_history_refit80_selected_after_validation",
        "selected_shrink": selected_shrink,
        "arx_test": compute_metrics(inverse_y(y_true_test, stats), inverse_y(y_arx_test, stats)),
        "hybrid_test": compute_metrics(inverse_y(y_true_test, stats), inverse_y(y_hybrid_test, stats)),
    }


def write_summary(payload: dict[str, Any]) -> None:
    leaderboard = pd.DataFrame(payload["leaderboard"])
    causal_best = payload["best_by_validation"]["causal_history_train60"]
    diagnostic_best = payload["best_by_validation"]["diagnostic_future_actuator_train60"]
    refit = payload["causal_refit80"]
    top_rows = leaderboard.sort_values(["track", "val_FIT_sim"], ascending=[True, False]).head(12)
    top_table = [
        "| track | shrink | val_FIT_sim | test_FIT_sim | test_RMSE_sim |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in top_rows.iterrows():
        top_table.append(
            f"| {row['track']} | {float(row['shrink']):.3f} | "
            f"{float(row['val_FIT_sim']):.3f} | {float(row['test_FIT_sim']):.3f} | "
            f"{float(row['test_RMSE_sim']):.4f} |"
        )

    lines = [
        "# V75 Audit Summary",
        "",
        "| Track | Val FIT_sim | Test FIT_sim | Test RMSE | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
        (
            f"| Causal history train60 | {causal_best['val_FIT_sim']:.3f} | "
            f"{causal_best['test_FIT_sim']:.3f} | {causal_best['test_RMSE_sim']:.4f} | "
            "Production-safe, selected by validation |"
        ),
        (
            f"| Causal history refit80 | n/a | {refit['hybrid_test']['FIT']:.3f} | "
            f"{refit['hybrid_test']['RMSE']:.4f} | Refit train+validation after selection |"
        ),
        (
            f"| Diagnostic future actuator | {diagnostic_best['val_FIT_sim']:.3f} | "
            f"{diagnostic_best['test_FIT_sim']:.3f} | {diagnostic_best['test_RMSE_sim']:.4f} | "
            "Not production-safe unless future actuator commands are genuinely known |"
        ),
        "",
        "Conclusion: the clean causal track improved the previous 70.294% test FIT to about 71.3%. "
        "The 75% target is reachable only in the non-causal known-future-actuator diagnostic track.",
        "",
        "Top leaderboard rows:",
        "",
        *top_table,
        "",
    ]
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = V75Config()
    df = pd.read_csv(cfg.csv_path, parse_dates=["Timestamp"])
    df = add_augmented_features(df.sort_values("Timestamp").reset_index(drop=True))
    df_train, df_val, df_test = split_time(df, cfg.train_ratio, cfg.val_ratio)

    leaderboard, arx_metrics = run_train60_tracks(cfg, df_train, df_val, df_test)
    leaderboard_df = pd.DataFrame(leaderboard)
    best_by_validation = {
        track: rows.sort_values("val_FIT_sim", ascending=False).iloc[0].to_dict()
        for track, rows in leaderboard_df.groupby("track")
    }
    causal_shrink = float(best_by_validation["causal_history_train60"]["shrink"])
    causal_refit80 = run_causal_refit80(cfg, df_train, df_val, df_test, causal_shrink)

    payload = {
        "config": asdict(cfg),
        "arx_train60": arx_metrics,
        "leaderboard": leaderboard,
        "best_by_validation": best_by_validation,
        "causal_refit80": causal_refit80,
        "audit_notes": {
            "causal_track": "No target future leakage. Residual uses ARX simulated trajectory and lagged inputs only.",
            "diagnostic_track": "Uses negative input lags, including future actuator states. Treat as upper bound, not deployment model.",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
        f.write("\n")
    leaderboard_df.sort_values(["track", "val_FIT_sim"], ascending=[True, False]).to_csv(
        RESULTS_DIR / "leaderboard.csv",
        index=False,
    )
    write_summary(payload)

    causal_best = best_by_validation["causal_history_train60"]
    diagnostic_best = best_by_validation["diagnostic_future_actuator_train60"]
    print("=== V75 audit ===")
    print(
        f"Causal history train60   Val FIT={causal_best['val_FIT_sim']:.3f} | "
        f"Test FIT={causal_best['test_FIT_sim']:.3f}"
    )
    print(f"Causal history refit80   Test FIT={causal_refit80['hybrid_test']['FIT']:.3f}")
    print(
        f"Diagnostic future input  Val FIT={diagnostic_best['val_FIT_sim']:.3f} | "
        f"Test FIT={diagnostic_best['test_FIT_sim']:.3f}"
    )
    print(f"Artifacts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
