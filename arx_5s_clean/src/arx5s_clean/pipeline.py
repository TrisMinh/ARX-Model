from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arx5s_clean.algorithm import ArxSpec, default_specs, fit_arx, max_lag, predict_one_step, simulate_chunked, simulate_free_run
from arx5s_clean.config import INSIDE_INPUT_COLS, ExperimentConfig
from arx5s_clean.data import generate_greenhouse_data
from arx5s_clean.evaluation import evaluate_model, fit_metrics
from arx5s_clean.preprocessing import add_features, apply_scale, fit_scale_stats, inverse_y, split_time
from arx5s_clean.preprocessing.scaling import ScaleStats
from arx5s_clean.utils import ensure_dir, write_json


EXPORT_DATA_COLS: tuple[str, ...] = (
    "Timestamp",
    "Day_Index",
    "Hour",
    "Protocol_Phase",
    "Greenhouse_Volume_m3",
    "Temperature_In",
    "Humidity_In",
    "Light_In",
    "Soil_Moisture",
    "Soil_Low_SP",
    "Soil_High_SP",
    "Drip",
    "Mist",
    "Fan",
    "Planned_Drip",
    "Planned_Mist",
    "Planned_Fan",
    "Command_Source",
    "Safety_Override",
    "Wet_Block",
    "Sampling_Seconds",
    "Generated_From",
)


def _metric_from_z(y_true_z: np.ndarray, y_pred_z: np.ndarray, stats: ScaleStats) -> dict[str, float]:
    return fit_metrics(inverse_y(y_true_z, stats), inverse_y(y_pred_z, stats))


def _audit_data(df: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, Any]:
    df = df.sort_values("Timestamp").reset_index(drop=True)
    diffs = pd.to_datetime(df["Timestamp"]).diff().dt.total_seconds().dropna()
    train, val, test = split_time(df, cfg)

    def split_summary(part: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(part)),
            "start": str(part["Timestamp"].iloc[0]),
            "end": str(part["Timestamp"].iloc[-1]),
            "soil_mean": float(part["Soil_Moisture"].mean()),
            "soil_std": float(part["Soil_Moisture"].std(ddof=0)),
            "drip_on_pct": float(100.0 * part["Drip"].mean()),
            "fan_on_pct": float(100.0 * part["Fan"].mean()),
        }

    return {
        "rows": int(len(df)),
        "start": str(df["Timestamp"].iloc[0]),
        "end": str(df["Timestamp"].iloc[-1]),
        "sampling_seconds": cfg.sampling_seconds,
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_timestamps": int(df["Timestamp"].duplicated().sum()),
        "median_sampling_seconds": float(diffs.median()),
        "irregular_sampling_count": int((diffs != diffs.median()).sum()),
        "soil_mean": float(df["Soil_Moisture"].mean()),
        "soil_std": float(df["Soil_Moisture"].std(ddof=0)),
        "soil_min": float(df["Soil_Moisture"].min()),
        "soil_max": float(df["Soil_Moisture"].max()),
        "temperature_in_range": [float(df["Temperature_In"].min()), float(df["Temperature_In"].max())],
        "humidity_in_range": [float(df["Humidity_In"].min()), float(df["Humidity_In"].max())],
        "drip_on_pct": float(100.0 * df["Drip"].mean()),
        "fan_on_pct": float(100.0 * df["Fan"].mean()),
        "mist_on_pct": float(100.0 * df["Mist"].mean()),
        "split": {
            "train": split_summary(train),
            "validation": split_summary(val),
            "test": split_summary(test),
        },
    }


def _train_candidates(
    train_z: pd.DataFrame,
    val_z: pd.DataFrame,
    stats: ScaleStats,
    clip: tuple[float, float],
    cfg: ExperimentConfig,
    grid: str,
) -> tuple[pd.DataFrame, dict[str, tuple[ArxSpec, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    fitted: dict[str, tuple[ArxSpec, np.ndarray]] = {}

    for spec in default_specs(grid):
        print(f"[ARX] fitting {spec.name}", flush=True)
        theta = fit_arx(train_z, spec, INSIDE_INPUT_COLS)
        val_1, val_true_1 = predict_one_step(val_z, theta, spec, INSIDE_INPUT_COLS, clip)
        val_free, val_true_free = simulate_free_run(val_z, theta, spec, INSIDE_INPUT_COLS, clip)
        fitted[spec.name] = (spec, theta)

        rows.append(
            {
                "model": spec.name,
                "na": spec.na,
                "nb": spec.nb,
                "nk": spec.nk,
                "alpha": spec.alpha,
                "output_memory_seconds": spec.na * cfg.sampling_seconds,
                "input_delay_seconds": spec.nk * cfg.sampling_seconds,
                "input_memory_seconds": spec.nb * cfg.sampling_seconds,
                "n_input_cols": len(INSIDE_INPUT_COLS),
                "n_params": int(len(theta)),
                "val_FIT_1step": _metric_from_z(val_true_1, val_1, stats)["FIT"],
                "val_FIT_free_run": _metric_from_z(val_true_free, val_free, stats)["FIT"],
                "val_RMSE_free_run": _metric_from_z(val_true_free, val_free, stats)["RMSE"],
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values("val_FIT_free_run", ascending=False).reset_index(drop=True)
    return leaderboard, fitted


def _build_predictions(
    test: pd.DataFrame,
    test_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    stats: ScaleStats,
    clip: tuple[float, float],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    y_1, y_true = predict_one_step(test_z, theta, spec, INSIDE_INPUT_COLS, clip)
    y_5m, _ = simulate_chunked(test_z, theta, spec, INSIDE_INPUT_COLS, clip, cfg.n_step_5min)
    y_20m, _ = simulate_chunked(test_z, theta, spec, INSIDE_INPUT_COLS, clip, cfg.n_step_control)
    y_free, _ = simulate_free_run(test_z, theta, spec, INSIDE_INPUT_COLS, clip)
    lag = max_lag(spec)

    return pd.DataFrame(
        {
            "Timestamp": test["Timestamp"].iloc[lag:].to_numpy(),
            "y_true": inverse_y(y_true, stats),
            "y_pred_1step": inverse_y(y_1, stats),
            "y_pred_5min_chunked": inverse_y(y_5m, stats),
            "y_pred_20min_chunked": inverse_y(y_20m, stats),
            "y_pred_free_run": inverse_y(y_free, stats),
        }
    )


def _runtime_artifact(
    cfg: ExperimentConfig,
    spec: ArxSpec,
    theta: np.ndarray,
    stats: ScaleStats,
    clip: tuple[float, float],
) -> dict[str, Any]:
    return {
        "schema": "arx5s_clean_runtime_v1",
        "target": "Soil_Moisture",
        "sampling_seconds": cfg.sampling_seconds,
        "input_cols": list(INSIDE_INPUT_COLS),
        "spec": asdict(spec),
        "theta": theta,
        "scale": {col: {"mean": mean, "std": std} for col, (mean, std) in stats.items()},
        "clip_scaled": list(clip),
        "feature_policy": {
            "source": "observed indoor sensors, actuator states, and time features",
            "created_features": [
                "Light_log",
                "TempIn_x_HumiIn",
                "TempIn_x_Light",
                "HumiIn_x_Light",
                "Indoor_Dryness",
                "VPD_Proxy_In",
                "Hour_sin",
                "Hour_cos",
                "Day_sin",
                "Day_cos",
            ],
        },
    }


def _write_summary(results_dir: Path, payload: dict[str, Any]) -> None:
    selected = payload["selected_by_validation"]
    test = payload["test"]
    lines = [
        "# ARX 5s Clean Results",
        "",
        f"- Selected: `{selected['model']}`",
        f"- Output memory: {selected['output_memory_seconds']:.0f}s",
        f"- Input delay: {selected['input_delay_seconds']:.0f}s",
        f"- Input memory: {selected['input_memory_seconds']:.0f}s",
        f"- Test FIT 1-step: {test['metrics_1step']['FIT']:.3f}",
        f"- Test FIT 5min chunked: {test['metrics_5min_chunked']['FIT']:.3f}",
        f"- Test FIT 20min chunked: {test['metrics_20min_chunked']['FIT']:.3f}",
        f"- Test FIT free-run: {test['metrics_free_run']['FIT']:.3f}",
        f"- Test RMSE free-run: {test['metrics_free_run']['RMSE']:.5f}",
        "",
        "Artifacts:",
        "",
        "- `../data/mini_greenhouse_5s_data.csv`",
        "- `leaderboard.csv`",
        "- `metrics.json`",
        "- `arx_5s_model.json`",
        "- `test_predictions.csv`",
    ]
    (results_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(project_root: Path, cfg: ExperimentConfig, grid: str) -> dict[str, Any]:
    data_dir = ensure_dir(project_root / "data")
    results_dir = ensure_dir(project_root / "results")

    print("[Data] generating native 5s greenhouse data", flush=True)
    raw = generate_greenhouse_data(cfg)
    df = add_features(raw)
    train, val, test = split_time(df, cfg)

    stats = fit_scale_stats(train, INSIDE_INPUT_COLS)
    train_z = apply_scale(train, stats)
    val_z = apply_scale(val, stats)
    test_z = apply_scale(test, stats)
    clip = tuple(float(v) for v in np.quantile(train_z["Soil_Moisture"], (0.005, 0.995)))

    leaderboard, fitted = _train_candidates(train_z, val_z, stats, clip, cfg, grid)
    selected = leaderboard.iloc[0].to_dict()
    spec, theta = fitted[str(selected["model"])]

    print(f"[ARX] selected {spec.name}", flush=True)
    validation = evaluate_model(val_z, theta, spec, INSIDE_INPUT_COLS, stats, clip, cfg)
    test_eval = evaluate_model(test_z, theta, spec, INSIDE_INPUT_COLS, stats, clip, cfg)
    predictions = _build_predictions(test, test_z, theta, spec, stats, clip, cfg)

    payload = {
        "config": asdict(cfg),
        "grid": grid,
        "data_policy": {
            "source_data": "native generated 5s physical protocol",
            "split": "time split 70/15/15",
            "selection_metric": "validation free-run FIT",
            "model_family": "linear ARX",
            "integration_policy": "standalone only",
        },
        "input_cols": list(INSIDE_INPUT_COLS),
        "clip_scaled": list(clip),
        "selected_by_validation": selected,
        "validation": validation,
        "test": test_eval,
        "data_audit": _audit_data(df, cfg),
    }

    raw.loc[:, EXPORT_DATA_COLS].to_csv(data_dir / "mini_greenhouse_5s_data.csv", index=False)
    leaderboard.to_csv(results_dir / "leaderboard.csv", index=False)
    predictions.to_csv(results_dir / "test_predictions.csv", index=False)
    write_json(results_dir / "metrics.json", payload)
    write_json(results_dir / "arx_5s_model.json", _runtime_artifact(cfg, spec, theta, stats, clip))
    _write_summary(results_dir, payload)
    return payload
