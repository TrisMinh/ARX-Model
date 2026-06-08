from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from algorithm import ArxSpec, default_specs, fit_arx, max_lag, predict_one_step, simulate_chunked, simulate_free_run
from config import INPUT_COLS, ExperimentConfig
from data.collection.step_00_data_io import MODEL_COLS, normalize_model_columns
from evaluation import evaluate_model, fit_metrics
from preprocessing import add_features, apply_scale, fit_scale_stats, inverse_y, split_time
from preprocessing.scaling import ScaleStats
from utils import ensure_dir, write_json


# Tính metric sau khi inverse scale Soil_Moisture.
def _metric_from_z(y_true_z: np.ndarray, y_pred_z: np.ndarray, stats: ScaleStats) -> dict[str, float]:
    return fit_metrics(inverse_y(y_true_z, stats), inverse_y(y_pred_z, stats))


# Tạo giới hạn dự đoán Soil_Moisture trên scale chuẩn hóa.
def _physical_clip(stats: ScaleStats, cfg: ExperimentConfig) -> tuple[float, float]:
    mean, std = stats["Soil_Moisture"]
    low = max(0.0, cfg.soil_low_sp - cfg.soil_clip_margin)
    high = min(100.0, cfg.soil_high_sp + cfg.soil_clip_margin)
    return (float((low - mean) / std), float((high - mean) / std))


# Fit các ứng viên ARX và chọn bằng validation free-run.
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
        theta = fit_arx(train_z, spec, INPUT_COLS)
        val_1, val_true_1 = predict_one_step(val_z, theta, spec, INPUT_COLS, clip)
        val_free, val_true_free = simulate_free_run(val_z, theta, spec, INPUT_COLS, clip)
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
                "n_input_cols": len(INPUT_COLS),
                "n_params": int(len(theta)),
                "val_FIT_1step": _metric_from_z(val_true_1, val_1, stats)["FIT"],
                "val_FIT_free_run": _metric_from_z(val_true_free, val_free, stats)["FIT"],
                "val_RMSE_free_run": _metric_from_z(val_true_free, val_free, stats)["RMSE"],
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values("val_FIT_free_run", ascending=False).reset_index(drop=True)
    return leaderboard, fitted


# Tạo file dự đoán trên test.
def _build_predictions(
    test: pd.DataFrame,
    test_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    stats: ScaleStats,
    clip: tuple[float, float],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    y_1, y_true = predict_one_step(test_z, theta, spec, INPUT_COLS, clip)
    y_5m, _ = simulate_chunked(test_z, theta, spec, INPUT_COLS, clip, cfg.n_step_5min)
    y_20m, _ = simulate_chunked(test_z, theta, spec, INPUT_COLS, clip, cfg.n_step_control)
    y_free, _ = simulate_free_run(test_z, theta, spec, INPUT_COLS, clip)
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


# Tạo artifact đủ để chạy lại model.
def _runtime_artifact(
    cfg: ExperimentConfig,
    spec: ArxSpec,
    theta: np.ndarray,
    stats: ScaleStats,
    clip: tuple[float, float],
) -> dict[str, Any]:
    return {
        "target": "Soil_Moisture",
        "sampling_seconds": cfg.sampling_seconds,
        "input_cols": list(INPUT_COLS),
        "spec": asdict(spec),
        "theta": theta,
        "scale": {col: {"mean": mean, "std": std} for col, (mean, std) in stats.items()},
        "clip_scaled": list(clip),
    }


# Ghi tóm tắt kết quả model.
def _write_summary(results_dir: Path, payload: dict[str, Any]) -> None:
    model = payload["model"]
    train_eval = payload["train"]
    validation = payload["validation"]
    test = payload["test"]
    lines = [
        "# ARX 5s Clean Results",
        "",
        f"- Selected: `{model['name']}`",
        f"- Output memory: {model['output_memory_seconds']:.0f}s",
        f"- Input delay: {model['input_delay_seconds']:.0f}s",
        f"- Input memory: {model['input_memory_seconds']:.0f}s",
        f"- Split: {model['split']}",
        f"- Train FIT 1-step: {train_eval['metrics_1step']['FIT']:.3f}",
        f"- Train FIT 5min chunked: {train_eval['metrics_5min_chunked']['FIT']:.3f}",
        f"- Train FIT 20min chunked: {train_eval['metrics_20min_chunked']['FIT']:.3f}",
        f"- Train FIT free-run: {train_eval['metrics_free_run']['FIT']:.3f}",
        f"- Validation FIT 1-step: {validation['metrics_1step']['FIT']:.3f}",
        f"- Validation FIT 5min chunked: {validation['metrics_5min_chunked']['FIT']:.3f}",
        f"- Validation FIT 20min chunked: {validation['metrics_20min_chunked']['FIT']:.3f}",
        f"- Validation FIT free-run: {validation['metrics_free_run']['FIT']:.3f}",
        f"- Test FIT 1-step: {test['metrics_1step']['FIT']:.3f}",
        f"- Test FIT 5min chunked: {test['metrics_5min_chunked']['FIT']:.3f}",
        f"- Test FIT 20min chunked: {test['metrics_20min_chunked']['FIT']:.3f}",
        f"- Test FIT free-run: {test['metrics_free_run']['FIT']:.3f}",
        f"- Train/Validation/Test RMSE free-run: "
        f"{train_eval['metrics_free_run']['RMSE']:.5f} / "
        f"{validation['metrics_free_run']['RMSE']:.5f} / "
        f"{test['metrics_free_run']['RMSE']:.5f}",
    ]
    (results_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Train ARX từ data cuối và ghi kết quả.
def run_pipeline(project_root: Path, cfg: ExperimentConfig, grid: str) -> dict[str, Any]:
    data_dir = ensure_dir(project_root / "data")
    results_dir = ensure_dir(project_root / "result")
    data_path = data_dir / "mini_greenhouse_5s_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. Run scripts/01_build_data.py before training."
        )

    print("[Data] loading collection-built 5s data", flush=True)
    raw = normalize_model_columns(pd.read_csv(data_path))
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    raw = raw.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    df = add_features(raw)
    train, val, test = split_time(df, cfg)

    stats = fit_scale_stats(train, INPUT_COLS)
    train_z = apply_scale(train, stats)
    val_z = apply_scale(val, stats)
    test_z = apply_scale(test, stats)
    clip = _physical_clip(stats, cfg)

    leaderboard, fitted = _train_candidates(train_z, val_z, stats, clip, cfg, grid)
    selected = leaderboard.iloc[0].to_dict()
    spec, theta = fitted[str(selected["model"])]

    print(f"[ARX] selected {spec.name}", flush=True)
    train_eval = evaluate_model(train_z, theta, spec, INPUT_COLS, stats, clip, cfg)
    validation = evaluate_model(val_z, theta, spec, INPUT_COLS, stats, clip, cfg)
    test_eval = evaluate_model(test_z, theta, spec, INPUT_COLS, stats, clip, cfg)
    predictions = _build_predictions(test, test_z, theta, spec, stats, clip, cfg)

    payload = {
        "model": {
            "name": spec.name,
            "na": spec.na,
            "nb": spec.nb,
            "nk": spec.nk,
            "alpha": spec.alpha,
            "sampling_seconds": cfg.sampling_seconds,
            "output_memory_seconds": selected["output_memory_seconds"],
            "input_delay_seconds": selected["input_delay_seconds"],
            "input_memory_seconds": selected["input_memory_seconds"],
            "n_input_cols": len(INPUT_COLS),
            "n_params": int(len(theta)),
            "input_cols": list(INPUT_COLS),
            "split": "train/validation/test use the same time blocks on different days",
            "time_blocks": [list(block) for block in cfg.eval_time_blocks],
            "horizons_seconds": {
                "one_step": cfg.sampling_seconds,
                "five_minute": cfg.five_minute_seconds,
                "control": cfg.control_horizon_seconds,
            },
        },
        "train": train_eval,
        "validation": validation,
        "test": test_eval,
    }

    raw.loc[:, MODEL_COLS].to_csv(data_path, index=False)
    leaderboard.to_csv(results_dir / "leaderboard.csv", index=False)
    predictions.to_csv(results_dir / "test_predictions.csv", index=False)
    write_json(results_dir / "metrics.json", payload)
    write_json(results_dir / "arx_5s_model.json", _runtime_artifact(cfg, spec, theta, stats, clip))
    _write_summary(results_dir, payload)
    return payload
