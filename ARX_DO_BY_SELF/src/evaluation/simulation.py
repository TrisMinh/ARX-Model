from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from algorithm import ArxSpec, max_lag, predict_one_step, simulate_chunked, simulate_free_run
from config import ExperimentConfig
from evaluation.metrics import fit_metrics
from preprocessing.scaling import ScaleStats, inverse_y


# Tính metric sau khi inverse scale.
def _metric(y_true_z: np.ndarray, y_pred_z: np.ndarray, stats: ScaleStats) -> dict[str, float]:
    return fit_metrics(inverse_y(y_true_z, stats), inverse_y(y_pred_z, stats))


# Đánh giá model theo 1-step, 5 phút, 20 phút và free-run.
def evaluate_model(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    stats: ScaleStats,
    clip: tuple[float, float],
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    y_1, yt_1 = predict_one_step(df_z, theta, spec, input_cols, clip)
    y_5m, yt_5m = simulate_chunked(df_z, theta, spec, input_cols, clip, cfg.n_step_5min)
    y_20m, yt_20m = simulate_chunked(df_z, theta, spec, input_cols, clip, cfg.n_step_control)
    y_free, yt_free = simulate_free_run(df_z, theta, spec, input_cols, clip)

    return {
        "lag_steps": max_lag(spec),
        "metrics_1step": _metric(yt_1, y_1, stats),
        "metrics_5min_chunked": _metric(yt_5m, y_5m, stats),
        "metrics_20min_chunked": _metric(yt_20m, y_20m, stats),
        "metrics_free_run": _metric(yt_free, y_free, stats),
    }
