from __future__ import annotations

import numpy as np
import pandas as pd

from arx5s_clean.config import ExperimentConfig


ScaleStats = dict[str, tuple[float, float]]


def split_time(df: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    n_train = int(n_rows * cfg.train_ratio)
    n_val = int(n_rows * cfg.val_ratio)
    return (
        df.iloc[:n_train].reset_index(drop=True),
        df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        df.iloc[n_train + n_val :].reset_index(drop=True),
    )


def fit_scale_stats(df_train: pd.DataFrame, input_cols: tuple[str, ...]) -> ScaleStats:
    stats: ScaleStats = {}
    for col in ("Soil_Moisture", *input_cols):
        mean = float(df_train[col].astype(float).mean())
        std = float(df_train[col].astype(float).std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        stats[col] = (mean, std)
    return stats


def apply_scale(df_in: pd.DataFrame, stats: ScaleStats) -> pd.DataFrame:
    df = df_in.copy()
    for col, (mean, std) in stats.items():
        df[col] = (df[col].astype(float) - mean) / std
    return df


def inverse_y(y_z: np.ndarray, stats: ScaleStats) -> np.ndarray:
    mean, std = stats["Soil_Moisture"]
    return np.asarray(y_z, dtype=float) * std + mean

