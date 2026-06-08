from __future__ import annotations

import numpy as np
import pandas as pd

from arx5s_clean.config import ExperimentConfig


ScaleStats = dict[str, tuple[float, float]]


def split_time(df: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if getattr(cfg, "split_strategy", "ratio") == "same_clock_by_day":
        sorted_df = df.sort_values("Timestamp").reset_index(drop=True)
        timestamp = pd.to_datetime(sorted_df["Timestamp"])
        days = sorted(timestamp.dt.normalize().dropna().unique())
        if len(days) >= 3:
            val_day = pd.Timestamp(days[-2])
            test_day = pd.Timestamp(days[-1])
            window_start = pd.to_timedelta(cfg.eval_window_start_hour, unit="h")
            window_end = pd.to_timedelta(cfg.eval_window_end_hour, unit="h")

            train = sorted_df[timestamp < val_day]
            val = sorted_df[(timestamp >= val_day + window_start) & (timestamp < val_day + window_end)]
            test = sorted_df[(timestamp >= test_day + window_start) & (timestamp < test_day + window_end)]
            if len(train) > 0 and len(val) > 0 and len(test) > 0:
                return (
                    train.reset_index(drop=True),
                    val.reset_index(drop=True),
                    test.reset_index(drop=True),
                )

    if getattr(cfg, "split_strategy", "ratio") == "last_day_clock":
        sorted_df = df.sort_values("Timestamp").reset_index(drop=True)
        timestamp = pd.to_datetime(sorted_df["Timestamp"])
        last_day = timestamp.dt.normalize().max()
        val_start = last_day + pd.to_timedelta(cfg.eval_window_start_hour, unit="h")
        test_start = last_day + pd.to_timedelta(cfg.eval_window_end_hour, unit="h")

        train = sorted_df[timestamp < val_start]
        val = sorted_df[(timestamp >= val_start) & (timestamp < test_start)]
        test = sorted_df[timestamp >= test_start]
        if len(train) > 0 and len(val) > 0 and len(test) > 0:
            return (
                train.reset_index(drop=True),
                val.reset_index(drop=True),
                test.reset_index(drop=True),
            )

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
