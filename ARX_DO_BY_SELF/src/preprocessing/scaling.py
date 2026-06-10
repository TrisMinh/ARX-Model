from __future__ import annotations

import numpy as np
import pandas as pd

from config import ExperimentConfig


ScaleStats = dict[str, tuple[float, float]]


# Chia train, validation, test theo thời gian.
def split_time(df: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if getattr(cfg, "split_strategy", "ratio") == "day_ratio":
        sorted_df = df.sort_values("Timestamp").reset_index(drop=True)
        timestamp = pd.to_datetime(sorted_df["Timestamp"])
        days = sorted(timestamp.dt.normalize().dropna().unique())
        if len(days) >= 3:
            n_days = len(days)
            n_train = max(1, int(round(n_days * cfg.train_ratio)))
            n_val = max(1, int(round(n_days * cfg.val_ratio)))
            if n_train + n_val >= n_days:
                n_val = max(1, n_days - n_train - 1)
            if n_train + n_val >= n_days:
                n_train = max(1, n_days - n_val - 1)

            train_days = days[:n_train]
            val_days = days[n_train : n_train + n_val]
            test_days = days[n_train + n_val :]
            day_key = timestamp.dt.normalize()

            train = sorted_df[day_key.isin(train_days)]
            val = sorted_df[day_key.isin(val_days)]
            test = sorted_df[day_key.isin(test_days)]
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


# Tính mean/std trên train.
def fit_scale_stats(df_train: pd.DataFrame, input_cols: tuple[str, ...]) -> ScaleStats:
    stats: ScaleStats = {}
    for col in ("Soil_Moisture", *input_cols):
        mean = float(df_train[col].astype(float).mean())
        std = float(df_train[col].astype(float).std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        stats[col] = (mean, std)
    return stats


# Chuẩn hóa dữ liệu bằng mean/std đã fit.
def apply_scale(df_in: pd.DataFrame, stats: ScaleStats) -> pd.DataFrame:
    df = df_in.copy()
    for col, (mean, std) in stats.items():
        df[col] = (df[col].astype(float) - mean) / std
    return df


# Đưa Soil_Moisture từ scale chuẩn hóa về scale thật.
def inverse_y(y_z: np.ndarray, stats: ScaleStats) -> np.ndarray:
    mean, std = stats["Soil_Moisture"]
    return np.asarray(y_z, dtype=float) * std + mean
