from __future__ import annotations

import numpy as np
import pandas as pd

from config import ExperimentConfig


ScaleStats = dict[str, tuple[float, float]]


# Đánh dấu các dòng nằm trong những block giờ được chọn.
def in_time_blocks(timestamp: pd.Series, blocks: tuple[tuple[float, float], ...]) -> pd.Series:
    time_of_day = timestamp - timestamp.dt.normalize()
    mask = pd.Series(False, index=timestamp.index)

    for start_hour, end_hour in blocks:
        start = pd.to_timedelta(start_hour, unit="h")
        end = pd.to_timedelta(end_hour, unit="h")
        mask = mask | ((time_of_day >= start) & (time_of_day < end))

    return mask


# Chia train, validation, test theo thời gian.
def split_time(df: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if getattr(cfg, "split_strategy", "ratio") == "same_clock_by_day":
        sorted_df = df.sort_values("Timestamp").reset_index(drop=True)
        timestamp = pd.to_datetime(sorted_df["Timestamp"])
        days = sorted(timestamp.dt.normalize().dropna().unique())
        if len(days) >= 3:
            val_day = pd.Timestamp(days[-2])
            test_day = pd.Timestamp(days[-1])
            blocks = getattr(cfg, "eval_time_blocks", ((0.0, 24.0),))
            selected_hours = in_time_blocks(timestamp, blocks)

            train = sorted_df[(timestamp < val_day) & selected_hours]
            val = sorted_df[(timestamp >= val_day) & (timestamp < test_day) & selected_hours]
            test = sorted_df[(timestamp >= test_day) & selected_hours]
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
