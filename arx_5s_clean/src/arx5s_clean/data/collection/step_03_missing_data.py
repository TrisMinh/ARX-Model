from __future__ import annotations

import numpy as np
import pandas as pd

from arx5s_clean.data.collection.step_00_schema import ACTUATOR_COLS, MODEL_COLS, SENSOR_RANGES


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MODEL_COLS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fill_sensor_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, (low, high) in SENSOR_RANGES.items():
        invalid = (out[col] < low) | (out[col] > high)
        out.loc[invalid, col] = np.nan
        out[col] = out[col].interpolate(method="time", limit=12, limit_direction="both")
        out[col] = out[col].ffill().bfill()
        out[col] = out[col].clip(low, high)
    return out


def fill_actuator_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ACTUATOR_COLS:
        invalid = ~out[col].isin([0.0, 1.0]) & out[col].notna()
        out.loc[invalid, col] = np.nan
        out[col] = out[col].ffill(limit=12).bfill(limit=2).fillna(0.0)
        out[col] = (out[col] >= 0.5).astype(float)
    return out
