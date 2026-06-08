from __future__ import annotations

import pandas as pd

from arx5s_clean.data.collection.step_00_schema import MODEL_COLS, SAMPLE_SECONDS, normalize_model_columns


def require_model_columns(raw: pd.DataFrame, file_name: str) -> None:
    try:
        normalize_model_columns(raw)
    except ValueError as exc:
        raise ValueError(f"{file_name} {exc}") from exc


def parse_sort_timestamps(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_model_columns(raw)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp")


def collapse_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "Temperature": "mean",
        "Humidity": "mean",
        "Light": "mean",
        "Soil_Moisture": "mean",
        "Drip": "max",
        "Mist": "max",
        "Fan": "max",
    }
    collapsed = df.groupby("Timestamp", as_index=False).agg(agg)
    return collapsed


def reindex_to_5s_grid(df: pd.DataFrame) -> pd.DataFrame:
    start = df["Timestamp"].min()
    end = df["Timestamp"].max()
    grid = pd.date_range(start, end, freq=f"{SAMPLE_SECONDS}s")
    gridded = df.set_index("Timestamp").reindex(grid)
    gridded.index.name = "Timestamp"
    return gridded
