from __future__ import annotations

from pathlib import Path

import pandas as pd

from arx5s_clean.data.collection.step_03_missing_data import (
    coerce_numeric,
    fill_actuator_missing,
    fill_sensor_missing,
)
from arx5s_clean.data.collection.step_00_schema import MODEL_COLS
from arx5s_clean.data.collection.step_02_timestamp_cleaning import (
    collapse_duplicate_timestamps,
    parse_sort_timestamps,
    reindex_to_5s_grid,
    require_model_columns,
)


def clean_session(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    require_model_columns(raw, path.name)

    parsed = parse_sort_timestamps(raw)
    unique_ts = collapse_duplicate_timestamps(parsed)
    gridded = reindex_to_5s_grid(unique_ts)

    filled = coerce_numeric(gridded)
    filled = fill_sensor_missing(filled)
    filled = fill_actuator_missing(filled)

    cleaned = filled.reset_index().loc[:, MODEL_COLS]
    return cleaned


def clean_all_sessions(
    raw_paths: list[Path],
    processed_dir: Path,
) -> pd.DataFrame:
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned_sessions: list[pd.DataFrame] = []

    for path in raw_paths:
        cleaned = clean_session(path)
        cleaned.to_csv(processed_dir / path.name.replace("_raw.csv", "_sau_xu_ly.csv"), index=False)
        cleaned_sessions.append(cleaned)

    cleaned_data = (
        pd.concat(cleaned_sessions, ignore_index=True)
        .sort_values("Timestamp")
        .reset_index(drop=True)
        .loc[:, MODEL_COLS]
    )
    cleaned_data.to_csv(processed_dir / "00_sau_xu_ly_tong_hop.csv", index=False)
    return cleaned_data
