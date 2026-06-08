from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from arx5s_clean.data.collection.step_00_source_data import (
    DataSource,
    load_model_csv,
    load_source_data,
    normalize_source,
    real_csv_paths,
)
from arx5s_clean.data.collection.step_00_schema import MODEL_COLS


def inject_collection_artifacts(df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    """Add small realistic raw-collection defects before preprocessing."""
    out = df.copy()
    if session_name == "morning_anchor":
        out.loc[out.index[86], "Light"] = np.nan
        out = out.drop(out.index[[214, 215]])
    elif session_name == "noon_anchor":
        out.loc[out.index[120:126], ["Temperature", "Humidity"]] = np.nan
        out = out.drop(out.index[[410]])
    elif session_name == "afternoon_anchor":
        out.loc[out.index[300:302], "Soil_Moisture"] = np.nan
        duplicate = out.iloc[[620]].copy()
        out = pd.concat([out.iloc[:621], duplicate, out.iloc[621:]], ignore_index=True)
    elif session_name == "night_anchor":
        out.loc[out.index[500], "Fan"] = np.nan
        out = out.drop(out.index[[740, 741, 742]])
    return out.loc[:, MODEL_COLS]


def raw_output_name(index: int, source_stem: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in source_stem)
    if safe.endswith("_raw"):
        return f"{index:02d}_{safe}.csv"
    return f"{index:02d}_{safe}_raw.csv"


def build_raw_sessions(raw_dir: Path, seed: int, source: DataSource = "legacy", real_dir: Path | None = None) -> list[Path]:
    _ = seed
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    normalized_source = normalize_source(source)

    if normalized_source == "real":
        for idx, source_path in enumerate(real_csv_paths(real_dir), start=1):
            raw = load_model_csv(source_path)
            path = raw_dir / raw_output_name(idx, source_path.stem)
            raw.to_csv(path, index=False)
            paths.append(path)
        return paths

    source_data = load_source_data(source, real_dir)
    source_data["Timestamp"] = pd.to_datetime(source_data["Timestamp"])

    windows = (
        ("morning_anchor", "2026-01-01 07:00:00", "2026-01-01 09:00:00"),
        ("noon_anchor", "2026-01-01 11:30:00", "2026-01-01 13:30:00"),
        ("afternoon_anchor", "2026-01-01 15:00:00", "2026-01-01 17:00:00"),
        ("night_anchor", "2026-01-01 20:00:00", "2026-01-01 22:00:00"),
    )

    for name, start, end in windows:
        df = source_data[
            (source_data["Timestamp"] >= pd.Timestamp(start)) & (source_data["Timestamp"] < pd.Timestamp(end))
        ].copy()
        df = df.reset_index(drop=True)
        df = inject_collection_artifacts(df, name)
        path = raw_dir / f"{name}_raw.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths
