from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.collection.step_00_data_io import (
    ACTUATOR_COLS,
    MODEL_COLS,
    SAMPLE_SECONDS,
    SENSOR_RANGES,
    format_model_data,
    normalize_model_columns,
)


# Kiểm tra file có đủ 8 cột bắt buộc.
def require_model_columns(raw: pd.DataFrame, file_name: str) -> None:
    try:
        normalize_model_columns(raw)
    except ValueError as exc:
        raise ValueError(f"{file_name} {exc}") from exc


# Parse Timestamp và sắp xếp theo thời gian.
def parse_sort_timestamps(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_model_columns(raw)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.round(f"{SAMPLE_SECONDS}s")
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp")


# Gộp các dòng bị trùng Timestamp.
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
    return df.groupby("Timestamp", as_index=False).agg(agg)


# Đưa dữ liệu về lưới thời gian 5 giây.
def reindex_to_5s_grid(df: pd.DataFrame) -> pd.DataFrame:
    start = df["Timestamp"].min().floor(f"{SAMPLE_SECONDS}s")
    end = df["Timestamp"].max().ceil(f"{SAMPLE_SECONDS}s")
    grid = pd.date_range(start, end, freq=f"{SAMPLE_SECONDS}s")
    gridded = df.set_index("Timestamp").reindex(grid)
    gridded.index.name = "Timestamp"
    return gridded


# Ép các cột sensor và actuator về dạng số.
def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MODEL_COLS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# Xử lý missing và outlier cho các cột sensor.
def fill_sensor_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, (low, high) in SENSOR_RANGES.items():
        invalid = (out[col] < low) | (out[col] > high)
        out.loc[invalid, col] = np.nan
        out[col] = out[col].interpolate(method="time", limit=12, limit_direction="both")
        out[col] = out[col].ffill().bfill()
        out[col] = out[col].clip(low, high)
    return out


# Lọc spike Soil_Moisture bị nhảy phi thực tế rồi nội suy lại.
def smooth_soil_moisture(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    soil = out["Soil_Moisture"].astype(float)

    local_median = soil.rolling(window=5, center=True, min_periods=1).median()
    spike = (soil - local_median).abs() > 5.0
    out.loc[spike, "Soil_Moisture"] = np.nan

    out["Soil_Moisture"] = out["Soil_Moisture"].interpolate(method="time", limit=12, limit_direction="both")
    out["Soil_Moisture"] = out["Soil_Moisture"].ffill().bfill()
    out["Soil_Moisture"] = out["Soil_Moisture"].rolling(window=3, center=True, min_periods=1).median()
    return out


# Xử lý missing cho Drip, Mist, Fan và ép về 0/1.
def fill_actuator_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ACTUATOR_COLS:
        invalid = ~out[col].isin([0.0, 1.0]) & out[col].notna()
        out.loc[invalid, col] = np.nan
        out[col] = out[col].ffill(limit=12).bfill(limit=2).fillna(0.0)
        out[col] = (out[col] >= 0.5).astype(float)
    return out


# Clean một file data thu được trong _01_data.
def clean_data_file(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    require_model_columns(raw, path.name)

    parsed = parse_sort_timestamps(raw)
    unique_ts = collapse_duplicate_timestamps(parsed)
    gridded = reindex_to_5s_grid(unique_ts)

    filled = coerce_numeric(gridded)
    filled = fill_sensor_missing(filled)
    filled = smooth_soil_moisture(filled)
    filled = fill_actuator_missing(filled)

    return format_model_data(filled.reset_index())


# Clean toàn bộ file trong _01_data và ghi data sạch vào _02_clean_data.
def clean_all_data_files(raw_paths: list[Path], processed_dir: Path) -> pd.DataFrame:
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned_files: list[pd.DataFrame] = []

    for path in raw_paths:
        cleaned = clean_data_file(path)
        output_dir = processed_dir
        if path.parent.name.startswith("2026-"):
            output_dir = processed_dir / path.parent.name
            output_dir.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output_dir / path.name.replace("_raw.csv", "_sau_xu_ly.csv"), index=False)
        cleaned_files.append(cleaned)

    cleaned_data = (
        pd.concat(cleaned_files, ignore_index=True)
        .sort_values("Timestamp")
        .reset_index(drop=True)
        .loc[:, MODEL_COLS]
    )
    cleaned_data.to_csv(processed_dir / "00_sau_xu_ly_tong_hop.csv", index=False)
    return cleaned_data
