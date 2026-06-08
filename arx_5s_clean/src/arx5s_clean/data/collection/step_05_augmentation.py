from __future__ import annotations

import numpy as np
import pandas as pd

from arx5s_clean.data.collection.step_00_source_data import DataSource, load_source_data, normalize_source
from arx5s_clean.data.collection.step_00_environment import base_environment, soil_response
from arx5s_clean.data.collection.step_00_schema import MODEL_COLS, SAMPLE_SECONDS


def _extract_actuator_template(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample = df.loc[:, ["Drip", "Mist", "Fan"]].copy()
    if sample.empty:
        raise ValueError("source data is empty")
    return (
        sample["Drip"].to_numpy(dtype=float),
        sample["Mist"].to_numpy(dtype=float),
        sample["Fan"].to_numpy(dtype=float),
    )


def _build_legacy_training_data(source_data: pd.DataFrame, days: int) -> pd.DataFrame:
    samples_per_day = int(round(24 * 3600 / SAMPLE_SECONDS))
    if days <= 4:
        n_rows = days * samples_per_day
        return source_data.iloc[:n_rows].reset_index(drop=True).loc[:, MODEL_COLS]

    day_chunks = [
        source_data.iloc[start : start + samples_per_day].copy().reset_index(drop=True)
        for start in range(0, len(source_data), samples_per_day)
    ]
    day_chunks = [chunk for chunk in day_chunks if len(chunk) > 0]
    if not day_chunks:
        raise ValueError("source data is empty")

    out: list[pd.DataFrame] = []
    rng = np.random.default_rng(10_000)
    first_timestamp = pd.Timestamp(source_data["Timestamp"].iloc[0])
    for day in range(days):
        chunk = day_chunks[day % len(day_chunks)].copy()
        old_start = pd.Timestamp(chunk["Timestamp"].iloc[0])
        new_start = first_timestamp + pd.to_timedelta(day * 24, unit="h")
        chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"]) - old_start + new_start
        out.append(chunk)
    return pd.concat(out, ignore_index=True).loc[:, MODEL_COLS]


def _build_real_training_data(source_data: pd.DataFrame, days: int, seed: int) -> pd.DataFrame:
    if source_data.empty:
        raise ValueError("source data is empty")

    samples_per_day = int(round(24 * 3600 / SAMPLE_SECONDS))
    rng = np.random.default_rng(seed + 10_000)

    actuator_template = source_data.iloc[: min(len(source_data), samples_per_day)].copy().reset_index(drop=True)
    temp_base = float(source_data["Temperature"].median())
    humi_base = float(source_data["Humidity"].median())
    soil0 = float(source_data["Soil_Moisture"].median())

    drip_template, mist_template, fan_template = _extract_actuator_template(actuator_template)
    session_offsets = (7 * 3600, 11 * 3600 + 30 * 60, 15 * 3600, 20 * 3600)
    session_len = len(actuator_template)

    out: list[pd.DataFrame] = []
    start_day = pd.Timestamp(source_data["Timestamp"].iloc[0]).normalize()

    for day in range(days):
        day_start = start_day + pd.to_timedelta(day, unit="D")
        index = pd.date_range(day_start, periods=samples_per_day, freq=f"{SAMPLE_SECONDS}s")

        temp_bias = (temp_base - 28.0) * 0.35 + rng.normal(0.0, 0.4)
        humi_bias = (humi_base - 75.0) * 0.45 + rng.normal(0.0, 1.0)
        light_scale = rng.uniform(0.90, 1.08)

        temp, humi, light = base_environment(
            index,
            rng,
            temp_bias=temp_bias,
            humi_bias=humi_bias,
            light_scale=light_scale,
        )

        day_drip = np.zeros(samples_per_day, dtype=float)
        day_mist = np.zeros(samples_per_day, dtype=float)
        day_fan = np.zeros(samples_per_day, dtype=float)

        for start_sec in session_offsets:
            start_idx = int(start_sec // SAMPLE_SECONDS)
            end_idx = min(samples_per_day, start_idx + session_len)
            n = end_idx - start_idx
            if n <= 0:
                continue
            day_drip[start_idx:end_idx] = drip_template[:n]
            day_mist[start_idx:end_idx] = mist_template[:n]
            day_fan[start_idx:end_idx] = fan_template[:n]

        soil = soil_response(temp, humi, light, day_drip, day_mist, day_fan, soil0, rng)

        day_df = pd.DataFrame(
            {
                "Timestamp": index,
                "Temperature": temp,
                "Humidity": humi,
                "Light": light,
                "Soil_Moisture": soil,
                "Drip": day_drip,
                "Mist": day_mist,
                "Fan": day_fan,
            }
        )

        # Keep soil around the real session baseline while allowing small day-to-day variation.
        day_df["Soil_Moisture"] = (day_df["Soil_Moisture"] + rng.normal(0.0, 0.03)).clip(0.0, 100.0)
        day_df["Temperature"] = day_df["Temperature"].clip(15.0, 45.0)
        day_df["Humidity"] = day_df["Humidity"].clip(30.0, 100.0)
        day_df["Light"] = day_df["Light"].clip(0.0, 1200.0)

        out.append(day_df)

    return pd.concat(out, ignore_index=True).loc[:, MODEL_COLS]


def build_augmented_training_data(source_data: pd.DataFrame, days: int, seed: int, source: DataSource = "legacy") -> pd.DataFrame:
    if days <= 0:
        raise ValueError("days must be greater than 0")

    normalized_source = normalize_source(source)
    source_data = source_data.copy()
    source_data["Timestamp"] = pd.to_datetime(source_data["Timestamp"])
    source_data = source_data.sort_values("Timestamp").reset_index(drop=True)

    if normalized_source == "legacy":
        legacy_data = load_source_data("legacy")
        legacy_data["Timestamp"] = pd.to_datetime(legacy_data["Timestamp"])
        legacy_data = legacy_data.sort_values("Timestamp").reset_index(drop=True)
        return _build_legacy_training_data(legacy_data, days)

    return _build_real_training_data(source_data, days, seed)
