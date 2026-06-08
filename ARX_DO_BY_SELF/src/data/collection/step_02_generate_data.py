from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from data.collection.step_00_data_io import (
    ACTUATOR_COLS,
    MODEL_COLS,
    SAMPLE_SECONDS,
    SENSOR_RANGES,
    format_model_data,
    inject_collection_artifacts,
)


# Đổi Timestamp thành giờ trong ngày.
def hour_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy() + index.minute.to_numpy() / 60.0 + index.second.to_numpy() / 3600.0


# Lấy các đoạn liên tục mà thiết bị đang bật.
def active_blocks(data: pd.DataFrame, actuator: str) -> list[tuple[int, int]]:
    state = data[actuator].to_numpy(dtype=float) >= 0.5
    blocks: list[tuple[int, int]] = []
    start: int | None = None

    for i, active in enumerate(state):
        if active and start is None:
            start = i
        elif not active and start is not None:
            blocks.append((start, i - 1))
            start = None

    if start is not None:
        blocks.append((start, len(state) - 1))
    return blocks


# Ước lượng Mist/Fan/Drip làm sensor thay đổi bao nhiêu từ data đã thu.
def estimate_actuator_effects(data: pd.DataFrame) -> dict[str, float]:
    effects = {
        "mist_humidity_gain": 28.0,
        "fan_humidity_drop": 25.0,
        "drip_soil_gain": 6.0,
    }

    mist_gains: list[float] = []
    for start, end in active_blocks(data, "Mist"):
        before = float(data.loc[max(0, start - 1), "Humidity"])
        window = data.loc[start : min(len(data) - 1, end + 24), "Humidity"]
        mist_gains.append(float(window.max() - before))
    if mist_gains:
        effects["mist_humidity_gain"] = float(np.clip(np.median(mist_gains), 8.0, 38.0))

    fan_drops: list[float] = []
    for start, end in active_blocks(data, "Fan"):
        before = float(data.loc[max(0, start - 1), "Humidity"])
        window = data.loc[start : min(len(data) - 1, end + 24), "Humidity"]
        fan_drops.append(float(before - window.min()))
    if fan_drops:
        effects["fan_humidity_drop"] = float(np.clip(np.median(fan_drops), 6.0, 36.0))

    drip_gains: list[float] = []
    for start, end in active_blocks(data, "Drip"):
        before = float(data.loc[max(0, start - 1), "Soil_Moisture"])
        window = data.loc[start : min(len(data) - 1, end + 36), "Soil_Moisture"]
        gain = float(window.max() - before)
        if before >= 25.0:
            drip_gains.append(gain)
    if drip_gains:
        effects["drip_soil_gain"] = float(np.clip(np.median(drip_gains), 3.0, 12.0))

    return effects


# Phân tích data sạch để lấy nền môi trường, nền đất và mẫu thiết bị.
def analyze_collected_data(source_data: pd.DataFrame) -> dict[str, object]:
    if source_data.empty:
        raise ValueError("source data is empty")

    data = source_data.copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data = data.sort_values("Timestamp").reset_index(drop=True)

    sensor_profile = {
        col: {
            "median": float(data[col].median()),
            "q10": float(data[col].quantile(0.10)),
            "q90": float(data[col].quantile(0.90)),
        }
        for col in SENSOR_RANGES
    }
    actuator_template = data.loc[:, ACTUATOR_COLS].copy().reset_index(drop=True)

    return {
        "data": data,
        "sensor": sensor_profile,
        "actuator_template": actuator_template,
        "effects": estimate_actuator_effects(data),
        "start_day": pd.Timestamp(data["Timestamp"].iloc[0]).normalize(),
        "soil0": sensor_profile["Soil_Moisture"]["median"],
    }


# Sinh nền Temperature, Humidity, Light theo profile phân tích từ data sạch.
def base_environment(
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
    profile: dict[str, object],
    temp_bias: float | np.ndarray = 0.0,
    humi_bias: float | np.ndarray = 0.0,
    light_scale: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor = profile["sensor"]
    temp_stat = sensor["Temperature"]
    humi_stat = sensor["Humidity"]
    light_stat = sensor["Light"]

    temp_base = float(temp_stat["median"])
    temp_gain = max(1.5, float(temp_stat["q90"]) - float(temp_stat["q10"]))
    humi_base = float(humi_stat["median"])
    humi_drop = max(4.0, float(humi_stat["q90"]) - float(humi_stat["q10"]))
    light_night = max(0.0, float(light_stat["q10"]))
    light_gain = max(80.0, float(light_stat["q90"]) - light_night)

    hour = hour_of_day(index)
    daylight = np.maximum(0.0, np.sin((hour - 6.0) / 12.0 * np.pi))
    low_freq = np.cumsum(rng.normal(0.0, 0.006, len(index)))
    low_freq = low_freq - np.linspace(low_freq[0], low_freq[-1], len(index))

    temp = (
        temp_base
        + temp_gain * (daylight - 0.45)
        + 0.8 * np.sin((hour - 13.5) / 24.0 * 2.0 * np.pi)
        + temp_bias
        + low_freq
        + rng.normal(0.0, 0.08, len(index))
    )
    humi = (
        humi_base
        - humi_drop * (daylight - 0.45)
        - 0.65 * (temp - 28.0)
        + humi_bias
        + rng.normal(0.0, 0.28, len(index))
    )
    light = light_night + light_gain * np.power(daylight, 1.25) * light_scale + rng.normal(0.0, 10.0, len(index))

    return (
        np.clip(temp, 18.0, 42.0),
        np.clip(humi, 38.0, 98.0),
        np.clip(light, 0.0, 1050.0),
    )


# Sinh Humidity có phản ứng theo Mist và Fan đã đo từ data thật.
def humidity_response(
    base_humi: np.ndarray,
    mist: np.ndarray,
    fan: np.ndarray,
    rng: np.random.Generator,
    effects: dict[str, float],
) -> np.ndarray:
    humi = np.zeros(len(base_humi), dtype=float)
    humi[0] = base_humi[0]

    mist_rate = float(np.clip(effects["mist_humidity_gain"] / 22.0, 0.35, 1.75))
    fan_rate = float(np.clip(effects["fan_humidity_drop"] / 28.0, 0.20, 1.30))

    for i in range(1, len(base_humi)):
        relax = 0.035 * (base_humi[i] - humi[i - 1])
        mist_push = mist_rate * max(0.0, 96.0 - humi[i - 1]) / 22.0 * mist[i - 1]
        fan_pull = fan_rate * max(0.0, humi[i - 1] - base_humi[i]) / 18.0 * fan[i - 1]
        humi[i] = humi[i - 1] + relax + mist_push - fan_pull + rng.normal(0.0, 0.08)

    return np.clip(humi, 30.0, 100.0)


# Sinh phản ứng Soil_Moisture theo môi trường và thiết bị.
def soil_response(
    temp: np.ndarray,
    humi: np.ndarray,
    light: np.ndarray,
    drip: np.ndarray,
    mist: np.ndarray,
    fan: np.ndarray,
    soil0: float,
    rng: np.random.Generator,
    effects: dict[str, float],
) -> np.ndarray:
    soil_true = np.zeros(len(temp), dtype=float)
    soil_meas = np.zeros(len(temp), dtype=float)
    soil_true[0] = soil0
    soil_meas[0] = soil0 + rng.normal(0.0, 0.04)
    drip_scale = float(np.clip(effects["drip_soil_gain"] / 6.0, 0.7, 2.2))

    for i in range(1, len(temp)):
        vpd_proxy = max(0.0, temp[i - 1] - 22.0) * max(0.0, 100.0 - humi[i - 1]) / 100.0
        evap = 0.00050 + 0.00033 * vpd_proxy + 0.00000055 * light[i - 1] + 0.00068 * fan[i - 1]
        if soil_true[i - 1] < 54.5:
            evap *= 0.58

        water = 0.0
        for lag, gain in ((2, 0.0090), (3, 0.0080), (6, 0.0060), (10, 0.0040)):
            if i - lag >= 0:
                water += drip_scale * gain * drip[i - lag]
        if i - 2 >= 0:
            water += 0.0018 * mist[i - 2]

        drainage = 0.010 * max(0.0, soil_true[i - 1] - 64.0)
        slow_balance = 0.00025 * (57.0 - soil_true[i - 1])
        soil_true[i] = soil_true[i - 1] + water - evap - drainage + slow_balance + rng.normal(0.0, 0.003)
        soil_true[i] = float(np.clip(soil_true[i], 0.0, 100.0))

        raw_sensor = soil_true[i] + rng.normal(0.0, 0.045)
        soil_meas[i] = float(np.clip(0.48 * soil_meas[i - 1] + 0.52 * raw_sensor, 0.0, 100.0))

    return soil_meas


# Dựng lịch bật/tắt thiết bị trong 1 ngày theo giờ thu thật.
def _daily_actuator_template(data: pd.DataFrame, samples_per_day: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    drip = np.zeros(samples_per_day, dtype=float)
    mist = np.zeros(samples_per_day, dtype=float)
    fan = np.zeros(samples_per_day, dtype=float)

    timestamp = pd.to_datetime(data["Timestamp"])
    second_of_day = timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second
    step_index = (second_of_day // SAMPLE_SECONDS).astype(int).clip(0, samples_per_day - 1)

    for idx, row in zip(step_index, data.loc[:, ["Drip", "Mist", "Fan"]].to_numpy(dtype=float)):
        drip[idx] = max(drip[idx], row[0])
        mist[idx] = max(mist[idx], row[1])
        fan[idx] = max(fan[idx], row[2])

    return drip, mist, fan


# Thêm lệch nhẹ lịch thiết bị giữa các ngày để data không bị copy y hệt.
def _shift_actuator_template(template: np.ndarray, shift_steps: int) -> np.ndarray:
    shifted = np.zeros_like(template)
    if shift_steps >= 0:
        shifted[shift_steps:] = template[: len(template) - shift_steps]
    else:
        shifted[:shift_steps] = template[-shift_steps:]
    return shifted


# Sinh data train nhiều ngày từ profile phân tích data sạch.
def _build_training_data(source_data: pd.DataFrame, days: int, seed: int) -> pd.DataFrame:
    samples_per_day = int(round(24 * 3600 / SAMPLE_SECONDS))
    rng = np.random.default_rng(seed + 10_000)
    profile = analyze_collected_data(source_data)

    sensor = profile["sensor"]
    temp_base = float(sensor["Temperature"]["median"])
    humi_base = float(sensor["Humidity"]["median"])
    soil0 = float(profile["soil0"])
    drip_template, mist_template, fan_template = _daily_actuator_template(profile["data"], samples_per_day)

    out: list[pd.DataFrame] = []
    start_day = profile["start_day"]
    current_soil = soil0

    for day in range(days):
        day_start = start_day + pd.to_timedelta(day, unit="D")
        index = pd.date_range(day_start, periods=samples_per_day, freq=f"{SAMPLE_SECONDS}s")

        temp_bias = (temp_base - 28.0) * 0.35 + rng.normal(0.0, 0.4)
        humi_bias = rng.normal(0.0, 1.0)
        light_scale = rng.uniform(0.90, 1.08)
        temp, humi, light = base_environment(index, rng, profile, temp_bias, humi_bias, light_scale)

        shift_steps = int(rng.integers(-12, 13))
        day_drip = _shift_actuator_template(drip_template, shift_steps)
        day_mist = _shift_actuator_template(mist_template, shift_steps)
        day_fan = _shift_actuator_template(fan_template, shift_steps)

        humi = humidity_response(humi, day_mist, day_fan, rng, profile["effects"])
        soil = soil_response(temp, humi, light, day_drip, day_mist, day_fan, current_soil, rng, profile["effects"])
        current_soil = float(soil[-1])
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

        # Giữ Soil_Moisture quanh nền raw, chỉ thêm dao động nhỏ giữa các ngày.
        day_df["Soil_Moisture"] = (day_df["Soil_Moisture"] + rng.normal(0.0, 0.03)).clip(0.0, 100.0)
        day_df["Temperature"] = day_df["Temperature"].clip(15.0, 45.0)
        day_df["Humidity"] = day_df["Humidity"].clip(30.0, 100.0)
        day_df["Light"] = day_df["Light"].clip(0.0, 1200.0)
        out.append(day_df)

    return format_model_data(pd.concat(out, ignore_index=True))


# Chuẩn bị raw clean rồi sinh data train cuối cùng.
def build_training_data(source_data: pd.DataFrame, days: int, seed: int) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("days must be greater than 0")

    source_data = source_data.copy()
    source_data["Timestamp"] = pd.to_datetime(source_data["Timestamp"])
    source_data = source_data.sort_values("Timestamp").reset_index(drop=True)
    return _build_training_data(source_data, days, seed)


# Sinh các phiên thu đại diện từ data thật rồi ghi vào _01_data.
def build_collection_session_files(output_dir: Path, source_data: pd.DataFrame, seed: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    one_day = build_training_data(source_data, days=1, seed=seed)
    one_day["Timestamp"] = pd.to_datetime(one_day["Timestamp"])
    day0 = pd.Timestamp(one_day["Timestamp"].iloc[0]).normalize()

    windows = (
        ("01_morning_raw.csv", "morning_anchor", 7.0, 9.0),
        ("02_noon_raw.csv", "noon_anchor", 11.5, 13.5),
        ("03_afternoon_raw.csv", "afternoon_anchor", 15.0, 17.0),
        ("04_night_raw.csv", "night_anchor", 20.0, 22.0),
    )

    paths: list[Path] = []
    for file_name, artifact_name, start_hour, end_hour in windows:
        start = day0 + pd.to_timedelta(start_hour, unit="h")
        end = day0 + pd.to_timedelta(end_hour, unit="h")
        session = one_day[(one_day["Timestamp"] >= start) & (one_day["Timestamp"] < end)].reset_index(drop=True)
        raw_session = inject_collection_artifacts(session, artifact_name)
        path = output_dir / file_name
        format_model_data(raw_session).to_csv(path, index=False)
        paths.append(path)

    return paths


# Tạo data train bằng cách lấy lại các ngày từ data 5s chuẩn.
def build_reference_training_data(reference_data: pd.DataFrame, days: int) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("days must be greater than 0")

    samples_per_day = int(round(24 * 3600 / SAMPLE_SECONDS))
    if days <= 4:
        n_rows = days * samples_per_day
        return format_model_data(reference_data.iloc[:n_rows].reset_index(drop=True))

    day_chunks = [
        reference_data.iloc[start : start + samples_per_day].copy().reset_index(drop=True)
        for start in range(0, len(reference_data), samples_per_day)
    ]
    day_chunks = [chunk for chunk in day_chunks if len(chunk) > 0]
    if not day_chunks:
        raise ValueError("reference data is empty")

    out: list[pd.DataFrame] = []
    first_timestamp = pd.Timestamp(reference_data["Timestamp"].iloc[0])
    for day in range(days):
        chunk = day_chunks[day % len(day_chunks)].copy()
        old_start = pd.Timestamp(chunk["Timestamp"].iloc[0])
        new_start = first_timestamp + pd.to_timedelta(day * 24, unit="h")
        chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"]) - old_start + new_start
        out.append(chunk)
    return format_model_data(pd.concat(out, ignore_index=True))
