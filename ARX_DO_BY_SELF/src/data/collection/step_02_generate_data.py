from __future__ import annotations

import numpy as np
import pandas as pd

from data.collection.step_00_data_io import ACTUATOR_COLS, MODEL_COLS, SAMPLE_SECONDS, SENSOR_RANGES, format_model_data


# Đổi Timestamp thành giờ trong ngày.
def hour_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy() + index.minute.to_numpy() / 60.0 + index.second.to_numpy() / 3600.0


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
) -> np.ndarray:
    soil_true = np.zeros(len(temp), dtype=float)
    soil_meas = np.zeros(len(temp), dtype=float)
    soil_true[0] = soil0
    soil_meas[0] = soil0 + rng.normal(0.0, 0.04)

    for i in range(1, len(temp)):
        vpd_proxy = max(0.0, temp[i - 1] - 22.0) * max(0.0, 100.0 - humi[i - 1]) / 100.0
        evap = 0.00050 + 0.00033 * vpd_proxy + 0.00000055 * light[i - 1] + 0.00068 * fan[i - 1]
        if soil_true[i - 1] < 54.5:
            evap *= 0.58

        water = 0.0
        for lag, gain in ((2, 0.0090), (3, 0.0080), (6, 0.0060), (10, 0.0040)):
            if i - lag >= 0:
                water += gain * drip[i - lag]
        if i - 2 >= 0:
            water += 0.0018 * mist[i - 2]

        drainage = 0.010 * max(0.0, soil_true[i - 1] - 64.0)
        slow_balance = 0.00025 * (57.0 - soil_true[i - 1])
        soil_true[i] = soil_true[i - 1] + water - evap - drainage + slow_balance + rng.normal(0.0, 0.003)
        soil_true[i] = float(np.clip(soil_true[i], 40.0, 82.0))

        raw_sensor = soil_true[i] + rng.normal(0.0, 0.045)
        soil_meas[i] = float(np.clip(0.48 * soil_meas[i - 1] + 0.52 * raw_sensor, 0.0, 100.0))

    return soil_meas


# Lấy mẫu bật/tắt thiết bị từ data sạch đã phân tích.
def _extract_actuator_template(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample = df.loc[:, ["Drip", "Mist", "Fan"]].copy()
    if sample.empty:
        raise ValueError("source data is empty")
    return (
        sample["Drip"].to_numpy(dtype=float),
        sample["Mist"].to_numpy(dtype=float),
        sample["Fan"].to_numpy(dtype=float),
    )


# Sinh data train nhiều ngày từ profile phân tích data sạch.
def _build_training_data(source_data: pd.DataFrame, days: int, seed: int) -> pd.DataFrame:
    samples_per_day = int(round(24 * 3600 / SAMPLE_SECONDS))
    rng = np.random.default_rng(seed + 10_000)
    profile = analyze_collected_data(source_data)

    actuator_template = profile["actuator_template"].iloc[: min(len(profile["actuator_template"]), samples_per_day)]
    sensor = profile["sensor"]
    temp_base = float(sensor["Temperature"]["median"])
    humi_base = float(sensor["Humidity"]["median"])
    soil0 = float(profile["soil0"])
    drip_template, mist_template, fan_template = _extract_actuator_template(actuator_template)
    session_offsets = (7 * 3600, 11 * 3600 + 30 * 60, 15 * 3600, 20 * 3600)
    session_len = len(actuator_template)

    out: list[pd.DataFrame] = []
    start_day = profile["start_day"]

    for day in range(days):
        day_start = start_day + pd.to_timedelta(day, unit="D")
        index = pd.date_range(day_start, periods=samples_per_day, freq=f"{SAMPLE_SECONDS}s")

        temp_bias = (temp_base - 28.0) * 0.35 + rng.normal(0.0, 0.4)
        humi_bias = (humi_base - 75.0) * 0.45 + rng.normal(0.0, 1.0)
        light_scale = rng.uniform(0.90, 1.08)
        temp, humi, light = base_environment(index, rng, profile, temp_bias, humi_bias, light_scale)

        day_drip = np.zeros(samples_per_day, dtype=float)
        day_mist = np.zeros(samples_per_day, dtype=float)
        day_fan = np.zeros(samples_per_day, dtype=float)

        for start_sec in session_offsets:
            start_idx = int(start_sec // SAMPLE_SECONDS)
            end_idx = min(samples_per_day, start_idx + session_len)
            n = end_idx - start_idx
            if n > 0:
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
