from __future__ import annotations

import numpy as np
import pandas as pd

from arx5s_clean.config import ExperimentConfig


def phase_for_day(day: int) -> str:
    if day < 2:
        return "commissioning_rule_based"
    if day < 8:
        return "identification_safe_excitation"
    return "deployment_validation"


def _planned_excitation(cfg: ExperimentConfig, rng: np.random.Generator, n_rows: int) -> dict[str, np.ndarray]:
    samples_per_day = int(round(24 * 3600 / cfg.sampling_seconds))
    planned_drip = np.zeros(n_rows, dtype=float)
    planned_mist = np.zeros(n_rows, dtype=float)
    planned_fan = np.zeros(n_rows, dtype=float)
    label = np.full(n_rows, "none", dtype=object)

    for day in range(cfg.days):
        phase = phase_for_day(day)
        if phase == "commissioning_rule_based":
            continue

        base = day * samples_per_day
        test_phase = phase == "deployment_validation"

        drip_prob = 0.90 if not test_phase else 0.72
        for center_hour in (7.0, 10.4, 13.7, 17.2, 20.5):
            if rng.random() < drip_prob:
                start_hour = center_hour + rng.normal(0.0, 0.18)
                start = base + int(np.clip(round(start_hour * 3600 / cfg.sampling_seconds), 0, samples_per_day - 1))
                duration = int(rng.integers(7, 23 if not test_phase else 17))
                end = min(base + samples_per_day, start + duration)
                planned_drip[start:end] = 1.0
                label[start:end] = "planned_drip_excitation"

        fan_prob = 0.76 if not test_phase else 0.54
        for center_hour in (8.2, 11.6, 14.4, 18.1):
            if rng.random() < fan_prob:
                start_hour = center_hour + rng.normal(0.0, 0.22)
                start = base + int(np.clip(round(start_hour * 3600 / cfg.sampling_seconds), 0, samples_per_day - 1))
                duration = int(rng.integers(4, 13))
                end = min(base + samples_per_day, start + duration)
                planned_fan[start:end] = 1.0
                label[start:end] = "planned_fan_ventilation_probe"

        mist_prob = 0.38 if not test_phase else 0.24
        for center_hour in (12.2, 15.0):
            if rng.random() < mist_prob:
                start_hour = center_hour + rng.normal(0.0, 0.20)
                start = base + int(np.clip(round(start_hour * 3600 / cfg.sampling_seconds), 0, samples_per_day - 1))
                duration = int(rng.integers(2, 8))
                end = min(base + samples_per_day, start + duration)
                planned_mist[start:end] = 1.0
                label[start:end] = "planned_mist_probe"

    return {
        "planned_drip": planned_drip,
        "planned_mist": planned_mist,
        "planned_fan": planned_fan,
        "label": label,
    }


def generate_greenhouse_data(cfg: ExperimentConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    samples_per_day = int(round(24 * 3600 / cfg.sampling_seconds))
    n_rows = cfg.days * samples_per_day
    t = np.arange(n_rows)
    timestamp = pd.date_range("2026-01-01", periods=n_rows, freq=f"{cfg.sampling_seconds}s")
    hour = (t % samples_per_day) / samples_per_day * 24.0
    day_index = t // samples_per_day
    phase = np.asarray([phase_for_day(int(day)) for day in day_index])
    volume_m3 = cfg.greenhouse_volume_m3

    day_temp_offset = rng.normal(0.0, 1.1, cfg.days + 1)
    day_humi_offset = rng.normal(0.0, 3.0, cfg.days + 1)
    day_cloud = rng.uniform(0.62, 1.16, cfg.days + 1)
    daylight = np.maximum(0.0, np.sin((hour - 6.0) / 12.0 * np.pi))
    late_cloud = 1.0 - 0.28 * np.maximum(0.0, np.sin((hour - 14.2) / 4.5 * np.pi)) * (day_index % 3 == 1)
    light_out = np.clip(
        18.0 + 920.0 * daylight * day_cloud[day_index] * late_cloud + rng.normal(0.0, 18.0, n_rows),
        0.0,
        1250.0,
    )
    light_in = np.clip(0.78 * light_out + 18.0 * daylight + rng.normal(0.0, 8.0, n_rows), 0.0, 1050.0)

    temp_wave = np.sin((hour - 8.0) / 24.0 * 2.0 * np.pi)
    humi_wave = np.sin((hour - 8.2) / 24.0 * 2.0 * np.pi)
    deployment_front = np.clip((day_index.astype(float) - 7.0) / 4.0, 0.0, 1.0)
    outdoor_temp = 27.0 + 5.0 * temp_wave + day_temp_offset[day_index] + rng.normal(0.0, 0.22, n_rows)
    outdoor_temp = outdoor_temp + cfg.front_temp_gain * deployment_front + cfg.front_temp_step * (day_index >= 10)
    outdoor_humi = np.clip(
        74.0
        - 11.0 * humi_wave
        + day_humi_offset[day_index]
        - cfg.front_humi_drop * deployment_front
        - cfg.front_humi_step * (day_index >= 10)
        + rng.normal(0.0, 0.85, n_rows),
        35.0,
        98.0,
    )

    planned = _planned_excitation(cfg, rng, n_rows)
    drip = np.zeros(n_rows, dtype=float)
    mist = np.zeros(n_rows, dtype=float)
    fan = np.zeros(n_rows, dtype=float)
    source = np.full(n_rows, "none", dtype=object)
    safety_override = np.zeros(n_rows, dtype=int)
    wet_block = np.zeros(n_rows, dtype=int)
    air_exchange_rate = np.zeros(n_rows, dtype=float)

    temp_true = np.zeros(n_rows, dtype=float)
    humi_true = np.zeros(n_rows, dtype=float)
    temp_in = np.zeros(n_rows, dtype=float)
    humi_in = np.zeros(n_rows, dtype=float)
    soil_true = np.zeros(n_rows, dtype=float)
    soil_meas = np.zeros(n_rows, dtype=float)

    temp_true[:5] = outdoor_temp[:5] + 0.6
    humi_true[:5] = outdoor_humi[:5] - 1.0
    temp_in[:5] = temp_true[:5] + rng.normal(0.0, 0.08, 5)
    humi_in[:5] = humi_true[:5] + rng.normal(0.0, 0.18, 5)
    soil_true[:5] = (59.2, 59.14, 59.08, 59.02, 58.96)
    soil_meas[:5] = soil_true[:5] + rng.normal(0.0, 0.08, 5)
    rescue_hold = 0

    for idx in range(5, n_rows):
        low_sp = cfg.soil_low_sp
        high_sp = cfg.soil_high_sp

        rescue = soil_meas[idx - 1] < (low_sp - 0.8)
        too_wet = soil_meas[idx - 1] > (high_sp + 0.7)
        if rescue_hold > 0:
            drip[idx] = 1.0
            source[idx] = "safety_rescue_hold"
            rescue_hold -= 1
        elif rescue:
            drip[idx] = 1.0
            source[idx] = "safety_rescue"
            safety_override[idx] = 1
            rescue_hold = 2
        elif too_wet and planned["planned_drip"][idx] > 0.5:
            drip[idx] = 0.0
            wet_block[idx] = 1
            safety_override[idx] = 1
            source[idx] = "planned_drip_blocked_wet"
        elif planned["planned_drip"][idx] > 0.5:
            drip[idx] = 1.0
            source[idx] = "planned_drip_excitation"

        fan_rule = (temp_in[idx - 1] > 30.5) or (humi_in[idx - 1] > 88.0)
        mist_rule = (temp_in[idx - 1] > 31.0) and (humi_in[idx - 1] < 68.0) and (9.0 <= hour[idx] <= 16.0)
        fan[idx] = 1.0 if fan_rule or planned["planned_fan"][idx] > 0.5 else 0.0
        mist[idx] = 1.0 if mist_rule or planned["planned_mist"][idx] > 0.5 else 0.0
        if source[idx] == "none" and planned["label"][idx] != "none":
            source[idx] = planned["label"][idx]

        exchange = 0.012 + 0.34 * fan[idx] + 0.006 * np.maximum(0.0, daylight[idx])
        air_exchange_rate[idx] = exchange

        solar_heat = 0.00175 * light_in[idx]
        temp_true[idx] = (
            (1.0 - exchange) * temp_true[idx - 1]
            + exchange * outdoor_temp[idx]
            + solar_heat
            - 0.34 * mist[idx]
            - 0.12 * fan[idx]
            + rng.normal(0.0, 0.08)
        )
        humi_true[idx] = (
            (1.0 - exchange) * humi_true[idx - 1]
            + exchange * outdoor_humi[idx]
            + 5.8 * mist[idx]
            - 1.1 * fan[idx]
            + 0.018 * max(0.0, soil_true[idx - 1] - 54.0)
            + rng.normal(0.0, 0.20)
        )
        temp_true[idx] = float(np.clip(temp_true[idx], 15.0, 43.0))
        humi_true[idx] = float(np.clip(humi_true[idx], 34.0, 100.0))

        temp_in[idx] = float(np.clip(0.90 * temp_in[idx - 1] + 0.10 * temp_true[idx] + rng.normal(0.0, 0.06), 15.0, 43.0))
        humi_in[idx] = float(np.clip(0.91 * humi_in[idx - 1] + 0.09 * humi_true[idx] + rng.normal(0.0, 0.18), 34.0, 100.0))

        vpd_proxy = max(0.0, temp_true[idx - 3] - 22.0) * max(0.0, 100.0 - humi_true[idx - 3]) / 100.0
        vent_dry = fan[idx - 2] * max(0.0, humi_true[idx - 2] - outdoor_humi[idx - 2]) * 0.00062
        vent_heat = fan[idx - 2] * max(0.0, outdoor_temp[idx - 2] - temp_true[idx - 2]) * 0.00125
        evap = (
            0.0030
            + 0.00185 * vpd_proxy
            + 0.000010 * light_in[idx - 3]
            + vent_dry
            + vent_heat
            + 0.0011 * fan[idx - 5]
        )
        evap = max(0.0010, evap)
        water = (
            0.092 * drip[idx - 2]
            + 0.080 * drip[idx - 3]
            + 0.066 * drip[idx - 5]
            + 0.040 * drip[idx - 8]
            + 0.022 * drip[idx - 12]
        )
        mist_to_soil = 0.0045 * mist[idx - 2]
        drainage = 0.020 * max(0.0, soil_true[idx - 1] - 65.3)
        dry_slowdown = 0.52 if soil_true[idx - 1] < 53.5 else 1.0
        small_volume_factor = 0.045 / max(volume_m3, 1e-6)
        soil_true[idx] = (
            soil_true[idx - 1]
            + small_volume_factor * (water + mist_to_soil)
            - dry_slowdown * evap
            - drainage
            + 0.0013 * (58.0 - soil_true[idx - 1])
            + rng.normal(0.0, 0.012)
        )
        soil_true[idx] = float(np.clip(soil_true[idx], 35.0, 85.0))

        raw_sensor = soil_true[idx] + rng.normal(0.0, 0.07)
        soil_meas[idx] = float(np.clip(0.46 * soil_meas[idx - 1] + 0.54 * raw_sensor, 0.0, 100.0))

    out = pd.DataFrame(
        {
            "Timestamp": timestamp,
            "Day_Index": day_index,
            "Hour": hour,
            "Protocol_Phase": phase,
            "Greenhouse_Volume_m3": np.full(n_rows, volume_m3),
            "Temperature_Air_True": temp_true,
            "Humidity_Air_True": humi_true,
            "Temperature_In": temp_in,
            "Humidity_In": humi_in,
            "Light_In": light_in,
            "Air_Exchange_Rate": air_exchange_rate,
            "Soil_Moisture_True": soil_true,
            "Soil_Moisture": soil_meas,
            "Soil_Low_SP": np.full(n_rows, cfg.soil_low_sp),
            "Soil_High_SP": np.full(n_rows, cfg.soil_high_sp),
            "Drip": drip,
            "Mist": mist,
            "Fan": fan,
            "Planned_Drip": planned["planned_drip"],
            "Planned_Mist": planned["planned_mist"],
            "Planned_Fan": planned["planned_fan"],
            "Command_Source": source,
            "Safety_Override": safety_override,
            "Wet_Block": wet_block,
        }
    )
    out["Sampling_Seconds"] = cfg.sampling_seconds
    out["Generated_From"] = "native_5s_physical_protocol"
    return out

