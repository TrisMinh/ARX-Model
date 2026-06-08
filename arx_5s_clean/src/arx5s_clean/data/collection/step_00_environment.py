from __future__ import annotations

import numpy as np
import pandas as pd


def hour_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy() + index.minute.to_numpy() / 60.0 + index.second.to_numpy() / 3600.0


def base_environment(
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
    temp_bias: float | np.ndarray = 0.0,
    humi_bias: float | np.ndarray = 0.0,
    light_scale: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hour = hour_of_day(index)
    daylight = np.maximum(0.0, np.sin((hour - 6.0) / 12.0 * np.pi))
    low_freq = np.cumsum(rng.normal(0.0, 0.006, len(index)))
    low_freq = low_freq - np.linspace(low_freq[0], low_freq[-1], len(index))

    temp = (
        25.2
        + 6.1 * daylight
        + 0.8 * np.sin((hour - 13.5) / 24.0 * 2.0 * np.pi)
        + temp_bias
        + low_freq
        + rng.normal(0.0, 0.08, len(index))
    )
    humi = (
        85.0
        - 18.0 * daylight
        - 0.65 * (temp - 28.0)
        + humi_bias
        + rng.normal(0.0, 0.28, len(index))
    )
    light = 18.0 + 850.0 * np.power(daylight, 1.25) * light_scale + rng.normal(0.0, 10.0, len(index))

    return (
        np.clip(temp, 18.0, 42.0),
        np.clip(humi, 38.0, 98.0),
        np.clip(light, 0.0, 1050.0),
    )


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
