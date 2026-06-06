from __future__ import annotations

import numpy as np
import pandas as pd


TRUE_PARAMS_REALISTIC = {
    "a1": 0.91,
    "a2": 0.05,
    "b_out_temp": -0.055,
    "b_out_humi": 0.020,
    "b_out_light": -0.030,
    "b_drip_1": 1.80,
    "b_drip_2": 1.05,
    "b_mist": 0.35,
    "b_fan": -0.55,
    "noise_sigma": 0.09,
}


def _season_signal(day_index: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * (day_index / 365.0 - 0.15))


def get_true_params() -> dict[str, float]:
    return dict(TRUE_PARAMS_REALISTIC)


def generate_greenhouse_data_realistic(
    days: int = 365,
    sampling_seconds: int = 300,
    seed: int = 42,
    start_date: str = "2025-01-01",
) -> tuple[pd.DataFrame, dict[str, float]]:
    if days <= 0:
        raise ValueError("days must be positive")
    if sampling_seconds <= 0:
        raise ValueError("sampling_seconds must be positive")

    rng = np.random.default_rng(seed)
    samples_per_day = int(round(24 * 3600 / sampling_seconds))
    total_samples = int(days * samples_per_day)

    time_index = np.arange(total_samples)
    timestamps = pd.date_range(start_date, periods=total_samples, freq=f"{sampling_seconds}s")
    hour = (time_index % samples_per_day) / samples_per_day * 24.0
    day_index = time_index // samples_per_day
    season = _season_signal(day_index)

    outside_temp = 28.0 + 6.0 * np.sin((hour - 14.0) / 24.0 * 2.0 * np.pi) + 3.0 * season + rng.normal(0.0, 0.6, total_samples)
    outside_humidity = 70.0 - 10.0 * np.sin((hour - 14.0) / 24.0 * 2.0 * np.pi) - 4.0 * season + rng.normal(0.0, 1.0, total_samples)
    outside_light = np.clip(np.sin((hour - 6.0) / 12.0 * np.pi), 0.0, None) * (850.0 + 120.0 * season) + rng.normal(0.0, 20.0, total_samples)
    outside_light = np.clip(outside_light, 0.0, 1300.0)

    drip = (((hour >= 5.0) & (hour <= 6.5)) | ((hour >= 17.5) & (hour <= 18.5))).astype(float)
    drip = np.where((season > 0.4) & (hour >= 12.0) & (hour <= 13.0), 1.0, drip)
    drip = np.where(rng.random(total_samples) < 0.01, 1.0, drip)

    mist = (((hour >= 11.0) & (hour <= 14.0)) & (outside_temp > 31.0) & (outside_humidity < 68.0)).astype(float)
    fan = ((outside_temp > 32.0) | (outside_humidity > 82.0) | ((hour >= 12.0) & (hour <= 15.0))).astype(float)

    soil_moisture = np.zeros(total_samples, dtype=float)
    soil_moisture[:2] = 58.0

    for idx in range(2, total_samples):
        evap = (
            0.055 * (outside_temp[idx - 1] - 28.0)
            + 0.03 * (outside_light[idx - 1] / 1300.0)
            - 0.02 * (outside_humidity[idx - 1] - 70.0) / 10.0
            + 0.55 * fan[idx]
        )
        water_input = 1.8 * drip[idx - 1] + 1.05 * drip[idx - 2] + 0.35 * mist[idx]

        soil_moisture[idx] = (
            TRUE_PARAMS_REALISTIC["a1"] * soil_moisture[idx - 1]
            + TRUE_PARAMS_REALISTIC["a2"] * soil_moisture[idx - 2]
            + water_input
            - evap
            + rng.normal(0.0, TRUE_PARAMS_REALISTIC["noise_sigma"])
        )
        soil_moisture[idx] = np.clip(soil_moisture[idx], 10.0, 100.0)

    months = timestamps.month.to_numpy()
    seasons = np.where(
        np.isin(months, [3, 4, 5]),
        "spring",
        np.where(
            np.isin(months, [6, 7, 8]),
            "summer",
            np.where(np.isin(months, [9, 10, 11]), "autumn", "winter"),
        ),
    )

    df = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Month": months,
            "Season": seasons,
            "Soil_Moisture": soil_moisture,
            "Outside_Temp": outside_temp,
            "Outside_Humidity": outside_humidity,
            "Outside_Light": outside_light,
            "Drip": drip,
            "Mist": mist,
            "Fan": fan,
        }
    )

    return df, get_true_params()


if __name__ == "__main__":
    frame, params = generate_greenhouse_data_realistic()
    frame.to_csv("greenhouse_data_realistic.csv", index=False)
    print(f"Saved greenhouse_data_realistic.csv with {len(frame)} rows")
    print(params)
