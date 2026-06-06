from __future__ import annotations

import numpy as np
import pandas as pd


TRUE_PARAMS_POLY = {
    "a1": 0.94,
    "a2": 0.03,
    "b_temp_1": -0.010,
    "b_temp_2": -0.003,
    "b_humi_1": 0.003,
    "b_humi_2": 0.001,
    "b_light_1": -0.00015,
    "b_light_2": -0.00006,
    "b_drip_1": 0.90,
    "b_drip_2": 1.20,
    "b_mist_1": 0.05,
    "b_fan_1": -0.03,
    "c_temp2": 0.90,
    "c_humi2": -0.50,
    "c_temp_humi": 0.70,
    "noise_sigma": 0.30,
}


def _month_to_season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def get_true_params() -> dict[str, float]:
    return dict(TRUE_PARAMS_POLY)


def generate_greenhouse_data_poly(
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
    n_total = int(days * samples_per_day)

    t = np.arange(n_total)
    hour = (t % samples_per_day) / samples_per_day * 24.0
    timestamps = pd.date_range(start_date, periods=n_total, freq=f"{sampling_seconds}s")

    temperature = 26.0 + 5.0 * np.sin((hour - 7.0) / 24.0 * 2.0 * np.pi) + rng.normal(0.0, 0.7, n_total)
    humidity = 68.0 - 10.0 * np.sin((hour - 7.0) / 24.0 * 2.0 * np.pi) + rng.normal(0.0, 1.4, n_total)
    light = np.clip(np.sin((hour - 6.0) / 12.0 * np.pi), 0.0, None) * 900.0 + rng.normal(0.0, 25.0, n_total)

    drip = (rng.random(n_total) < 0.08).astype(float)
    mist = (rng.random(n_total) < 0.05).astype(float)
    fan = (rng.random(n_total) < 0.06).astype(float)

    y = np.zeros(n_total, dtype=float)
    y[:2] = 58.0

    p = TRUE_PARAMS_POLY
    for k in range(2, n_total):
        y_lin = (
            p["a1"] * y[k - 1]
            + p["a2"] * y[k - 2]
            + p["b_temp_1"] * temperature[k - 1]
            + p["b_temp_2"] * temperature[k - 2]
            + p["b_humi_1"] * humidity[k - 1]
            + p["b_humi_2"] * humidity[k - 2]
            + p["b_light_1"] * light[k - 1]
            + p["b_light_2"] * light[k - 2]
            + p["b_drip_1"] * drip[k - 1]
            + p["b_drip_2"] * drip[k - 2]
            + p["b_mist_1"] * mist[k - 1]
            + p["b_fan_1"] * fan[k - 1]
        )

        temp_n = (temperature[k - 1] - 25.0) / 10.0
        humi_n = (humidity[k - 1] - 65.0) / 20.0
        y_nl = p["c_temp2"] * (temp_n**2) + p["c_humi2"] * (humi_n**2) + p["c_temp_humi"] * (temp_n * humi_n)

        y[k] = np.clip(y_lin + y_nl + rng.normal(0.0, p["noise_sigma"]), 15.0, 95.0)

    months = timestamps.month.to_numpy()
    seasons = np.array([_month_to_season(int(m)) for m in months])

    df = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Month": months,
            "Season": seasons,
            "Soil_Moisture": y,
            "Temperature": temperature,
            "Humidity": humidity,
            "Light": light,
            "Drip": drip,
            "Mist": mist,
            "Fan": fan,
        }
    )
    return df, get_true_params()


if __name__ == "__main__":
    df_out, params = generate_greenhouse_data_poly()
    df_out.to_csv("greenhouse_data_poly.csv", index=False)
    print(f"Saved greenhouse_data_poly.csv with {len(df_out)} rows")
    print(params)
