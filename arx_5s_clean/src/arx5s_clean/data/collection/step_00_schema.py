from __future__ import annotations


MODEL_COLS: tuple[str, ...] = (
    "Timestamp",
    "Temperature",
    "Humidity",
    "Light",
    "Soil_Moisture",
    "Drip",
    "Mist",
    "Fan",
)

SAMPLE_SECONDS = 5

SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "Temperature": (15.0, 45.0),
    "Humidity": (30.0, 100.0),
    "Light": (0.0, 1200.0),
    "Soil_Moisture": (0.0, 100.0),
}

ACTUATOR_COLS: tuple[str, ...] = ("Drip", "Mist", "Fan")


def normalize_model_columns(df):
    out = df.copy()
    missing_cols = [col for col in MODEL_COLS if col not in out.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")
    return out.loc[:, MODEL_COLS].copy()
