from __future__ import annotations

from dataclasses import dataclass


INPUT_COLS: tuple[str, ...] = (
    "Temperature",
    "Humidity",
    "Light",
    "Drip",
    "Mist",
    "Fan",
    "Light_log",
    "Temp_x_Humidity",
    "Temp_x_Light",
    "Humidity_x_Light",
    "Air_Dryness",
    "Dry_Air_Effect",
    "Hour_sin",
    "Hour_cos",
    "Day_sin",
    "Day_cos",
)


@dataclass(frozen=True)
class ExperimentConfig:
    days: int = 12
    sampling_seconds: int = 5
    seed: int = 505031
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    split_strategy: str = "same_clock_by_day"
    eval_window_start_hour: float = 6.0
    eval_window_end_hour: float = 15.0
    five_minute_seconds: int = 300
    control_horizon_seconds: int = 1200
    soil_low_sp: float = 55.0
    soil_high_sp: float = 65.0
    soil_clip_margin: float = 10.0

    @property
    def n_step_5min(self) -> int:
        return max(1, int(round(self.five_minute_seconds / self.sampling_seconds)))

    @property
    def n_step_control(self) -> int:
        return max(1, int(round(self.control_horizon_seconds / self.sampling_seconds)))
