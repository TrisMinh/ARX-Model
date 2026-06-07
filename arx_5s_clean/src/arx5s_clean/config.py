from __future__ import annotations

from dataclasses import dataclass


INSIDE_INPUT_COLS: tuple[str, ...] = (
    "Temperature_In",
    "Humidity_In",
    "Light_In",
    "Drip",
    "Mist",
    "Fan",
    "Light_log",
    "TempIn_x_HumiIn",
    "TempIn_x_Light",
    "HumiIn_x_Light",
    "Indoor_Dryness",
    "VPD_Proxy_In",
    "Hour_sin",
    "Hour_cos",
    "Day_sin",
    "Day_cos",
)


@dataclass(frozen=True)
class ExperimentConfig:
    days: int = 4
    sampling_seconds: int = 5
    seed: int = 505031
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    five_minute_seconds: int = 300
    control_horizon_seconds: int = 1200
    soil_low_sp: float = 55.0
    soil_high_sp: float = 65.0
    width_cm: float = 30.0
    length_cm: float = 50.0
    height_cm: float = 30.0
    front_temp_gain: float = 0.90
    front_temp_step: float = 0.30
    front_humi_drop: float = 3.20
    front_humi_step: float = 0.90

    @property
    def n_step_5min(self) -> int:
        return max(1, int(round(self.five_minute_seconds / self.sampling_seconds)))

    @property
    def n_step_control(self) -> int:
        return max(1, int(round(self.control_horizon_seconds / self.sampling_seconds)))

    @property
    def greenhouse_volume_m3(self) -> float:
        return self.width_cm * self.length_cm * self.height_cm / 1_000_000.0
