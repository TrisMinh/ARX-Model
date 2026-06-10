from __future__ import annotations
from dataclasses import dataclass

#input collums
INPUT_COLS = (
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
    "Temp_x_Air_Dryness",
    "Hour_sin",
    "Hour_cos",
    "Day_sin",
    "Day_cos",
)


#cấu hình
@dataclass(frozen=True)
class ExperimentConfig:
    sampling_seconds: int = 5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    split_strategy: str = "day_ratio"
    eval_time_blocks: tuple[tuple[float, float], ...] = (
        (0.0, 24.0),
    )
    five_minute_seconds: int = 300
    control_horizon_seconds: int = 1200
    soil_low_sp: float = 55.0
    soil_high_sp: float = 65.0
    soil_clip_margin: float = 10.0
    selected_model_name: str = "ARX_na96_nb16_nk2_alpha10"
    report_model_names: tuple[str, ...] = (
        "ARX_na96_nb16_nk2_alpha10",
        "ARX_na24_nb4_nk2_alpha10",
        "ARX_na8_nb1_nk1_alpha1",
        "ARX_na8_nb2_nk1_alpha1",
        "ARX_na8_nb4_nk8_alpha100",
        "ARX_na8_nb4_nk12_alpha1",
    )

    #đổi từ thời gian sang số bước
    # Số bước tương ứng 5 phút.
    @property
    def n_step_5min(self) -> int:
        return max(1, int(round(self.five_minute_seconds / self.sampling_seconds)))

    # Số bước tương ứng horizon điều khiển.
    @property
    def n_step_control(self) -> int:
        return max(1, int(round(self.control_horizon_seconds / self.sampling_seconds)))
