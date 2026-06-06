from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class OutdoorConfig:
    days: int = 16
    sampling_seconds: int = 20
    seed: int = 305031
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    n_step_short: int = 12
    n_step_control: int = 60
    validation_blocks: int = 4
    validation_std_penalty: float = 0.5
    soil_low_sp: float = 55.0
    soil_high_sp: float = 65.0
    width_cm: float = 30.0
    length_cm: float = 50.0
    height_cm: float = 30.0
    front_temp_gain: float = 0.90
    front_temp_step: float = 0.30
    front_humi_drop: float = 3.20
    front_humi_step: float = 0.90


@dataclass(frozen=True)
class ArxSpec:
    na: int
    nb: int
    nk: int
    alpha: float

    @property
    def name(self) -> str:
        return f"ARX_na{self.na}_nb{self.nb}_nk{self.nk}_alpha{self.alpha:g}"


INSIDE_INPUT_COLS = (
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
    "SP_Center",
    "SP_Width",
    "Hour_sin",
    "Hour_cos",
    "Day_sin",
    "Day_cos",
    "Phase_identification",
)

OUTDOOR_EXTRA_COLS = (
    "Temperature_Out",
    "Humidity_Out",
    "Outdoor_Dryness",
    "VPD_Proxy_Out",
)

OUTDOOR_INPUT_COLS = INSIDE_INPUT_COLS + OUTDOOR_EXTRA_COLS


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def fit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    denom = np.linalg.norm(y_true - y_true.mean())
    fit = 100.0 * (1.0 - np.linalg.norm(residual) / denom) if denom > 1e-12 else float("nan")
    return {
        "FIT": float(fit),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "Bias": float(np.mean(residual)),
    }


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return None
    left = left[mask]
    right = right[mask]
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def p_on(mask: np.ndarray, values: np.ndarray) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return None
    return float(100.0 * np.mean(np.asarray(values, dtype=float)[mask] > 0.5))


def phase_for_day(day: int) -> str:
    if day < 2:
        return "commissioning_rule_based"
    if day < 8:
        return "identification_safe_excitation"
    return "deployment_validation"


def make_planned_excitation(cfg: OutdoorConfig, rng: np.random.Generator, n_rows: int) -> dict[str, np.ndarray]:
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


def generate_outdoor_data(cfg: OutdoorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    samples_per_day = int(round(24 * 3600 / cfg.sampling_seconds))
    n_rows = cfg.days * samples_per_day
    t = np.arange(n_rows)
    timestamp = pd.date_range("2026-01-01", periods=n_rows, freq=f"{cfg.sampling_seconds}s")
    hour = (t % samples_per_day) / samples_per_day * 24.0
    day_index = t // samples_per_day
    phase = np.asarray([phase_for_day(int(day)) for day in day_index])
    volume_m3 = cfg.width_cm * cfg.length_cm * cfg.height_cm / 1_000_000.0

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

    planned = make_planned_excitation(cfg, rng, n_rows)
    drip = np.zeros(n_rows, dtype=float)
    mist = np.zeros(n_rows, dtype=float)
    fan = np.zeros(n_rows, dtype=float)
    source = np.full(n_rows, "none", dtype=object)
    safety_override = np.zeros(n_rows, dtype=int)
    block_wet = np.zeros(n_rows, dtype=int)
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
            block_wet[idx] = 1
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

    return pd.DataFrame(
        {
            "Timestamp": timestamp,
            "Day_Index": day_index,
            "Hour": hour,
            "Protocol_Phase": phase,
            "Greenhouse_Volume_m3": np.full(n_rows, volume_m3),
            "Temperature_Out": outdoor_temp,
            "Humidity_Out": outdoor_humi,
            "Light_Out": light_out,
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
            "Wet_Block": block_wet,
        }
    )


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    timestamp = pd.to_datetime(df["Timestamp"])
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0 + timestamp.dt.second / 3600.0
    day_num = df["Day_Index"].astype(float)
    df["Light_log"] = np.log1p(df["Light_In"].clip(lower=0))
    df["TempIn_x_HumiIn"] = df["Temperature_In"] * df["Humidity_In"]
    df["TempIn_x_Light"] = df["Temperature_In"] * df["Light_log"]
    df["HumiIn_x_Light"] = df["Humidity_In"] * df["Light_log"]
    df["Indoor_Dryness"] = 100.0 - df["Humidity_In"]
    df["Outdoor_Dryness"] = 100.0 - df["Humidity_Out"]
    df["VPD_Proxy_In"] = (df["Temperature_In"] - 20.0).clip(lower=0.0) * df["Indoor_Dryness"].clip(lower=0.0) / 100.0
    df["VPD_Proxy_Out"] = (df["Temperature_Out"] - 20.0).clip(lower=0.0) * df["Outdoor_Dryness"].clip(lower=0.0) / 100.0
    df["Temp_Delta_Out_In"] = df["Temperature_Out"] - df["Temperature_In"]
    df["Humi_Delta_Out_In"] = df["Humidity_Out"] - df["Humidity_In"]
    df["Light_Delta_Out_In"] = df["Light_Out"] - df["Light_In"]
    df["Fan_x_Temp_Delta"] = df["Fan"] * df["Temp_Delta_Out_In"]
    df["Fan_x_Humi_Delta"] = df["Fan"] * df["Humi_Delta_Out_In"]
    df["Fan_x_Outdoor_Dryness"] = df["Fan"] * df["Outdoor_Dryness"]
    df["Mist_x_Humi_In"] = df["Mist"] * df["Humidity_In"]
    df["SP_Center"] = 0.5 * (df["Soil_Low_SP"] + df["Soil_High_SP"])
    df["SP_Width"] = df["Soil_High_SP"] - df["Soil_Low_SP"]
    df["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["Day_sin"] = np.sin(2.0 * np.pi * day_num / 7.0)
    df["Day_cos"] = np.cos(2.0 * np.pi * day_num / 7.0)
    df["Phase_identification"] = (df["Protocol_Phase"] == "identification_safe_excitation").astype(float)
    return df


def split_time(df: pd.DataFrame, cfg: OutdoorConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    n_train = int(n_rows * cfg.train_ratio)
    n_val = int(n_rows * cfg.val_ratio)
    return (
        df.iloc[:n_train].reset_index(drop=True),
        df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        df.iloc[n_train + n_val :].reset_index(drop=True),
    )


def fit_scale_stats(df_train: pd.DataFrame, input_cols: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in ("Soil_Moisture", *input_cols):
        mean = float(df_train[col].astype(float).mean())
        std = float(df_train[col].astype(float).std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        stats[col] = (mean, std)
    return stats


def apply_scale(df_in: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    df = df_in.copy()
    for col, (mean, std) in stats.items():
        df[col] = (df[col].astype(float) - mean) / std
    return df


def inverse_y(y_z: np.ndarray, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    mean, std = stats["Soil_Moisture"]
    return np.asarray(y_z, dtype=float) * std + mean


def arx_max_lag(spec: ArxSpec) -> int:
    return max(spec.na, spec.nb + spec.nk - 1)


def build_arx_matrix(df_z: pd.DataFrame, spec: ArxSpec, input_cols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    lag = arx_max_lag(spec)
    cols: list[np.ndarray] = []
    for y_lag in range(1, spec.na + 1):
        cols.append(y[lag - y_lag : len(y) - y_lag])
    for col in input_cols:
        values = df_z[col].to_numpy(dtype=float)
        for u_lag in range(spec.nk, spec.nk + spec.nb):
            cols.append(values[lag - u_lag : len(values) - u_lag])
    cols.append(np.ones(len(y) - lag))
    return np.vstack(cols).T, y[lag:]


def fit_arx(df_train_z: pd.DataFrame, spec: ArxSpec, input_cols: tuple[str, ...]) -> np.ndarray:
    x_train, y_train = build_arx_matrix(df_train_z, spec, input_cols)
    if spec.alpha <= 0.0:
        theta, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
        return theta
    penalty = np.eye(x_train.shape[1], dtype=float)
    penalty[-1, -1] = 0.0
    lhs = x_train.T @ x_train + spec.alpha * penalty
    rhs = x_train.T @ y_train
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        theta, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
        return theta


def arx_predict_at(
    y_source: np.ndarray,
    inputs: list[np.ndarray],
    t: int,
    theta: np.ndarray,
    spec: ArxSpec,
) -> float:
    idx = 0
    y_next = 0.0
    for y_lag in range(1, spec.na + 1):
        y_next += theta[idx] * y_source[t - y_lag]
        idx += 1
    for values in inputs:
        for u_lag in range(spec.nk, spec.nk + spec.nb):
            y_next += theta[idx] * values[t - u_lag]
            idx += 1
    y_next += theta[idx]
    return float(y_next)


def predict_arx_one_step(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x, y_true = build_arx_matrix(df_z, spec, input_cols)
    return np.clip(x @ theta, clip[0], clip[1]), y_true


def simulate_arx(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_sim = y.copy()
    inputs = [df_z[col].to_numpy(dtype=float) for col in input_cols]
    lag = arx_max_lag(spec)
    for t in range(lag, len(y)):
        y_sim[t] = float(np.clip(arx_predict_at(y_sim, inputs, t, theta, spec), clip[0], clip[1]))
    return y_sim[lag:], y[lag:]


def simulate_arx_n_step(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    clip: tuple[float, float],
    n_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_work = y.copy()
    y_pred = y.copy()
    inputs = [df_z[col].to_numpy(dtype=float) for col in input_cols]
    lag = arx_max_lag(spec)
    for t in range(lag, len(y)):
        origin = max(lag - 1, t - n_step)
        reset_start = max(0, origin - spec.na)
        y_work[reset_start : t + 1] = y[reset_start : t + 1]
        for step_t in range(origin + 1, t + 1):
            y_work[step_t] = float(np.clip(arx_predict_at(y_work, inputs, step_t, theta, spec), clip[0], clip[1]))
        y_pred[t] = y_work[t]
    return y_pred[lag:], y[lag:]


def evaluate_arx(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    cfg: OutdoorConfig,
    include_control_horizon: bool,
) -> dict[str, dict[str, float]]:
    y_1, yt_1 = predict_arx_one_step(df_z, theta, spec, input_cols, clip)
    y_12, yt_12 = simulate_arx_n_step(df_z, theta, spec, input_cols, clip, cfg.n_step_short)
    y_sim, yt_sim = simulate_arx(df_z, theta, spec, input_cols, clip)
    out = {
        "metrics_1step": fit_metrics(inverse_y(yt_1, stats), inverse_y(y_1, stats)),
        "metrics_12": fit_metrics(inverse_y(yt_12, stats), inverse_y(y_12, stats)),
        "metrics_sim": fit_metrics(inverse_y(yt_sim, stats), inverse_y(y_sim, stats)),
    }
    if include_control_horizon:
        y_60, yt_60 = simulate_arx_n_step(df_z, theta, spec, input_cols, clip, cfg.n_step_control)
        out["metrics_60"] = fit_metrics(inverse_y(yt_60, stats), inverse_y(y_60, stats))
    return out


def validation_blocks_arx(
    df_val_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    cfg: OutdoorConfig,
) -> dict[str, Any]:
    scores: list[float] = []
    for block in np.array_split(np.arange(len(df_val_z)), cfg.validation_blocks):
        df_block = df_val_z.iloc[int(block[0]) : int(block[-1]) + 1].reset_index(drop=True)
        y_sim, yt = simulate_arx(df_block, theta, spec, input_cols, clip)
        scores.append(fit_metrics(inverse_y(yt, stats), inverse_y(y_sim, stats))["FIT"])
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=0))
    return {
        "val_block_FIT_sim": scores,
        "val_block_mean_FIT_sim": mean_score,
        "val_block_std_FIT_sim": std_score,
        "val_robust_score": mean_score - cfg.validation_std_penalty * std_score,
    }


def arx_specs() -> list[ArxSpec]:
    specs: list[ArxSpec] = []
    for na in (12, 18, 24):
        for nb in (3, 6):
            for nk in (1, 2, 3):
                for alpha in (0.1, 1.0, 10.0):
                    specs.append(ArxSpec(na, nb, nk, alpha))
    return specs


def run_arx_search(
    label: str,
    input_cols: tuple[str, ...],
    df_train_z: pd.DataFrame,
    df_val_z: pd.DataFrame,
    df_test_z: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    cfg: OutdoorConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fitted: dict[str, tuple[ArxSpec, np.ndarray]] = {}
    for spec in arx_specs():
        theta = fit_arx(df_train_z, spec, input_cols)
        val_1, val_true_1 = predict_arx_one_step(df_val_z, theta, spec, input_cols, clip)
        val_sim, val_true_sim = simulate_arx(df_val_z, theta, spec, input_cols, clip)
        robust = validation_blocks_arx(df_val_z, theta, spec, input_cols, stats, clip, cfg)
        fitted[spec.name] = (spec, theta)
        rows.append(
            {
                "input_set": label,
                "model": spec.name,
                "na": spec.na,
                "nb": spec.nb,
                "nk": spec.nk,
                "alpha": spec.alpha,
                "physical_memory_seconds": spec.na * cfg.sampling_seconds,
                "input_delay_seconds": spec.nk * cfg.sampling_seconds,
                "input_memory_seconds": spec.nb * cfg.sampling_seconds,
                "n_input_cols": len(input_cols),
                "n_params": int(len(theta)),
                "val_FIT_1step": fit_metrics(inverse_y(val_true_1, stats), inverse_y(val_1, stats))["FIT"],
                "val_FIT_sim": fit_metrics(inverse_y(val_true_sim, stats), inverse_y(val_sim, stats))["FIT"],
                **robust,
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values("val_robust_score", ascending=False).reset_index(drop=True)
    selected = leaderboard.iloc[0].to_dict()
    spec, theta = fitted[str(selected["model"])]
    return {
        "input_set": label,
        "input_cols": list(input_cols),
        "leaderboard": leaderboard,
        "selected_by_validation": selected,
        "spec": spec,
        "theta": theta,
        "validation": evaluate_arx(df_val_z, theta, spec, input_cols, stats, clip, cfg, include_control_horizon=True),
        "test": evaluate_arx(df_test_z, theta, spec, input_cols, stats, clip, cfg, include_control_horizon=True),
    }


def build_fixed_arx(
    label: str,
    input_cols: tuple[str, ...],
    df_train_z: pd.DataFrame,
    df_val_z: pd.DataFrame,
    df_test_z: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    cfg: OutdoorConfig,
) -> dict[str, Any]:
    spec = ArxSpec(5, 1, 2, 0.0)
    theta = fit_arx(df_train_z, spec, input_cols)
    return {
        "input_set": label,
        "input_cols": list(input_cols),
        "selected_by_validation": {
            "input_set": label,
            "model": spec.name,
            "na": spec.na,
            "nb": spec.nb,
            "nk": spec.nk,
            "alpha": spec.alpha,
            "physical_memory_seconds": spec.na * cfg.sampling_seconds,
            "input_delay_seconds": spec.nk * cfg.sampling_seconds,
            "input_memory_seconds": spec.nb * cfg.sampling_seconds,
            "n_input_cols": len(input_cols),
            "n_params": int(len(theta)),
        },
        "validation": evaluate_arx(df_val_z, theta, spec, input_cols, stats, clip, cfg, include_control_horizon=True),
        "test": evaluate_arx(df_test_z, theta, spec, input_cols, stats, clip, cfg, include_control_horizon=True),
        "spec": spec,
        "theta": theta,
    }


def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, max_lag: int = 60) -> dict[str, Any]:
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    acf: list[float] = []
    for lag in range(1, max_lag + 1):
        acf.append(safe_corr(residual[lag:], residual[:-lag]) or 0.0)
    return {
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "max_abs_acf_lag_1_to_60": float(np.max(np.abs(acf))),
        "acf_lag_1_to_60": acf,
    }


def audit_data(df: pd.DataFrame, cfg: OutdoorConfig) -> dict[str, Any]:
    df = df.sort_values("Timestamp").reset_index(drop=True)
    diffs = pd.to_datetime(df["Timestamp"]).diff().dt.total_seconds().dropna()
    y_prev = df["Soil_Moisture"].to_numpy(dtype=float)[:-1]
    low_prev = df["Soil_Low_SP"].to_numpy(dtype=float)[:-1]
    high_prev = df["Soil_High_SP"].to_numpy(dtype=float)[:-1]
    center_prev = 0.5 * (low_prev + high_prev)
    drip_now = df["Drip"].to_numpy(dtype=float)[1:]
    planned_now = df["Planned_Drip"].to_numpy(dtype=float)[1:]
    planned_fan_now = df["Planned_Fan"].to_numpy(dtype=float)[1:]
    below_low = y_prev < low_prev
    safe_mid = (y_prev >= low_prev + 1.0) & (y_prev <= high_prev - 1.5)
    above_high = y_prev > high_prev
    train, val, test = split_time(df, cfg)

    def split_summary(name: str, part: pd.DataFrame) -> dict[str, Any]:
        return {
            "name": name,
            "rows": int(len(part)),
            "start": str(part["Timestamp"].iloc[0]),
            "end": str(part["Timestamp"].iloc[-1]),
            "soil_mean": float(part["Soil_Moisture"].mean()),
            "soil_std": float(part["Soil_Moisture"].std(ddof=0)),
            "soil_min": float(part["Soil_Moisture"].min()),
            "soil_max": float(part["Soil_Moisture"].max()),
            "drip_on_pct": float(100.0 * part["Drip"].mean()),
            "fan_on_pct": float(100.0 * part["Fan"].mean()),
            "planned_drip_pct": float(100.0 * part["Planned_Drip"].mean()),
            "planned_fan_pct": float(100.0 * part["Planned_Fan"].mean()),
        }

    return {
        "rows": int(len(df)),
        "start": str(df["Timestamp"].iloc[0]),
        "end": str(df["Timestamp"].iloc[-1]),
        "sampling_seconds": cfg.sampling_seconds,
        "greenhouse_volume_m3": float(df["Greenhouse_Volume_m3"].iloc[0]),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_timestamps": int(df["Timestamp"].duplicated().sum()),
        "median_sampling_seconds": float(diffs.median()),
        "irregular_sampling_count": int((diffs != diffs.median()).sum()),
        "soil_mean": float(df["Soil_Moisture"].mean()),
        "soil_std": float(df["Soil_Moisture"].std(ddof=0)),
        "soil_min": float(df["Soil_Moisture"].min()),
        "soil_max": float(df["Soil_Moisture"].max()),
        "temperature_in_range": [float(df["Temperature_In"].min()), float(df["Temperature_In"].max())],
        "humidity_in_range": [float(df["Humidity_In"].min()), float(df["Humidity_In"].max())],
        "temperature_out_range": [float(df["Temperature_Out"].min()), float(df["Temperature_Out"].max())],
        "humidity_out_range": [float(df["Humidity_Out"].min()), float(df["Humidity_Out"].max())],
        "mean_abs_temp_delta_out_in": float(np.mean(np.abs(df["Temperature_Out"] - df["Temperature_In"]))),
        "mean_abs_humi_delta_out_in": float(np.mean(np.abs(df["Humidity_Out"] - df["Humidity_In"]))),
        "air_exchange_fan_off_mean": float(df.loc[df["Fan"] < 0.5, "Air_Exchange_Rate"].mean()),
        "air_exchange_fan_on_mean": float(df.loc[df["Fan"] > 0.5, "Air_Exchange_Rate"].mean()),
        "phase_counts": {str(k): int(v) for k, v in df["Protocol_Phase"].value_counts().to_dict().items()},
        "command_source_counts": {str(k): int(v) for k, v in df["Command_Source"].value_counts().to_dict().items()},
        "safety_override_pct": float(100.0 * df["Safety_Override"].mean()),
        "wet_block_pct": float(100.0 * df["Wet_Block"].mean()),
        "drip_on_pct": float(100.0 * df["Drip"].mean()),
        "fan_on_pct": float(100.0 * df["Fan"].mean()),
        "mist_on_pct": float(100.0 * df["Mist"].mean()),
        "planned_drip_pct": float(100.0 * df["Planned_Drip"].mean()),
        "planned_fan_pct": float(100.0 * df["Planned_Fan"].mean()),
        "p_drip_on_when_prev_soil_below_low_pct": p_on(below_low, drip_now),
        "p_drip_on_when_prev_soil_safe_mid_pct": p_on(safe_mid, drip_now),
        "p_drip_on_when_prev_soil_above_high_pct": p_on(above_high, drip_now),
        "corr_drip_t_with_prev_soil_minus_center": safe_corr(drip_now, y_prev - center_prev),
        "corr_planned_drip_t_with_prev_soil_minus_center": safe_corr(planned_now, y_prev - center_prev),
        "corr_planned_fan_t_with_prev_soil_minus_center": safe_corr(planned_fan_now, y_prev - center_prev),
        "corr_fan_with_humi_delta_out_in": safe_corr(df["Fan"].to_numpy(dtype=float), df["Humidity_Out"].to_numpy(dtype=float) - df["Humidity_In"].to_numpy(dtype=float)),
        "split": {
            "train": split_summary("train", train),
            "validation": split_summary("validation", val),
            "test": split_summary("test", test),
        },
    }


def comparison_row(case: str, family: str, result: dict[str, Any], note: str) -> dict[str, Any]:
    test = result["test"]
    selected = result["selected_by_validation"]
    return {
        "case": case,
        "family": family,
        "input_set": result["input_set"],
        "n_input_cols": selected["n_input_cols"],
        "selected_model": selected["model"],
        "selection_policy": "validation robust score",
        "test_FIT_1step_20s": test["metrics_1step"]["FIT"],
        "test_FIT_12_4min": test["metrics_12"]["FIT"],
        "test_FIT_60_20min": test["metrics_60"]["FIT"],
        "test_FIT_sim": test["metrics_sim"]["FIT"],
        "test_RMSE_sim": test["metrics_sim"]["RMSE"],
        "note": note,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def write_report(payload: dict[str, Any]) -> None:
    data = payload["data_audit"]
    rows = payload["comparison"]
    inside = next(row for row in rows if row["case"] == "ARX robust chỉ cảm biến trong")
    outdoor = next(row for row in rows if row["case"] == "ARX robust có cảm biến ngoài")
    improvement = outdoor["test_FIT_sim"] - inside["test_FIT_sim"]
    fit60_improvement = outdoor["test_FIT_60_20min"] - inside["test_FIT_60_20min"]
    if improvement > 1.0:
        outdoor_conclusion = "cải thiện rõ free-run so với bản chỉ dùng cảm biến trong"
    elif improvement < -1.0:
        outdoor_conclusion = "chưa cải thiện free-run so với bản chỉ dùng cảm biến trong"
    else:
        outdoor_conclusion = "gần như ngang free-run với bản chỉ dùng cảm biến trong"
    arx_outdoor_selected = payload["arx_outdoor"]["selected_by_validation"]
    arx_inside_selected = payload["arx_inside"]["selected_by_validation"]

    lines = [
        "# Báo Cáo Cuối: ARX Có Cảm Biến Ngoài",
        "",
        "## Kết luận ngắn",
        "",
        "- Với mô hình nhỏ `30x50x30 cm`, quạt làm không khí trong hộp trao đổi với môi trường ngoài rất nhanh.",
        "- Vì vậy `Temperature_Out` và `Humidity_Out` là biến nhiễu đo được, không nên bỏ qua nếu có thể lắp cảm biến.",
        "- Kết quả dưới đây là kết quả chạy thật của pipeline trên dữ liệu mô phỏng vật lý có log rõ, không phải dữ liệu phần cứng.",
        f"- ARX chỉ dùng cảm biến trong đạt `FIT_sim = {inside['test_FIT_sim']:.3f}`.",
        f"- ARX có thêm cảm biến ngoài đạt `FIT_sim = {outdoor['test_FIT_sim']:.3f}`.",
        f"- Mức chênh free-run khi thêm cảm biến ngoài: `{improvement:.3f}` điểm FIT, tức là {outdoor_conclusion}.",
        f"- Mức chênh horizon 20 phút khi thêm cảm biến ngoài: `{fit60_improvement:.3f}` điểm FIT.",
        "- Nếu thu dữ liệu thật, vẫn nên log cả biến trong và biến ngoài. Sau khi có dữ liệu thật, chọn dùng hay bỏ sensor ngoài phải dựa trên validation/test, không dựa trên cảm tính.",
        "",
        "## Kiểm tra dữ liệu",
        "",
        "| Hạng mục | Giá trị |",
        "| --- | ---: |",
        f"| Thể tích nhà kính | {fmt(data['greenhouse_volume_m3'], 4)} m3 |",
        f"| Thời gian lấy mẫu | {fmt(data['sampling_seconds'], 0)} giây |",
        f"| Số dòng dữ liệu | {data['rows']} |",
        f"| Missing | {data['missing_total']} |",
        f"| Timestamp trùng | {data['duplicate_timestamps']} |",
        f"| Sai chu kỳ lấy mẫu | {data['irregular_sampling_count']} |",
        f"| Soil min-max | {fmt(data['soil_min'])} - {fmt(data['soil_max'])} |",
        f"| Soil std | {fmt(data['soil_std'])} |",
        f"| Temperature_In range | {fmt(data['temperature_in_range'][0])} - {fmt(data['temperature_in_range'][1])} |",
        f"| Temperature_Out range | {fmt(data['temperature_out_range'][0])} - {fmt(data['temperature_out_range'][1])} |",
        f"| Humidity_In range | {fmt(data['humidity_in_range'][0])} - {fmt(data['humidity_in_range'][1])} |",
        f"| Humidity_Out range | {fmt(data['humidity_out_range'][0])} - {fmt(data['humidity_out_range'][1])} |",
        f"| Mean abs temp delta out-in | {fmt(data['mean_abs_temp_delta_out_in'])} |",
        f"| Mean abs humi delta out-in | {fmt(data['mean_abs_humi_delta_out_in'])} |",
        f"| Air exchange fan OFF mean | {fmt(data['air_exchange_fan_off_mean'])} |",
        f"| Air exchange fan ON mean | {fmt(data['air_exchange_fan_on_mean'])} |",
        f"| Drip ON % | {fmt(data['drip_on_pct'])} |",
        f"| Fan ON % | {fmt(data['fan_on_pct'])} |",
        f"| Planned Drip % | {fmt(data['planned_drip_pct'])} |",
        f"| Planned Fan % | {fmt(data['planned_fan_pct'])} |",
        f"| Safety override % | {fmt(data['safety_override_pct'])} |",
        f"| corr(Planned Drip, prev soil-center) | {fmt(data['corr_planned_drip_t_with_prev_soil_minus_center'])} |",
        f"| corr(Planned Fan, prev soil-center) | {fmt(data['corr_planned_fan_t_with_prev_soil_minus_center'])} |",
        "",
        "Các planned pulse được tạo theo lịch/seed và vẫn bị safety supervisor chặn nếu nguy hiểm. Vì vậy đây là nhận dạng hệ thống có kiểm soát, không phải dùng tương lai để làm đẹp kết quả.",
        "",
        "## So sánh mô hình",
        "",
        "| Trường hợp | Họ mô hình | Input | Số cột input | Model chọn | FIT_1step 20s | FIT_12 4 phút | FIT_60 20 phút | FIT_sim | RMSE_sim | Ghi chú |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['family']} | {row['input_set']} | {row['n_input_cols']} | `{row['selected_model']}` | "
            f"{fmt(row['test_FIT_1step_20s'])} | {fmt(row['test_FIT_12_4min'])} | {fmt(row['test_FIT_60_20min'])} | "
            f"{fmt(row['test_FIT_sim'])} | {fmt(row['test_RMSE_sim'], 4)} | {row['note']} |"
        )

    lines.extend(
        [
            "",
            "## Chi tiết ARX",
            "",
            f"- ARX chỉ cảm biến trong chọn `{arx_inside_selected['model']}`, robust validation score `{arx_inside_selected['val_robust_score']:.3f}`.",
            f"- ARX có cảm biến ngoài chọn `{arx_outdoor_selected['model']}`, robust validation score `{arx_outdoor_selected['val_robust_score']:.3f}`.",
            f"- ARX ngoài có input delay `{arx_outdoor_selected['input_delay_seconds']:.0f}` giây, output memory `{arx_outdoor_selected['physical_memory_seconds']:.0f}` giây, input memory `{arx_outdoor_selected['input_memory_seconds']:.0f}` giây.",
            f"- Residual free-run ARX ngoài max abs ACF lag 1..60: `{payload['arx_outdoor_sim_residual_diagnostics']['max_abs_acf_lag_1_to_60']:.3f}`.",
            "",
            "## Kết luận cho đồ án",
            "",
            "Nếu có điều kiện phần cứng, nên thêm một cảm biến nhiệt độ/độ ẩm ngoài hộp. Khi quạt bật, biến ngoài quyết định chiều và tốc độ trao đổi không khí, nên nó là nhiễu đo được của plant. Tuy nhiên, kết quả cuối phải được chọn theo validation/test: nếu sensor ngoài không cải thiện rõ thì vẫn có thể dùng ARX chỉ cảm biến trong để giữ mô hình gọn hơn.",
            "",
            "## Giới hạn",
            "",
            "1. Dữ liệu này vẫn là mô phỏng vật lý để kiểm thử quy trình, không được gọi là dữ liệu phần cứng.",
            "2. Khi có dữ liệu thật, phải retrain và báo cáo lại trên test thật; không được lấy chỉ số mô phỏng làm kết quả thực nghiệm cuối.",
            "3. Các biến `Temperature_Air_True`, `Humidity_Air_True`, `Air_Exchange_Rate` chỉ là trạng thái ẩn trong mô phỏng để audit; khi train model chỉ dùng các cột input đã khai báo.",
            "4. Nếu thêm cảm biến ngoài mà dữ liệu thật cho thấy không cải thiện validation/test, vẫn phải báo cáo trung thực và giữ mô hình đơn giản hơn.",
            "",
            "## Chạy lại",
            "",
            "```powershell",
            "python -B .\\ARX_MiniGreenhouse_20s_Outdoor_PBL5\\src\\outdoor_pipeline.py",
            "```",
            "",
        ]
    )
    (RESULTS_DIR / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_self_critique(payload: dict[str, Any]) -> None:
    rows = payload["comparison"]
    inside = next(row for row in rows if row["case"] == "ARX robust chỉ cảm biến trong")
    outdoor = next(row for row in rows if row["case"] == "ARX robust có cảm biến ngoài")
    improvement = outdoor["test_FIT_sim"] - inside["test_FIT_sim"]
    fit60_improvement = outdoor["test_FIT_60_20min"] - inside["test_FIT_60_20min"]
    lines = [
        "# Tự Phản Biện Bản Có Cảm Biến Ngoài",
        "",
        "## Những điểm thầy có thể bắt lỗi",
        "",
        "- Nếu nói đây là dữ liệu thật thì sai. Đây là dữ liệu mô phỏng vật lý để kiểm thử pipeline và thiết kế thí nghiệm.",
        "- Nếu dùng `Temperature_Air_True` hoặc `Humidity_Air_True` để train thì leakage vì đó là trạng thái ẩn của mô phỏng. Pipeline không đưa các cột này vào input.",
        "- Nếu chọn model theo test thì leakage. Pipeline chọn model bằng validation robust score, sau đó mới báo cáo test.",
        "- Nếu chỉ nhìn `FIT_1step` thì dễ ảo vì sampling 20 giây. Báo cáo phải có `FIT_12`, `FIT_60` và `FIT_sim`.",
        "- Nếu thêm quá nhiều feature ngoài nhưng không có cải thiện test thì nên bỏ bớt. Bản này giữ cả mô hình chỉ-cảm-biến-trong để đối chứng.",
        "",
        "## Vì sao thêm cảm biến ngoài là hợp lý",
        "",
        "- Quạt không chỉ là actuator làm mát; nó tạo trao đổi khối khí giữa trong và ngoài.",
        "- Cùng một lệnh quạt nhưng nếu bên ngoài khô hơn thì đất khô nhanh hơn; nếu bên ngoài ẩm hơn thì tác động khác.",
        "- Cảm biến trong có độ trễ và chỉ đo tại một vị trí, nên biến ngoài giúp giải thích phần nhiễu môi trường trước khi nó phản ánh đầy đủ vào cảm biến trong.",
        f"- Trong lần chạy này, thêm biến ngoài chênh `FIT_sim` khoảng `{improvement:.3f}` điểm và chênh `FIT_60` khoảng `{fit60_improvement:.3f}` điểm so với ARX chỉ dùng cảm biến trong.",
        "",
        "## Cách nói khi bảo vệ",
        "",
        "Nhóm không giả định nhà kính là hệ kín. Với mô hình nhỏ, khi quạt bật thì không khí ngoài đi vào và không khí trong đi ra, làm thay đổi tốc độ bay hơi và độ ẩm đất. Vì vậy nhóm đề xuất log thêm `Temperature_Out` và `Humidity_Out` như nhiễu đo được. Để kiểm chứng, nhóm train hai ARX trên cùng dữ liệu và cùng split: một bản chỉ dùng cảm biến trong, một bản có thêm cảm biến ngoài. Model được chọn bằng validation robust score và đánh giá cuối trên test theo thứ tự thời gian.",
        "",
    ]
    (RESULTS_DIR / "SELF_CRITIQUE.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(cfg: OutdoorConfig) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = generate_outdoor_data(cfg)
    df = add_features(raw_df)
    train, val, test = split_time(df, cfg)

    inside_stats = fit_scale_stats(train, INSIDE_INPUT_COLS)
    train_inside_z = apply_scale(train, inside_stats)
    val_inside_z = apply_scale(val, inside_stats)
    test_inside_z = apply_scale(test, inside_stats)
    inside_clip = tuple(float(v) for v in np.quantile(train_inside_z["Soil_Moisture"], (0.005, 0.995)))

    outdoor_stats = fit_scale_stats(train, OUTDOOR_INPUT_COLS)
    train_outdoor_z = apply_scale(train, outdoor_stats)
    val_outdoor_z = apply_scale(val, outdoor_stats)
    test_outdoor_z = apply_scale(test, outdoor_stats)
    outdoor_clip = tuple(float(v) for v in np.quantile(train_outdoor_z["Soil_Moisture"], (0.005, 0.995)))

    arx_inside_fixed = build_fixed_arx("inside_only", INSIDE_INPUT_COLS, train_inside_z, val_inside_z, test_inside_z, inside_stats, inside_clip, cfg)
    arx_inside = run_arx_search("inside_only", INSIDE_INPUT_COLS, train_inside_z, val_inside_z, test_inside_z, inside_stats, inside_clip, cfg)
    arx_outdoor = run_arx_search("outdoor", OUTDOOR_INPUT_COLS, train_outdoor_z, val_outdoor_z, test_outdoor_z, outdoor_stats, outdoor_clip, cfg)

    arx_outdoor_sim_z, arx_outdoor_true_sim_z = simulate_arx(
        test_outdoor_z,
        arx_outdoor["theta"],
        arx_outdoor["spec"],
        OUTDOOR_INPUT_COLS,
        outdoor_clip,
    )
    arx_outdoor_one_z, arx_outdoor_true_one_z = predict_arx_one_step(
        test_outdoor_z,
        arx_outdoor["theta"],
        arx_outdoor["spec"],
        OUTDOOR_INPUT_COLS,
        outdoor_clip,
    )
    arx_outdoor_sim_diag = residual_diagnostics(inverse_y(arx_outdoor_true_sim_z, outdoor_stats), inverse_y(arx_outdoor_sim_z, outdoor_stats))
    arx_outdoor_one_diag = residual_diagnostics(inverse_y(arx_outdoor_true_one_z, outdoor_stats), inverse_y(arx_outdoor_one_z, outdoor_stats))

    arx_inside_sim_z, arx_inside_true_sim_z = simulate_arx(
        test_inside_z,
        arx_inside["theta"],
        arx_inside["spec"],
        INSIDE_INPUT_COLS,
        inside_clip,
    )
    arx_inside_lag = arx_max_lag(arx_inside["spec"])
    arx_outdoor_lag = arx_max_lag(arx_outdoor["spec"])
    test_predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "Timestamp_ARX_Inside": test["Timestamp"].iloc[arx_inside_lag:].to_numpy(),
                    "y_true_inside_window": inverse_y(arx_inside_true_sim_z, inside_stats),
                    "y_arx_inside_sim": inverse_y(arx_inside_sim_z, inside_stats),
                }
            ),
            pd.DataFrame(
                {
                    "Timestamp_ARX_Outdoor": test["Timestamp"].iloc[arx_outdoor_lag:].to_numpy(),
                    "y_true_outdoor_window": inverse_y(arx_outdoor_true_sim_z, outdoor_stats),
                    "y_arx_outdoor_sim": inverse_y(arx_outdoor_sim_z, outdoor_stats),
                }
            ),
        ],
        axis=1,
    )

    comparison = [
        comparison_row("ARX baseline cũ chỉ cảm biến trong", "ARX tuyến tính", arx_inside_fixed, "Order ARX(5,1,2) để so với cách cũ"),
        comparison_row("ARX robust chỉ cảm biến trong", "ARX tuyến tính", arx_inside, "Không dùng Temperature_Out/Humidity_Out"),
        comparison_row("ARX robust có cảm biến ngoài", "ARX tuyến tính", arx_outdoor, "Có thêm Temperature_Out, Humidity_Out, Outdoor_Dryness và VPD_Proxy_Out"),
    ]

    payload = {
        "config": asdict(cfg),
        "data_policy": {
            "model_scale": "mô hình nhà kính nhỏ 30x50x30 cm",
            "sampling": "20 giây/mẫu",
            "initial_control": "rule-based safety supervisor",
            "excitation": "planned drip/fan/mist theo lịch, có safety override",
            "target_leakage": False,
            "hidden_state_not_used_for_training": ["Temperature_Air_True", "Humidity_Air_True", "Air_Exchange_Rate"],
            "split": "chia theo thời gian 70/15/15",
            "selection_metric": "validation robust score = mean(block FIT_sim) - 0.5*std(block FIT_sim)",
        },
        "inside_input_cols": list(INSIDE_INPUT_COLS),
        "outdoor_input_cols": list(OUTDOOR_INPUT_COLS),
        "inside_clip_bounds_scaled": list(inside_clip),
        "outdoor_clip_bounds_scaled": list(outdoor_clip),
        "data_audit": audit_data(df, cfg),
        "arx_inside_fixed": {
            "selected_by_validation": arx_inside_fixed["selected_by_validation"],
            "validation": arx_inside_fixed["validation"],
            "test": arx_inside_fixed["test"],
        },
        "arx_inside": {
            "leaderboard_top20": arx_inside["leaderboard"].head(20).to_dict(orient="records"),
            "selected_by_validation": arx_inside["selected_by_validation"],
            "validation": arx_inside["validation"],
            "test": arx_inside["test"],
        },
        "arx_outdoor": {
            "leaderboard_top20": arx_outdoor["leaderboard"].head(20).to_dict(orient="records"),
            "selected_by_validation": arx_outdoor["selected_by_validation"],
            "validation": arx_outdoor["validation"],
            "test": arx_outdoor["test"],
        },
        "arx_outdoor_one_step_residual_diagnostics": arx_outdoor_one_diag,
        "arx_outdoor_sim_residual_diagnostics": arx_outdoor_sim_diag,
        "comparison": comparison,
    }

    df.to_csv(RESULTS_DIR / "mini_greenhouse_20s_outdoor_data.csv", index=False)
    arx_inside["leaderboard"].to_csv(RESULTS_DIR / "arx_inside_leaderboard.csv", index=False)
    arx_outdoor["leaderboard"].to_csv(RESULTS_DIR / "arx_outdoor_leaderboard.csv", index=False)
    pd.DataFrame(comparison).to_csv(RESULTS_DIR / "comparison.csv", index=False)
    test_predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
        f.write("\n")
    write_report(payload)
    write_self_critique(payload)
    return payload


def main() -> None:
    payload = run_pipeline(OutdoorConfig())
    rows = payload["comparison"]
    inside = next(row for row in rows if row["case"] == "ARX robust chỉ cảm biến trong")
    outdoor = next(row for row in rows if row["case"] == "ARX robust có cảm biến ngoài")
    improvement = outdoor["test_FIT_sim"] - inside["test_FIT_sim"]
    print("=== Mini Greenhouse 20 giây: ARX có cảm biến ngoài ===")
    print(f"ARX trong:   {inside['selected_model']} FIT_sim={inside['test_FIT_sim']:.3f} FIT_60={inside['test_FIT_60_20min']:.3f}")
    print(f"ARX ngoài:   {outdoor['selected_model']} FIT_sim={outdoor['test_FIT_sim']:.3f} FIT_60={outdoor['test_FIT_60_20min']:.3f}")
    print(f"Cải thiện FIT_sim khi thêm cảm biến ngoài: {improvement:.3f} điểm")
    print(f"Đã lưu kết quả vào {RESULTS_DIR}")


if __name__ == "__main__":
    main()
