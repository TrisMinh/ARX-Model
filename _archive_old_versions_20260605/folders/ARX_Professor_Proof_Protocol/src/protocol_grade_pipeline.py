from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class ProtocolConfig:
    days: int = 60
    sampling_seconds: int = 300
    seed: int = 2605
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    n_step: int = 12
    validation_blocks: int = 4
    validation_std_penalty: float = 0.5
    soil_low_sp: float = 55.0
    soil_high_sp: float = 65.0


@dataclass(frozen=True)
class ArxSpec:
    na: int
    nb: int
    nk: int
    alpha: float

    @property
    def name(self) -> str:
        return f"ARX_na{self.na}_nb{self.nb}_nk{self.nk}_alpha{self.alpha:g}"


@dataclass(frozen=True)
class NarxSpec:
    name: str
    y_lags: tuple[int, ...]
    input_lags: tuple[int, ...]
    hidden_layer_sizes: tuple[int, ...]
    alpha: float
    learning_rate_init: float
    random_state: int

    @property
    def start_lag(self) -> int:
        return max(max(self.y_lags), max(self.input_lags))


INPUT_COLS = (
    "Temperature",
    "Humidity",
    "Light",
    "Drip",
    "Mist",
    "Fan",
    "Light_log",
    "Temp_x_Humi",
    "Temp_x_Light",
    "Humi_x_Light",
    "SP_Center",
    "SP_Width",
    "Hour_sin",
    "Hour_cos",
    "Day_sin",
    "Day_cos",
)


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


def month_to_season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def phase_for_day(day: int) -> str:
    if day < 7:
        return "commissioning_rule_based"
    if day < 42:
        return "identification_safe_excitation"
    return "deployment_validation"


def make_planned_excitation(cfg: ProtocolConfig, rng: np.random.Generator, n_rows: int) -> dict[str, np.ndarray]:
    samples_per_day = int(round(24 * 3600 / cfg.sampling_seconds))
    planned_drip = np.zeros(n_rows, dtype=float)
    planned_mist = np.zeros(n_rows, dtype=float)
    planned_fan = np.zeros(n_rows, dtype=float)
    planned_label = np.full(n_rows, "none", dtype=object)

    # Small planned pulses. These are scheduled from clock/random seed, not from soil moisture.
    drip_windows = (7.0, 12.0, 17.0, 21.0)
    for day in range(cfg.days):
        phase = phase_for_day(day)
        if phase == "commissioning_rule_based":
            continue
        day_base = day * samples_per_day
        is_test_phase = phase == "deployment_validation"
        pulse_prob = 0.70 if not is_test_phase else 0.45
        for center_hour in drip_windows:
            if rng.random() < pulse_prob:
                start_hour = center_hour + rng.normal(0.0, 0.28)
                start = day_base + int(np.clip(round(start_hour * 3600 / cfg.sampling_seconds), 0, samples_per_day - 1))
                duration = int(rng.integers(2, 6 if not is_test_phase else 5))
                end = min(day_base + samples_per_day, start + duration)
                planned_drip[start:end] = 1.0
                planned_label[start:end] = "planned_drip_excitation"

        # Fan/mist probes are also clock/weather-experiment style, not soil-driven.
        for center_hour, actuator in ((10.5, "mist"), (14.0, "fan")):
            if rng.random() < (0.30 if not is_test_phase else 0.18):
                start_hour = center_hour + rng.normal(0.0, 0.35)
                start = day_base + int(np.clip(round(start_hour * 3600 / cfg.sampling_seconds), 0, samples_per_day - 1))
                duration = int(rng.integers(1, 4))
                end = min(day_base + samples_per_day, start + duration)
                if actuator == "mist":
                    planned_mist[start:end] = 1.0
                    planned_label[start:end] = "planned_mist_probe"
                else:
                    planned_fan[start:end] = 1.0
                    planned_label[start:end] = "planned_fan_probe"

    return {
        "planned_drip": planned_drip,
        "planned_mist": planned_mist,
        "planned_fan": planned_fan,
        "planned_label": planned_label,
    }


def generate_protocol_data(cfg: ProtocolConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    samples_per_day = int(round(24 * 3600 / cfg.sampling_seconds))
    n_rows = cfg.days * samples_per_day
    t = np.arange(n_rows)
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq=f"{cfg.sampling_seconds}s")
    hour = (t % samples_per_day) / samples_per_day * 24.0
    day_index = t // samples_per_day
    day_fraction = day_index / max(cfg.days - 1, 1)
    months = timestamps.month.to_numpy()
    seasons = np.asarray([month_to_season(int(m)) for m in months])
    phase = np.asarray([phase_for_day(int(d)) for d in day_index])

    day_temp_offset = rng.normal(0.0, 1.0, cfg.days + 1)
    day_humi_offset = rng.normal(0.0, 2.2, cfg.days + 1)
    day_cloud = rng.uniform(0.72, 1.12, cfg.days + 1)

    daylight = np.maximum(0.0, np.sin((hour - 6.0) / 12.0 * np.pi))
    outside_light = np.clip(
        18.0 + 880.0 * daylight * day_cloud[day_index] + rng.normal(0.0, 12.0, n_rows),
        0.0,
        1250.0,
    )
    outside_temp = (
        27.0
        + 5.8 * np.sin((hour - 8.0) / 24.0 * 2.0 * np.pi)
        + 1.3 * np.sin(2.0 * np.pi * day_fraction)
        + day_temp_offset[day_index]
        + rng.normal(0.0, 0.35, n_rows)
    )
    outside_humi = np.clip(
        73.0
        - 9.5 * np.sin((hour - 8.0) / 24.0 * 2.0 * np.pi)
        - 2.5 * np.sin(2.0 * np.pi * day_fraction)
        + day_humi_offset[day_index]
        + rng.normal(0.0, 1.1, n_rows),
        38.0,
        98.0,
    )

    planned = make_planned_excitation(cfg, rng, n_rows)
    drip = np.zeros(n_rows, dtype=float)
    mist = np.zeros(n_rows, dtype=float)
    fan = np.zeros(n_rows, dtype=float)
    source = np.full(n_rows, "none", dtype=object)
    safety_override = np.zeros(n_rows, dtype=int)

    indoor_temp = np.zeros(n_rows, dtype=float)
    indoor_humi = np.zeros(n_rows, dtype=float)
    soil_true = np.zeros(n_rows, dtype=float)
    soil_meas = np.zeros(n_rows, dtype=float)

    indoor_temp[:3] = outside_temp[:3]
    indoor_humi[:3] = outside_humi[:3]
    soil_true[:3] = (59.0, 58.9, 58.8)
    soil_meas[:3] = soil_true[:3] + rng.normal(0.0, 0.10, 3)

    drip_hold = 0
    for idx in range(3, n_rows):
        low_sp = cfg.soil_low_sp
        high_sp = cfg.soil_high_sp

        # Safety supervisor uses measured past soil only. It can block planned pulses or add rescue irrigation.
        # Safety has a tolerance band around the agronomic target. This allows
        # small identification pulses while still preventing dry/wet extremes.
        rescue = soil_meas[idx - 1] < (low_sp - 0.8)
        too_wet = soil_meas[idx - 1] > (high_sp + 0.8)
        if drip_hold > 0:
            drip[idx] = 1.0
            source[idx] = "safety_rescue_hold"
            drip_hold -= 1
        elif rescue:
            drip[idx] = 1.0
            source[idx] = "safety_rescue"
            safety_override[idx] = 1
            drip_hold = 1
        elif too_wet and planned["planned_drip"][idx] > 0.5:
            drip[idx] = 0.0
            source[idx] = "planned_drip_blocked_wet"
            safety_override[idx] = 1
        elif planned["planned_drip"][idx] > 0.5:
            drip[idx] = 1.0
            source[idx] = "planned_drip_excitation"

        fan_rule = (indoor_temp[idx - 1] > 30.2) or (indoor_humi[idx - 1] > 87.0)
        fan[idx] = 1.0 if fan_rule or planned["planned_fan"][idx] > 0.5 else 0.0

        mist_rule = (indoor_temp[idx - 1] > 31.0) and (indoor_humi[idx - 1] < 67.0) and (9.0 <= hour[idx] <= 16.0)
        mist[idx] = 1.0 if mist_rule or planned["planned_mist"][idx] > 0.5 else 0.0

        if source[idx] == "none" and planned["planned_label"][idx] != "none":
            source[idx] = planned["planned_label"][idx]

        indoor_temp[idx] = (
            0.88 * indoor_temp[idx - 1]
            + 0.12 * outside_temp[idx]
            - 0.75 * fan[idx]
            - 0.45 * mist[idx]
            + rng.normal(0.0, 0.18)
        )
        indoor_humi[idx] = (
            0.86 * indoor_humi[idx - 1]
            + 0.14 * outside_humi[idx]
            + 5.8 * mist[idx]
            - 2.4 * fan[idx]
            + rng.normal(0.0, 0.45)
        )
        indoor_temp[idx] = float(np.clip(indoor_temp[idx], 15.0, 42.0))
        indoor_humi[idx] = float(np.clip(indoor_humi[idx], 35.0, 100.0))

        evap = (
            0.024
            + 0.0052 * max(0.0, indoor_temp[idx - 2] - 24.0)
            + 0.000062 * outside_light[idx - 2]
            - 0.0025 * max(0.0, indoor_humi[idx - 2] - 70.0)
        )
        evap = max(0.004, evap)
        water = 0.55 * drip[idx - 1] + 0.42 * drip[idx - 2] + 0.20 * drip[idx - 3]
        micro_water = 0.030 * mist[idx - 1]
        fan_dry = 0.010 * fan[idx - 1] + 0.006 * fan[idx - 2]
        drainage = 0.040 * max(0.0, soil_true[idx - 1] - 65.0)
        dry_soil_slowdown = 0.45 if soil_true[idx - 1] < 53.5 else 1.0
        nonlinear_heat_dry = 0.00008 * max(0.0, indoor_temp[idx - 2] - 30.0) * outside_light[idx - 2] / 100.0
        soil_true[idx] = (
            soil_true[idx - 1]
            + water
            + micro_water
            - dry_soil_slowdown * evap
            - fan_dry
            - drainage
            - nonlinear_heat_dry
            + 0.003 * (58.0 - soil_true[idx - 1])
            + rng.normal(0.0, 0.030)
        )
        soil_true[idx] = float(np.clip(soil_true[idx], 35.0, 85.0))
        soil_meas[idx] = float(np.clip(soil_true[idx] + rng.normal(0.0, 0.11), 0.0, 100.0))

    low_sp = np.full(n_rows, cfg.soil_low_sp, dtype=float)
    high_sp = np.full(n_rows, cfg.soil_high_sp, dtype=float)
    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Day_Index": day_index,
            "Hour": hour,
            "Month": months,
            "Season": seasons,
            "Protocol_Phase": phase,
            "Soil_Moisture_True": soil_true,
            "Soil_Moisture": soil_meas,
            "Soil_Low_SP": low_sp,
            "Soil_High_SP": high_sp,
            "Temperature": indoor_temp,
            "Humidity": indoor_humi,
            "Light": outside_light,
            "Drip": drip,
            "Mist": mist,
            "Fan": fan,
            "Command_Source": source,
            "Planned_Drip": planned["planned_drip"],
            "Planned_Mist": planned["planned_mist"],
            "Planned_Fan": planned["planned_fan"],
            "Safety_Override": safety_override,
        }
    )


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    timestamp = pd.to_datetime(df["Timestamp"])
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    day_num = df["Day_Index"].astype(float)
    df["Light_log"] = np.log1p(df["Light"].clip(lower=0))
    df["Temp_x_Humi"] = df["Temperature"] * df["Humidity"]
    df["Temp_x_Light"] = df["Temperature"] * df["Light_log"]
    df["Humi_x_Light"] = df["Humidity"] * df["Light_log"]
    df["SP_Center"] = 0.5 * (df["Soil_Low_SP"] + df["Soil_High_SP"])
    df["SP_Width"] = df["Soil_High_SP"] - df["Soil_Low_SP"]
    df["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["Day_sin"] = np.sin(2.0 * np.pi * day_num / 30.0)
    df["Day_cos"] = np.cos(2.0 * np.pi * day_num / 30.0)
    return df


def split_time(df: pd.DataFrame, cfg: ProtocolConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    n_train = int(n_rows * cfg.train_ratio)
    n_val = int(n_rows * cfg.val_ratio)
    return (
        df.iloc[:n_train].reset_index(drop=True),
        df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        df.iloc[n_train + n_val :].reset_index(drop=True),
    )


def fit_scale_stats(df_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in ("Soil_Moisture", *INPUT_COLS):
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


def build_arx_matrix(df_z: pd.DataFrame, spec: ArxSpec) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    lag = arx_max_lag(spec)
    cols: list[np.ndarray] = []
    for y_lag in range(1, spec.na + 1):
        cols.append(y[lag - y_lag : len(y) - y_lag])
    for col in INPUT_COLS:
        values = df_z[col].to_numpy(dtype=float)
        for u_lag in range(spec.nk, spec.nk + spec.nb):
            cols.append(values[lag - u_lag : len(values) - u_lag])
    cols.append(np.ones(len(y) - lag))
    return np.vstack(cols).T, y[lag:]


def fit_arx(df_train_z: pd.DataFrame, spec: ArxSpec) -> np.ndarray:
    x_train, y_train = build_arx_matrix(df_train_z, spec)
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


def arx_predict_at(y_source: np.ndarray, input_arrays: list[np.ndarray], t: int, theta: np.ndarray, spec: ArxSpec) -> float:
    idx = 0
    y_next = 0.0
    for y_lag in range(1, spec.na + 1):
        y_next += theta[idx] * y_source[t - y_lag]
        idx += 1
    for values in input_arrays:
        for u_lag in range(spec.nk, spec.nk + spec.nb):
            y_next += theta[idx] * values[t - u_lag]
            idx += 1
    y_next += theta[idx]
    return float(y_next)


def predict_arx_one_step(df_z: pd.DataFrame, theta: np.ndarray, spec: ArxSpec, clip: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x, y_true = build_arx_matrix(df_z, spec)
    return np.clip(x @ theta, clip[0], clip[1]), y_true


def simulate_arx(df_z: pd.DataFrame, theta: np.ndarray, spec: ArxSpec, clip: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_sim = y.copy()
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in INPUT_COLS]
    lag = arx_max_lag(spec)
    for t in range(lag, len(y)):
        y_sim[t] = float(np.clip(arx_predict_at(y_sim, input_arrays, t, theta, spec), clip[0], clip[1]))
    return y_sim[lag:], y[lag:]


def simulate_arx_n_step(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    clip: tuple[float, float],
    n_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_work = y.copy()
    y_pred = y.copy()
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in INPUT_COLS]
    lag = arx_max_lag(spec)
    for t in range(lag, len(y)):
        origin = max(lag - 1, t - n_step)
        reset_start = max(0, origin - spec.na)
        y_work[reset_start : t + 1] = y[reset_start : t + 1]
        for step_t in range(origin + 1, t + 1):
            y_work[step_t] = float(np.clip(arx_predict_at(y_work, input_arrays, step_t, theta, spec), clip[0], clip[1]))
        y_pred[t] = y_work[t]
    return y_pred[lag:], y[lag:]


def evaluate_arx_split(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    n_step: int,
) -> dict[str, dict[str, float]]:
    y_1, y_true_1 = predict_arx_one_step(df_z, theta, spec, clip)
    y_12, y_true_12 = simulate_arx_n_step(df_z, theta, spec, clip, n_step)
    y_sim, y_true_sim = simulate_arx(df_z, theta, spec, clip)
    return {
        "metrics_1step": fit_metrics(inverse_y(y_true_1, stats), inverse_y(y_1, stats)),
        "metrics_12": fit_metrics(inverse_y(y_true_12, stats), inverse_y(y_12, stats)),
        "metrics_sim": fit_metrics(inverse_y(y_true_sim, stats), inverse_y(y_sim, stats)),
    }


def validation_block_score_arx(
    df_val_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    cfg: ProtocolConfig,
) -> dict[str, Any]:
    scores: list[float] = []
    for block in np.array_split(np.arange(len(df_val_z)), cfg.validation_blocks):
        df_block = df_val_z.iloc[int(block[0]) : int(block[-1]) + 1].reset_index(drop=True)
        y_sim, y_true = simulate_arx(df_block, theta, spec, clip)
        scores.append(fit_metrics(inverse_y(y_true, stats), inverse_y(y_sim, stats))["FIT"])
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
    for na in (2, 3, 5, 8, 12):
        for nb in (1, 2, 3, 5, 8):
            for nk in (1, 2, 3, 4):
                for alpha in (0.0, 0.001, 0.01, 0.1):
                    specs.append(ArxSpec(na, nb, nk, alpha))
    return specs


def run_arx_search(
    df_train_z: pd.DataFrame,
    df_val_z: pd.DataFrame,
    df_test_z: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    cfg: ProtocolConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fitted: dict[str, tuple[ArxSpec, np.ndarray]] = {}
    for spec in arx_specs():
        theta = fit_arx(df_train_z, spec)
        val_eval = evaluate_arx_split(df_val_z, theta, spec, stats, clip, cfg.n_step)
        robust = validation_block_score_arx(df_val_z, theta, spec, stats, clip, cfg)
        fitted[spec.name] = (spec, theta)
        rows.append(
            {
                "model": spec.name,
                "na": spec.na,
                "nb": spec.nb,
                "nk": spec.nk,
                "alpha": spec.alpha,
                "n_params": int(len(theta)),
                "val_FIT_1step": val_eval["metrics_1step"]["FIT"],
                "val_FIT_12": val_eval["metrics_12"]["FIT"],
                "val_FIT_sim": val_eval["metrics_sim"]["FIT"],
                **robust,
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values("val_robust_score", ascending=False).reset_index(drop=True)
    selected = leaderboard.iloc[0].to_dict()
    spec, theta = fitted[str(selected["model"])]
    return {
        "leaderboard": leaderboard,
        "selected_by_validation": selected,
        "theta": theta,
        "spec": spec,
        "validation": evaluate_arx_split(df_val_z, theta, spec, stats, clip, cfg.n_step),
        "test": evaluate_arx_split(df_test_z, theta, spec, stats, clip, cfg.n_step),
    }


def narx_specs() -> list[NarxSpec]:
    return [
        NarxSpec("Delta_NNARX_32_16_a01", (1, 2, 3, 6, 12), (1, 2, 3, 6, 12), (32, 16), 0.01, 0.0008, 11),
        NarxSpec("Delta_NNARX_64_32_a01", (1, 2, 3, 6, 12), (1, 2, 3, 6, 12), (64, 32), 0.01, 0.0006, 12),
        NarxSpec("Delta_NNARX_64_32_a05", (1, 2, 3, 6, 12), (1, 2, 3, 6, 12), (64, 32), 0.05, 0.0006, 13),
    ]


def build_narx_matrix(df_z: pd.DataFrame, spec: NarxSpec) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in INPUT_COLS]
    lag = spec.start_lag
    n_rows = len(y) - lag
    n_cols = len(spec.y_lags) + len(INPUT_COLS) * len(spec.input_lags)
    x = np.empty((n_rows, n_cols), dtype=float)
    target = np.empty(n_rows, dtype=float)
    for row_idx, t in enumerate(range(lag, len(y))):
        row: list[float] = []
        row.extend(float(y[t - y_lag]) for y_lag in spec.y_lags)
        for values in input_arrays:
            row.extend(float(values[t - u_lag]) for u_lag in spec.input_lags)
        x[row_idx] = row
        target[row_idx] = y[t] - y[t - 1]
    return x, target


def mlp_predict_one(model: MLPRegressor, row: np.ndarray) -> float:
    values = row.reshape(1, -1)
    for idx, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        values = values @ weights + bias
        if idx < len(model.coefs_) - 1:
            values = np.maximum(values, 0.0)
    return float(values.ravel()[0])


def narx_row_from_state(y_source: np.ndarray, input_arrays: list[np.ndarray], t: int, spec: NarxSpec) -> np.ndarray:
    row: list[float] = []
    row.extend(float(y_source[t - y_lag]) for y_lag in spec.y_lags)
    for values in input_arrays:
        row.extend(float(values[t - u_lag]) for u_lag in spec.input_lags)
    return np.asarray(row, dtype=float)


def predict_narx_one_step(df_z: pd.DataFrame, model: MLPRegressor, spec: NarxSpec, clip: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x, _ = build_narx_matrix(df_z, spec)
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    delta = model.predict(x)
    y_pred = np.asarray(
        [np.clip(y[t - 1] + delta[row_idx], clip[0], clip[1]) for row_idx, t in enumerate(range(spec.start_lag, len(y)))],
        dtype=float,
    )
    return y_pred, y[spec.start_lag :]


def simulate_narx(df_z: pd.DataFrame, model: MLPRegressor, spec: NarxSpec, clip: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_sim = y.copy()
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in INPUT_COLS]
    for t in range(spec.start_lag, len(y)):
        row = narx_row_from_state(y_sim, input_arrays, t, spec)
        delta = mlp_predict_one(model, row)
        y_sim[t] = float(np.clip(y_sim[t - 1] + delta, clip[0], clip[1]))
    return y_sim[spec.start_lag :], y[spec.start_lag :]


def simulate_narx_n_step(
    df_z: pd.DataFrame,
    model: MLPRegressor,
    spec: NarxSpec,
    clip: tuple[float, float],
    n_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_work = y.copy()
    y_pred = y.copy()
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in INPUT_COLS]
    for t in range(spec.start_lag, len(y)):
        origin = max(spec.start_lag - 1, t - n_step)
        reset_start = max(0, origin - max(spec.y_lags))
        y_work[reset_start : t + 1] = y[reset_start : t + 1]
        for step_t in range(origin + 1, t + 1):
            row = narx_row_from_state(y_work, input_arrays, step_t, spec)
            delta = mlp_predict_one(model, row)
            y_work[step_t] = float(np.clip(y_work[step_t - 1] + delta, clip[0], clip[1]))
        y_pred[t] = y_work[t]
    return y_pred[spec.start_lag :], y[spec.start_lag :]


def evaluate_narx_split(
    df_z: pd.DataFrame,
    model: MLPRegressor,
    spec: NarxSpec,
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    n_step: int,
) -> dict[str, dict[str, float]]:
    y_1, y_true_1 = predict_narx_one_step(df_z, model, spec, clip)
    y_12, y_true_12 = simulate_narx_n_step(df_z, model, spec, clip, n_step)
    y_sim, y_true_sim = simulate_narx(df_z, model, spec, clip)
    return {
        "metrics_1step": fit_metrics(inverse_y(y_true_1, stats), inverse_y(y_1, stats)),
        "metrics_12": fit_metrics(inverse_y(y_true_12, stats), inverse_y(y_12, stats)),
        "metrics_sim": fit_metrics(inverse_y(y_true_sim, stats), inverse_y(y_sim, stats)),
    }


def validation_block_score_narx(
    df_val_z: pd.DataFrame,
    model: MLPRegressor,
    spec: NarxSpec,
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    cfg: ProtocolConfig,
) -> dict[str, Any]:
    scores: list[float] = []
    for block in np.array_split(np.arange(len(df_val_z)), cfg.validation_blocks):
        df_block = df_val_z.iloc[int(block[0]) : int(block[-1]) + 1].reset_index(drop=True)
        y_sim, y_true = simulate_narx(df_block, model, spec, clip)
        scores.append(fit_metrics(inverse_y(y_true, stats), inverse_y(y_sim, stats))["FIT"])
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=0))
    return {
        "val_block_FIT_sim": scores,
        "val_block_mean_FIT_sim": mean_score,
        "val_block_std_FIT_sim": std_score,
        "val_robust_score": mean_score - cfg.validation_std_penalty * std_score,
    }


def run_narx_search(
    df_train_z: pd.DataFrame,
    df_val_z: pd.DataFrame,
    df_test_z: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    cfg: ProtocolConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fitted: dict[str, tuple[NarxSpec, MLPRegressor]] = {}
    for spec in narx_specs():
        x_train, y_train = build_narx_matrix(df_train_z, spec)
        model = MLPRegressor(
            hidden_layer_sizes=spec.hidden_layer_sizes,
            activation="relu",
            alpha=spec.alpha,
            learning_rate_init=spec.learning_rate_init,
            max_iter=220,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=18,
            random_state=spec.random_state,
        )
        model.fit(x_train, y_train)
        val_eval = evaluate_narx_split(df_val_z, model, spec, stats, clip, cfg.n_step)
        robust = validation_block_score_narx(df_val_z, model, spec, stats, clip, cfg)
        rows.append(
            {
                "model": spec.name,
                "n_features": int(x_train.shape[1]),
                "hidden_layer_sizes": list(spec.hidden_layer_sizes),
                "alpha": spec.alpha,
                "val_FIT_1step": val_eval["metrics_1step"]["FIT"],
                "val_FIT_12": val_eval["metrics_12"]["FIT"],
                "val_FIT_sim": val_eval["metrics_sim"]["FIT"],
                **robust,
            }
        )
        fitted[spec.name] = (spec, model)
    leaderboard = pd.DataFrame(rows).sort_values("val_robust_score", ascending=False).reset_index(drop=True)
    selected = leaderboard.iloc[0].to_dict()
    spec, model = fitted[str(selected["model"])]
    return {
        "leaderboard": leaderboard,
        "selected_by_validation": selected,
        "spec": spec,
        "model": model,
        "validation": evaluate_narx_split(df_val_z, model, spec, stats, clip, cfg.n_step),
        "test": evaluate_narx_split(df_test_z, model, spec, stats, clip, cfg.n_step),
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
    if int(mask.sum()) == 0:
        return None
    return float(100.0 * np.mean(values[mask] > 0.5))


def audit_data(df: pd.DataFrame, cfg: ProtocolConfig) -> dict[str, Any]:
    df_sorted = df.sort_values("Timestamp").reset_index(drop=True)
    diffs = pd.to_datetime(df_sorted["Timestamp"]).diff().dt.total_seconds().dropna()
    y_prev = df_sorted["Soil_Moisture"].to_numpy(dtype=float)[:-1]
    low_prev = df_sorted["Soil_Low_SP"].to_numpy(dtype=float)[:-1]
    high_prev = df_sorted["Soil_High_SP"].to_numpy(dtype=float)[:-1]
    center_prev = 0.5 * (low_prev + high_prev)
    drip_now = df_sorted["Drip"].to_numpy(dtype=float)[1:]
    planned_drip_now = df_sorted["Planned_Drip"].to_numpy(dtype=float)[1:]
    below_low = y_prev < low_prev
    safe_mid = (y_prev >= low_prev + 1.0) & (y_prev <= high_prev - 1.5)
    above_high = y_prev > high_prev
    train, val, test = split_time(df_sorted, cfg)

    def split_summary(name: str, part: pd.DataFrame) -> dict[str, Any]:
        return {
            "name": name,
            "rows": int(len(part)),
            "start": str(part["Timestamp"].iloc[0]),
            "end": str(part["Timestamp"].iloc[-1]),
            "soil_mean": float(part["Soil_Moisture"].mean()),
            "soil_std": float(part["Soil_Moisture"].std(ddof=0)),
            "drip_on_pct": float(100.0 * part["Drip"].mean()),
            "planned_drip_pct": float(100.0 * part["Planned_Drip"].mean()),
        }

    return {
        "rows": int(len(df_sorted)),
        "start": str(df_sorted["Timestamp"].iloc[0]),
        "end": str(df_sorted["Timestamp"].iloc[-1]),
        "missing_total": int(df_sorted.isna().sum().sum()),
        "duplicate_timestamps": int(df_sorted["Timestamp"].duplicated().sum()),
        "median_sampling_seconds": float(diffs.median()),
        "irregular_sampling_count": int((diffs != diffs.median()).sum()),
        "phase_counts": {str(k): int(v) for k, v in df_sorted["Protocol_Phase"].value_counts().to_dict().items()},
        "command_source_counts": {str(k): int(v) for k, v in df_sorted["Command_Source"].value_counts().to_dict().items()},
        "safety_override_pct": float(100.0 * df_sorted["Safety_Override"].mean()),
        "drip_on_pct": float(100.0 * df_sorted["Drip"].mean()),
        "planned_drip_pct": float(100.0 * df_sorted["Planned_Drip"].mean()),
        "p_drip_on_when_prev_soil_below_low_pct": p_on(below_low, drip_now),
        "p_drip_on_when_prev_soil_safe_mid_pct": p_on(safe_mid, drip_now),
        "p_drip_on_when_prev_soil_above_high_pct": p_on(above_high, drip_now),
        "p_planned_drip_when_prev_soil_safe_mid_pct": p_on(safe_mid, planned_drip_now),
        "corr_drip_t_with_prev_soil_minus_center": safe_corr(drip_now, y_prev - center_prev),
        "corr_planned_drip_t_with_prev_soil_minus_center": safe_corr(planned_drip_now, y_prev - center_prev),
        "split": {
            "train": split_summary("train", train),
            "validation": split_summary("validation", val),
            "test": split_summary("test", test),
        },
    }


def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, max_lag: int = 24) -> dict[str, Any]:
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    acf: list[float] = []
    for lag in range(1, max_lag + 1):
        acf.append(safe_corr(residual[lag:], residual[:-lag]) or 0.0)
    return {
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "max_abs_acf_lag_1_to_24": float(np.max(np.abs(acf))),
        "acf_lag_1_to_24": acf,
    }


def comparison_row(name: str, family: str, result: dict[str, Any], mpc_note: str) -> dict[str, Any]:
    test = result["test"]
    selected = result["selected_by_validation"]
    return {
        "case": name,
        "family": family,
        "selected_model": selected["model"],
        "selection_policy": "validation robust score",
        "test_FIT_1step": test["metrics_1step"]["FIT"],
        "test_FIT_12": test["metrics_12"]["FIT"],
        "test_FIT_sim": test["metrics_sim"]["FIT"],
        "test_RMSE_sim": test["metrics_sim"]["RMSE"],
        "mpc_note": mpc_note,
    }


def write_report(payload: dict[str, Any]) -> None:
    data = payload["data_audit"]
    arx = payload["arx"]
    narx = payload["narx"]
    rows = payload["comparison"]
    arx_test = arx["test"]
    selected_arx = arx["selected_by_validation"]
    selected_narx = narx["selected_by_validation"]
    diff = rows[1]["test_FIT_sim"] - rows[0]["test_FIT_sim"]

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

    lines = [
        "# Protocol-Grade ARX/NARX Final Report",
        "",
        "## Ket luan",
        "",
        "- Data duoc sinh theo quy trinh thu thap thuc te: rule-based safety truoc, planned excitation nho/an toan sau.",
        "- Plant simulation tach rieng voi model ARX, co nonlinear evaporation, drainage, sensor noise va actuator dynamics.",
        "- ARX la model chinh vi phu hop MPC tuyen tinh/RLS va de giai thich.",
        "- NARX chi la doi chung phi tuyen; khong thay truc tiep ARX trong MPC neu chua doi sang NMPC/local linearization.",
        f"- Model ARX tot nhat: `{selected_arx['model']}`.",
        "",
        "## Data audit",
        "",
        "| Check | Value |",
        "| --- | ---: |",
        f"| Rows | {data['rows']} |",
        f"| Missing total | {data['missing_total']} |",
        f"| Duplicate timestamps | {data['duplicate_timestamps']} |",
        f"| Median sampling seconds | {fmt(data['median_sampling_seconds'], 0)} |",
        f"| Irregular sampling count | {data['irregular_sampling_count']} |",
        f"| Drip ON % | {fmt(data['drip_on_pct'])} |",
        f"| Planned Drip % | {fmt(data['planned_drip_pct'])} |",
        f"| Safety override % | {fmt(data['safety_override_pct'])} |",
        f"| P(Drip ON | prev soil below low) | {fmt(data['p_drip_on_when_prev_soil_below_low_pct'])} |",
        f"| P(Drip ON | prev soil safe mid) | {fmt(data['p_drip_on_when_prev_soil_safe_mid_pct'])} |",
        f"| corr(Drip, prev soil-center) | {fmt(data['corr_drip_t_with_prev_soil_minus_center'])} |",
        f"| corr(Planned Drip, prev soil-center) | {fmt(data['corr_planned_drip_t_with_prev_soil_minus_center'])} |",
        "",
        "Diem quan trong: `Planned_Drip` la lich excitation sinh tu clock/random seed, khong tu soil moisture. `Drip` thuc te van co safety override nen co the phu thuoc soil, dung voi he that.",
        "",
        "## Model comparison",
        "",
        "| Case | Family | Selected | Test FIT_1step | Test FIT_12 | Test FIT_sim | Test RMSE_sim | MPC note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['case']} | {row['family']} | `{row['selected_model']}` | "
            f"{fmt(row['test_FIT_1step'])} | {fmt(row['test_FIT_12'])} | "
            f"{fmt(row['test_FIT_sim'])} | {fmt(row['test_RMSE_sim'], 4)} | {row['mpc_note']} |"
        )

    lines.extend(
        [
            "",
            "## ARX details",
            "",
            f"- Selected by robust validation score: `{selected_arx['val_robust_score']:.3f}`.",
            f"- Validation block FIT_sim: `{[round(v, 3) for v in selected_arx['val_block_FIT_sim']]}`.",
            f"- Test one-step residual max abs ACF lag 1..24: `{payload['arx_one_step_residual_diagnostics']['max_abs_acf_lag_1_to_24']:.3f}`.",
            f"- Test free-run residual max abs ACF lag 1..24: `{payload['arx_sim_residual_diagnostics']['max_abs_acf_lag_1_to_24']:.3f}`. Free-run residual tu tuong quan cao hon vi loi duoc tich luy theo thoi gian.",
            f"- Clip bounds are learned from train 0.5%-99.5% scaled quantiles: `{[round(v, 3) for v in payload['clip_bounds_scaled']]}`.",
            "",
            "## NARX details",
            "",
            f"- Selected NARX: `{selected_narx['model']}`.",
            f"- NARX - ARX FIT_sim difference: `{diff:.3f}` points.",
            "",
            "## Cach bao ve voi thay",
            "",
            "1. Khong noi data nay la data that. Noi day la protocol-grade simulation de kiem thu quy trinh; khi co nha kinh mini se log cung schema va retrain.",
            "2. Neu thay hoi AI dieu khien the nao khi chua co data: tra loi rule-based safety thu data truoc, ARX/MPC dung sau.",
            "3. Neu thay hoi vi sao co pulse excitation: do la persistent excitation de nhan dang he, co safety supervisor nen khong nguy hiem.",
            "4. Neu thay hoi vi sao khong dung NARX: vi ARX dat muc tot, giai thich duoc, va dung voi MPC tuyen tinh; NARX khong plug-compatible.",
            "",
        ]
    )
    (RESULTS_DIR / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(cfg: ProtocolConfig) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = generate_protocol_data(cfg)
    df = add_features(df_raw)
    df_train, df_val, df_test = split_time(df, cfg)
    stats = fit_scale_stats(df_train)
    df_train_z = apply_scale(df_train, stats)
    df_val_z = apply_scale(df_val, stats)
    df_test_z = apply_scale(df_test, stats)
    clip = tuple(float(v) for v in np.quantile(df_train_z["Soil_Moisture"], (0.005, 0.995)))

    arx = run_arx_search(df_train_z, df_val_z, df_test_z, stats, clip, cfg)
    narx = run_narx_search(df_train_z, df_val_z, df_test_z, stats, clip, cfg)

    arx_one_z, arx_true_one_z = predict_arx_one_step(df_test_z, arx["theta"], arx["spec"], clip)
    arx_sim_z, arx_true_z = simulate_arx(df_test_z, arx["theta"], arx["spec"], clip)
    arx_one_diag = residual_diagnostics(inverse_y(arx_true_one_z, stats), inverse_y(arx_one_z, stats))
    arx_sim_diag = residual_diagnostics(inverse_y(arx_true_z, stats), inverse_y(arx_sim_z, stats))

    narx_sim_z, narx_true_z = simulate_narx(df_test_z, narx["model"], narx["spec"], clip)
    lag_arx = arx_max_lag(arx["spec"])
    lag_narx = narx["spec"].start_lag
    test_predictions = pd.DataFrame(
        {
            "Timestamp_ARX": df_test["Timestamp"].iloc[lag_arx:].to_numpy(),
            "y_true_arx_window": inverse_y(arx_true_z, stats),
            "y_arx_sim": inverse_y(arx_sim_z, stats),
        }
    )
    narx_pred_df = pd.DataFrame(
        {
            "Timestamp_NARX": df_test["Timestamp"].iloc[lag_narx:].to_numpy(),
            "y_true_narx_window": inverse_y(narx_true_z, stats),
            "y_narx_sim": inverse_y(narx_sim_z, stats),
        }
    )
    test_predictions = pd.concat([test_predictions, narx_pred_df], axis=1)

    comparison = [
        comparison_row("ARX robust protocol", "Linear ARX", arx, "MPC-friendly linear plant model"),
        comparison_row("NARX protocol comparison", "Nonlinear NNARX", narx, "Needs NMPC/local linearization for direct MPC use"),
    ]
    payload = {
        "config": asdict(cfg),
        "data_policy": {
            "plant_and_identifier_are_separate": True,
            "initial_control": "rule-based safety supervisor",
            "excitation": "small clock/random planned pulses with safety override",
            "target_leakage": False,
            "split": "time ordered 70/15/15",
            "selection_metric": "validation robust score = mean(block FIT_sim) - 0.5*std(block FIT_sim)",
        },
        "input_cols": list(INPUT_COLS),
        "clip_bounds_scaled": list(clip),
        "data_audit": audit_data(df, cfg),
        "arx": {
            "leaderboard_top20": arx["leaderboard"].head(20).to_dict(orient="records"),
            "selected_by_validation": arx["selected_by_validation"],
            "validation": arx["validation"],
            "test": arx["test"],
        },
        "narx": {
            "leaderboard": narx["leaderboard"].to_dict(orient="records"),
            "selected_by_validation": narx["selected_by_validation"],
            "validation": narx["validation"],
            "test": narx["test"],
        },
        "arx_one_step_residual_diagnostics": arx_one_diag,
        "arx_sim_residual_diagnostics": arx_sim_diag,
        "comparison": comparison,
    }

    df.to_csv(RESULTS_DIR / "protocol_greenhouse_data.csv", index=False)
    arx["leaderboard"].to_csv(RESULTS_DIR / "arx_leaderboard.csv", index=False)
    narx["leaderboard"].to_csv(RESULTS_DIR / "narx_leaderboard.csv", index=False)
    pd.DataFrame(comparison).to_csv(RESULTS_DIR / "comparison.csv", index=False)
    test_predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
        f.write("\n")
    write_report(payload)
    return payload


def main() -> None:
    payload = run_pipeline(ProtocolConfig())
    arx = payload["comparison"][0]
    narx = payload["comparison"][1]
    print("=== Protocol-Grade ARX/NARX ===")
    print(f"ARX  selected: {arx['selected_model']}")
    print(f"ARX  test FIT_1={arx['test_FIT_1step']:.3f} FIT_12={arx['test_FIT_12']:.3f} FIT_sim={arx['test_FIT_sim']:.3f}")
    print(f"NARX selected: {narx['selected_model']}")
    print(f"NARX test FIT_1={narx['test_FIT_1step']:.3f} FIT_12={narx['test_FIT_12']:.3f} FIT_sim={narx['test_FIT_sim']:.3f}")
    print(f"Artifacts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
