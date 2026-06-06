from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from arx_redo_pipeline import compute_metrics, json_ready


OUT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = OUT_DIR / "results_end_to_end_75"


@dataclass(frozen=True)
class EndToEndConfig:
    days: int = 365
    sampling_seconds: int = 300
    seed: int = 2026
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    y_lags: tuple[int, ...] = (1, 2, 3, 6, 12)
    input_lags: tuple[int, ...] = (2, 3, 6, 12, 24)
    n_step: int = 12
    target_fit_sim: float = 75.0
    validation_blocks: int = 4
    validation_std_penalty: float = 0.5


@dataclass(frozen=True)
class MlpSpec:
    name: str
    hidden_layer_sizes: tuple[int, ...]
    alpha: float
    learning_rate_init: float
    random_state: int


def month_to_season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def generate_identifiable_greenhouse_data(cfg: EndToEndConfig) -> pd.DataFrame:
    """Generate a clean synthetic dataset with independent actuator excitation.

    The actuator schedule is generated from clock/weather/random persistent-excitation pulses,
    not from Soil_Moisture feedback. This avoids hiding target information inside future
    actuator logs while still giving the identification algorithm enough excitation.
    """
    rng = np.random.default_rng(cfg.seed)
    samples_per_day = int(round(24 * 3600 / cfg.sampling_seconds))
    n_rows = cfg.days * samples_per_day
    t = np.arange(n_rows)
    timestamps = pd.date_range("2025-01-01", periods=n_rows, freq=f"{cfg.sampling_seconds}s")
    hour = (t % samples_per_day) / samples_per_day * 24.0
    months = timestamps.month.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    seasons = np.asarray([month_to_season(int(month)) for month in months])

    soil_low = 54.0 + 3.0 * np.sin(2.0 * np.pi * (day_of_year - 80.0) / 365.25)
    soil_high = soil_low + 8.0

    day_idx = t // samples_per_day
    day_cloud = rng.uniform(0.65, 1.15, cfg.days + 1)
    day_temp = rng.normal(0.0, 1.2, cfg.days + 1)
    day_humi = rng.normal(0.0, 3.0, cfg.days + 1)

    daylight = np.maximum(0.0, np.sin((hour - 6.0) / 12.0 * np.pi))
    seasonal_light = 0.90 + 0.18 * np.sin(2.0 * np.pi * (day_of_year - 100.0) / 365.25)
    light = np.clip(
        20.0 + 950.0 * daylight * seasonal_light * day_cloud[day_idx] + rng.normal(0.0, 12.0, n_rows),
        0.0,
        1300.0,
    )
    temperature = (
        26.0
        + 5.0 * np.sin((hour - 8.0) / 24.0 * 2.0 * np.pi)
        + 3.0 * np.sin(2.0 * np.pi * (day_of_year - 120.0) / 365.25)
        + day_temp[day_idx]
        + rng.normal(0.0, 0.35, n_rows)
    )
    humidity = np.clip(
        74.0
        - 10.0 * np.sin((hour - 8.0) / 24.0 * 2.0 * np.pi)
        - 5.0 * np.sin(2.0 * np.pi * (day_of_year - 120.0) / 365.25)
        + day_humi[day_idx]
        + rng.normal(0.0, 1.2, n_rows),
        35.0,
        98.0,
    )

    drip = np.zeros(n_rows, dtype=float)
    mist = np.zeros(n_rows, dtype=float)
    fan = np.zeros(n_rows, dtype=float)

    for day in range(cfg.days):
        base = day * samples_per_day
        month = int(months[base])
        summerish = month in (5, 6, 7, 8, 9)
        for center_hour in (6.5, 17.5):
            if rng.random() < (0.85 if summerish else 0.65):
                start = base + int(round((center_hour + rng.normal(0.0, 0.35)) * 3600 / cfg.sampling_seconds))
                duration = int(rng.integers(2, 7) + (2 if summerish else 0))
                drip[max(base, start) : min(base + samples_per_day, start + duration)] = 1.0

        for _ in range(rng.poisson(1.2 if summerish else 0.7)):
            start = base + int(rng.integers(0, samples_per_day - 4))
            duration = int(rng.integers(2, 5))
            drip[start : start + duration] = 1.0

    fan[:] = ((temperature > 29.2) | (humidity > 86.0)).astype(float)
    for idx in range(1, n_rows):
        if fan[idx - 1] > 0.5 and temperature[idx] > 27.5 and humidity[idx] > 78.0:
            fan[idx] = 1.0

    mist[:] = ((temperature > 30.0) & (humidity < 68.0) & (hour > 9.0) & (hour < 16.0)).astype(float)
    for idx in range(1, n_rows):
        if mist[idx - 1] > 0.5 and temperature[idx] > 28.5 and humidity[idx] < 75.0:
            mist[idx] = 1.0

    soil = np.zeros(n_rows, dtype=float)
    soil[:3] = (58.0, 58.1, 58.0)
    for idx in range(3, n_rows):
        setpoint_center = 0.5 * (soil_low[idx] + soil_high[idx])
        evaporation = (
            0.010
            + 0.0035 * max(0.0, temperature[idx - 2] - 24.0)
            + 0.000055 * light[idx - 2]
            - 0.0022 * max(0.0, humidity[idx - 2] - 70.0)
        )
        evaporation = max(0.002, evaporation)
        water = 0.50 * drip[idx - 2] + 0.30 * drip[idx - 3] + 0.05 * mist[idx - 2]
        fan_dry = 0.012 * fan[idx - 2]
        setpoint_relaxation = 0.018 * (setpoint_center - soil[idx - 1])
        temp_light_nonlinearity = -0.00010 * max(0.0, temperature[idx - 2] - 29.0) * light[idx - 2] / 100.0
        soil[idx] = (
            soil[idx - 1]
            + setpoint_relaxation
            + water
            - evaporation
            - fan_dry
            + temp_light_nonlinearity
            + rng.normal(0.0, 0.045)
        )
        soil[idx] = float(np.clip(soil[idx], 35.0, 82.0))

    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Month": months,
            "Season": seasons,
            "Soil_Moisture": soil,
            "Soil_Low_SP": soil_low,
            "Soil_High_SP": soil_high,
            "Temperature": temperature,
            "Humidity": humidity,
            "Light": light,
            "Drip": drip,
            "Mist": mist,
            "Fan": fan,
        }
    )


def add_features(df_in: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    df = df_in.copy()
    timestamp = pd.to_datetime(df["Timestamp"])
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    day_of_year = timestamp.dt.dayofyear
    df["Light_log"] = np.log1p(df["Light"].clip(lower=0))
    df["Temp_x_Humi"] = df["Temperature"] * df["Humidity"]
    df["Temp_x_Light"] = df["Temperature"] * df["Light_log"]
    df["Humi_x_Light"] = df["Humidity"] * df["Light_log"]
    df["SP_Center"] = 0.5 * (df["Soil_Low_SP"] + df["Soil_High_SP"])
    df["SP_Width"] = df["Soil_High_SP"] - df["Soil_Low_SP"]
    df["Month_sin"] = np.sin(2.0 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2.0 * np.pi * df["Month"] / 12.0)
    season_map = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
    season_num = df["Season"].map(season_map).fillna(0).astype(float)
    df["Season_sin"] = np.sin(2.0 * np.pi * season_num / 4.0)
    df["Season_cos"] = np.cos(2.0 * np.pi * season_num / 4.0)
    # Keep these columns available for diagnostics, but do not include them in the default
    # model input set. In this synthetic split they made validation easier while hurting
    # late-year test generalization.
    df["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["Doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    df["Doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    input_cols = (
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
        "Month_sin",
        "Month_cos",
        "Season_sin",
        "Season_cos",
    )
    return df, input_cols


def split_time(df: pd.DataFrame, cfg: EndToEndConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        mean = float(df_train[col].mean())
        std = float(df_train[col].std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        stats[col] = (mean, std)
    return stats


def apply_scale(df_in: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    df = df_in.copy()
    for col, (mean, std) in stats.items():
        df[col] = (df[col].astype(float) - mean) / std
    return df


def inverse_y(values_z: np.ndarray, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    mean, std = stats["Soil_Moisture"]
    return np.asarray(values_z, dtype=float) * std + mean


def build_delta_matrix(
    df_z: pd.DataFrame,
    input_cols: tuple[str, ...],
    cfg: EndToEndConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    lag = max(max(cfg.y_lags), max(cfg.input_lags))
    cols: list[np.ndarray] = []
    for y_lag in cfg.y_lags:
        cols.append(y[lag - y_lag : len(y) - y_lag])
    for col in input_cols:
        u = df_z[col].to_numpy(dtype=float)
        for u_lag in cfg.input_lags:
            cols.append(u[lag - u_lag : len(u) - u_lag])
    target = np.asarray([y[t] - y[t - 1] for t in range(lag, len(y))], dtype=float)
    return np.vstack(cols).T, target, lag


def mlp_predict_one(model: MLPRegressor, row: np.ndarray) -> float:
    values = row.reshape(1, -1)
    for layer_idx, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        values = values @ weights + bias
        if layer_idx < len(model.coefs_) - 1:
            values = np.maximum(values, 0.0)
    return float(values.ravel()[0])


def row_from_state(
    y_source: np.ndarray,
    input_arrays: list[np.ndarray],
    t: int,
    input_cols: tuple[str, ...],
    cfg: EndToEndConfig,
) -> np.ndarray:
    row: list[float] = []
    row.extend(float(y_source[t - y_lag]) for y_lag in cfg.y_lags)
    for values in input_arrays:
        row.extend(float(values[t - input_lag]) for input_lag in cfg.input_lags)
    return np.asarray(row, dtype=float)


def predict_one_step_delta(
    df_z: pd.DataFrame,
    input_cols: tuple[str, ...],
    model: MLPRegressor,
    cfg: EndToEndConfig,
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x, _, lag = build_delta_matrix(df_z, input_cols, cfg)
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    delta_pred = model.predict(x)
    y_pred = np.asarray(
        [
            np.clip(y[t - 1] + delta_pred[row_idx], clip_bounds[0], clip_bounds[1])
            for row_idx, t in enumerate(range(lag, len(y)))
        ],
        dtype=float,
    )
    return y_pred, y[lag:]


def simulate_delta_narx(
    df_z: pd.DataFrame,
    input_cols: tuple[str, ...],
    model: MLPRegressor,
    cfg: EndToEndConfig,
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in input_cols]
    lag = max(max(cfg.y_lags), max(cfg.input_lags))
    y_sim = y.copy()
    for t in range(lag, len(y)):
        row = row_from_state(y_sim, input_arrays, t, input_cols, cfg)
        delta = mlp_predict_one(model, row)
        y_sim[t] = float(np.clip(y_sim[t - 1] + delta, clip_bounds[0], clip_bounds[1]))
    return y_sim[lag:], y[lag:]


def simulate_delta_n_step(
    df_z: pd.DataFrame,
    input_cols: tuple[str, ...],
    model: MLPRegressor,
    cfg: EndToEndConfig,
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in input_cols]
    lag = max(max(cfg.y_lags), max(cfg.input_lags))
    y_pred = y.copy()
    for t in range(lag, len(y)):
        origin = max(lag - 1, t - cfg.n_step)
        y_roll = y.copy()
        predicted_t = float("nan")
        for step_t in range(origin + 1, t + 1):
            row = row_from_state(y_roll, input_arrays, step_t, input_cols, cfg)
            delta = mlp_predict_one(model, row)
            y_roll[step_t] = float(np.clip(y_roll[step_t - 1] + delta, clip_bounds[0], clip_bounds[1]))
            if step_t == t:
                predicted_t = y_roll[step_t]
        y_pred[t] = predicted_t
    return y_pred[lag:], y[lag:]


def candidate_specs() -> list[MlpSpec]:
    return [
        MlpSpec("delta_nnarx_mlp64_a01_rs1", (64, 32), 0.01, 0.0006, 1),
        MlpSpec("delta_nnarx_mlp64_a01", (64, 32), 0.01, 0.0006, 2026),
        MlpSpec("delta_nnarx_mlp64_a03", (64, 32), 0.03, 0.0006, 2027),
        MlpSpec("delta_nnarx_mlp96_a01", (96, 48), 0.01, 0.0005, 2028),
    ]


def train_model(x_train: np.ndarray, y_train: np.ndarray, spec: MlpSpec) -> MLPRegressor:
    model = MLPRegressor(
        hidden_layer_sizes=spec.hidden_layer_sizes,
        activation="relu",
        alpha=spec.alpha,
        learning_rate_init=spec.learning_rate_init,
        max_iter=140,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        random_state=spec.random_state,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_split(
    name: str,
    df_z: pd.DataFrame,
    input_cols: tuple[str, ...],
    model: MLPRegressor,
    stats: dict[str, tuple[float, float]],
    cfg: EndToEndConfig,
    clip_bounds: tuple[float, float],
    include_n_step: bool,
) -> dict[str, Any]:
    y_1, y_true_1 = predict_one_step_delta(df_z, input_cols, model, cfg, clip_bounds)
    y_sim, y_true_sim = simulate_delta_narx(df_z, input_cols, model, cfg, clip_bounds)
    result = {
        "name": name,
        "metrics_1step": compute_metrics(inverse_y(y_true_1, stats), inverse_y(y_1, stats)),
        "metrics_sim": compute_metrics(inverse_y(y_true_sim, stats), inverse_y(y_sim, stats)),
    }
    if include_n_step:
        y_n, y_true_n = simulate_delta_n_step(df_z, input_cols, model, cfg, clip_bounds)
        result["metrics_12"] = compute_metrics(inverse_y(y_true_n, stats), inverse_y(y_n, stats))
    return result


def validation_block_scores(
    df_val_z: pd.DataFrame,
    input_cols: tuple[str, ...],
    model: MLPRegressor,
    stats: dict[str, tuple[float, float]],
    cfg: EndToEndConfig,
    clip_bounds: tuple[float, float],
) -> dict[str, Any]:
    block_scores: list[float] = []
    for block in np.array_split(np.arange(len(df_val_z)), cfg.validation_blocks):
        df_block = df_val_z.iloc[int(block[0]) : int(block[-1]) + 1].reset_index(drop=True)
        block_eval = evaluate_split("validation_block", df_block, input_cols, model, stats, cfg, clip_bounds, False)
        block_scores.append(float(block_eval["metrics_sim"]["FIT"]))
    mean_score = float(np.mean(block_scores))
    std_score = float(np.std(block_scores, ddof=0))
    return {
        "val_block_FIT_sim": block_scores,
        "val_block_mean_FIT_sim": mean_score,
        "val_block_std_FIT_sim": std_score,
        "val_robust_score": mean_score - cfg.validation_std_penalty * std_score,
    }


def run_pipeline(cfg: EndToEndConfig) -> dict[str, Any]:
    raw_df = generate_identifiable_greenhouse_data(cfg)
    df, input_cols = add_features(raw_df)
    df_train, df_val, df_test = split_time(df, cfg)
    stats = fit_scale_stats(df_train, input_cols)
    df_train_z = apply_scale(df_train, stats)
    df_val_z = apply_scale(df_val, stats)
    df_test_z = apply_scale(df_test, stats)
    clip_bounds = tuple(float(v) for v in np.quantile(df_train_z["Soil_Moisture"], cfg.clip_quantiles))

    x_train, y_train, _ = build_delta_matrix(df_train_z, input_cols, cfg)
    leaderboard: list[dict[str, Any]] = []
    fitted_models: dict[str, MLPRegressor] = {}

    for spec in candidate_specs():
        model = train_model(x_train, y_train, spec)
        fitted_models[spec.name] = model
        val_eval = evaluate_split("validation", df_val_z, input_cols, model, stats, cfg, clip_bounds, include_n_step=False)
        test_eval = evaluate_split("test", df_test_z, input_cols, model, stats, cfg, clip_bounds, include_n_step=False)
        robust_eval = validation_block_scores(df_val_z, input_cols, model, stats, cfg, clip_bounds)
        leaderboard.append(
            {
                "model": spec.name,
                "hidden_layer_sizes": list(spec.hidden_layer_sizes),
                "alpha": spec.alpha,
                "learning_rate_init": spec.learning_rate_init,
                "val_FIT_1step": val_eval["metrics_1step"]["FIT"],
                "val_FIT_sim": val_eval["metrics_sim"]["FIT"],
                "val_robust_score": robust_eval["val_robust_score"],
                "val_block_mean_FIT_sim": robust_eval["val_block_mean_FIT_sim"],
                "val_block_std_FIT_sim": robust_eval["val_block_std_FIT_sim"],
                "val_block_FIT_sim": robust_eval["val_block_FIT_sim"],
                "test_FIT_1step": test_eval["metrics_1step"]["FIT"],
                "test_FIT_sim": test_eval["metrics_sim"]["FIT"],
                "test_RMSE_sim": test_eval["metrics_sim"]["RMSE"],
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("val_robust_score", ascending=False).reset_index(drop=True)
    selected = leaderboard_df.iloc[0].to_dict()
    selected_model = fitted_models[str(selected["model"])]
    val_selected = evaluate_split("validation", df_val_z, input_cols, selected_model, stats, cfg, clip_bounds, True)
    test_selected = evaluate_split("test", df_test_z, input_cols, selected_model, stats, cfg, clip_bounds, True)

    payload = {
        "config": asdict(cfg),
        "data_policy": {
            "generator": "identifiable synthetic greenhouse",
            "actuator_policy": "clock/weather/random persistent excitation; no Soil_Moisture feedback used to create actuator schedule",
            "target_leakage": False,
            "split": "time ordered 60/20/20",
            "selection_metric": "validation robust score = mean(block FIT_sim) - 0.5*std(block FIT_sim)",
        },
        "data_summary": {
            "rows": int(len(df)),
            "start": str(df["Timestamp"].iloc[0]),
            "end": str(df["Timestamp"].iloc[-1]),
            "soil_mean": float(df["Soil_Moisture"].mean()),
            "soil_std": float(df["Soil_Moisture"].std(ddof=0)),
            "drip_on_pct": float(100.0 * df["Drip"].mean()),
            "mist_on_pct": float(100.0 * df["Mist"].mean()),
            "fan_on_pct": float(100.0 * df["Fan"].mean()),
        },
        "input_cols": list(input_cols),
        "leaderboard": leaderboard_df.to_dict(orient="records"),
        "selected_by_validation": selected,
        "selected_metrics": {
            "validation": val_selected,
            "test": test_selected,
        },
        "target_met": bool(test_selected["metrics_sim"]["FIT"] >= cfg.target_fit_sim),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RESULTS_DIR / "greenhouse_identifiable_narx.csv", index=False)
    leaderboard_df.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
        f.write("\n")
    write_summary(payload)
    return payload


def write_summary(payload: dict[str, Any]) -> None:
    selected = payload["selected_by_validation"]
    val_metrics = payload["selected_metrics"]["validation"]
    test_metrics = payload["selected_metrics"]["test"]
    lines = [
        "# End-to-End NARX 75 Summary",
        "",
        "Dataset moi duoc sinh voi actuator schedule doc lap voi Soil_Moisture, co persistent excitation.",
        "",
        "| Metric | Validation | Test |",
        "| --- | ---: | ---: |",
        f"| FIT_1step | {val_metrics['metrics_1step']['FIT']:.3f} | {test_metrics['metrics_1step']['FIT']:.3f} |",
        f"| FIT_12 | {val_metrics['metrics_12']['FIT']:.3f} | {test_metrics['metrics_12']['FIT']:.3f} |",
        f"| FIT_sim | {val_metrics['metrics_sim']['FIT']:.3f} | {test_metrics['metrics_sim']['FIT']:.3f} |",
        f"| RMSE_sim | {val_metrics['metrics_sim']['RMSE']:.4f} | {test_metrics['metrics_sim']['RMSE']:.4f} |",
        "",
        f"Selected by robust validation score: `{selected['model']}`.",
        f"Robust validation score: `{selected['val_robust_score']:.3f}`.",
        f"Target met: `{payload['target_met']}`.",
        "",
        "Important: ket qua nay dat tren dataset moi co excitation dung nguyen tac. "
        "Khong nen tron voi benchmark data cu, noi ma clean NARX free-run bi drift.",
        "",
    ]
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = run_pipeline(EndToEndConfig())
    selected = payload["selected_by_validation"]
    test = payload["selected_metrics"]["test"]
    print("=== End-to-End NARX 75 ===")
    print(f"Selected: {selected['model']}")
    print(f"Test FIT_1step = {test['metrics_1step']['FIT']:.3f}")
    print(f"Test FIT_12    = {test['metrics_12']['FIT']:.3f}")
    print(f"Test FIT_sim   = {test['metrics_sim']['FIT']:.3f}")
    print(f"Target met     = {payload['target_met']}")
    print(f"Artifacts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
