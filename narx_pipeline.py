from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


@dataclass(frozen=True)
class NarxConfig:
    na: int
    nb: int
    nk: int
    input_cols: tuple[str, ...]
    output_col: str = "Soil_Moisture"
    simulation_clip: tuple[float, float] | None = None

    @property
    def max_lag(self) -> int:
        return max(self.na, self.nb + self.nk - 1)

    @property
    def n_features(self) -> int:
        return self.na + len(self.input_cols) * self.nb


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_params: int = 0) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    n_obs = len(resid)
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y_true - y_true.mean(), y_true - y_true.mean()))
    sigma2 = ss_res / max(1, n_obs)
    rmse = float(np.sqrt(sigma2))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    fit = float(100.0 * (1.0 - np.linalg.norm(resid) / np.linalg.norm(y_true - y_true.mean())))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    aic = float(n_obs * np.log(sigma2 + 1e-12) + 2 * n_params)
    bic = float(n_obs * np.log(sigma2 + 1e-12) + n_params * np.log(max(1, n_obs)))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "FIT": fit,
        "R2": r2,
        "AIC": aic,
        "BIC": bic,
    }


def build_augmented_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
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
    return df


def fit_zscore_stats(df_train: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in cols:
        mean = float(df_train[col].astype(float).mean())
        std = float(df_train[col].astype(float).std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_zscore(df_in: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    df_out = df_in.copy()
    for col, st in stats.items():
        df_out[col] = (df_out[col].astype(float) - st["mean"]) / st["std"]
    return df_out


def inverse_zscore_y(values: np.ndarray, stats: dict[str, dict[str, float]], output_col: str = "Soil_Moisture") -> np.ndarray:
    st = stats[output_col]
    return np.asarray(values, dtype=float) * st["std"] + st["mean"]


def scaled_clip_bounds(
    df_train: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    quantiles: tuple[float, float] = (0.01, 0.99),
    output_col: str = "Soil_Moisture",
) -> tuple[tuple[float, float], tuple[float, float]]:
    lo_real = float(df_train[output_col].quantile(quantiles[0]))
    hi_real = float(df_train[output_col].quantile(quantiles[1]))
    st = stats[output_col]
    lo_scaled = float((lo_real - st["mean"]) / st["std"])
    hi_scaled = float((hi_real - st["mean"]) / st["std"])
    return (lo_real, hi_real), (lo_scaled, hi_scaled)


def build_narx_matrix(df: pd.DataFrame, config: NarxConfig) -> tuple[np.ndarray, np.ndarray]:
    y = df[config.output_col].astype(float).to_numpy()
    inputs = [df[col].astype(float).to_numpy() for col in config.input_cols]
    lag = config.max_lag
    n_eff = len(y) - lag
    if n_eff <= 0:
        raise ValueError("Not enough rows for configured NARX lags")
    x = np.zeros((n_eff, config.n_features), dtype=float)
    target = np.zeros(n_eff, dtype=float)
    for row_idx in range(n_eff):
        t = row_idx + lag
        row: list[float] = []
        for y_lag in range(1, config.na + 1):
            row.append(float(y[t - y_lag]))
        for u in inputs:
            for u_lag in range(config.nk, config.nk + config.nb):
                row.append(float(u[t - u_lag]))
        x[row_idx] = row
        target[row_idx] = float(y[t])
    return x, target


def _clip(value: np.ndarray | float, bounds: tuple[float, float] | None):
    if bounds is None:
        return value
    return np.clip(value, bounds[0], bounds[1])


def _row_from_state(y_source: np.ndarray, inputs: list[np.ndarray], t: int, config: NarxConfig) -> np.ndarray:
    row: list[float] = []
    for y_lag in range(1, config.na + 1):
        row.append(float(y_source[t - y_lag]))
    for u in inputs:
        for u_lag in range(config.nk, config.nk + config.nb):
            row.append(float(u[t - u_lag]))
    return np.asarray(row, dtype=float).reshape(1, -1)


def simulate_narx(df: pd.DataFrame, model: RegressorMixin, config: NarxConfig) -> tuple[np.ndarray, np.ndarray]:
    y = df[config.output_col].astype(float).to_numpy().copy()
    inputs = [df[col].astype(float).to_numpy() for col in config.input_cols]
    lag = config.max_lag
    y_sim = y.copy()
    for t in range(lag, len(y)):
        row = _row_from_state(y_sim, inputs, t, config)
        y_next = float(model.predict(row)[0])
        y_next = float(_clip(y_next, config.simulation_clip))
        y_sim[t] = y_next
    return y_sim[lag:], y[lag:]


def simulate_narx_n_step(
    df: pd.DataFrame,
    model: RegressorMixin,
    config: NarxConfig,
    n_steps: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    y = df[config.output_col].astype(float).to_numpy().copy()
    inputs = [df[col].astype(float).to_numpy() for col in config.input_cols]
    lag = config.max_lag
    y_sim = y.copy()
    for t in range(lag, len(y)):
        origin = max(lag - 1, t - n_steps)
        y_roll = y.copy()
        predicted_t = float("nan")
        for step_t in range(origin + 1, t + 1):
            row = _row_from_state(y_roll, inputs, step_t, config)
            y_next = float(model.predict(row)[0])
            y_next = float(_clip(y_next, config.simulation_clip))
            y_roll[step_t] = y_next
            if step_t == t:
                predicted_t = y_next
        y_sim[t] = predicted_t
    return y_sim[lag:], y[lag:]


def evaluate_narx_split(
    name: str,
    df_z: pd.DataFrame,
    model: RegressorMixin,
    config: NarxConfig,
    stats: dict[str, dict[str, float]],
    n_step: int = 12,
) -> dict[str, Any]:
    x_mat, y_vec = build_narx_matrix(df_z, config)
    y_pred_1 = model.predict(x_mat)
    y_pred_sim, y_true_sim = simulate_narx(df_z, model, config)
    y_pred_n, _ = simulate_narx_n_step(df_z, model, config, n_steps=n_step)
    return {
        "name": name,
        "metrics_1step": compute_metrics(inverse_zscore_y(y_vec, stats, config.output_col), inverse_zscore_y(y_pred_1, stats, config.output_col), x_mat.shape[1]),
        "metrics_n_step": compute_metrics(inverse_zscore_y(y_true_sim, stats, config.output_col), inverse_zscore_y(y_pred_n, stats, config.output_col), x_mat.shape[1]),
        "metrics_sim": compute_metrics(inverse_zscore_y(y_true_sim, stats, config.output_col), inverse_zscore_y(y_pred_sim, stats, config.output_col), x_mat.shape[1]),
        "arrays": {
            "y_true_sim": y_true_sim,
            "y_pred_sim": y_pred_sim,
            "y_pred_n_step": y_pred_n,
        },
    }
