from __future__ import annotations

import numpy as np
import pandas as pd

from arx5s_clean.algorithm.specs import ArxSpec


def max_lag(spec: ArxSpec) -> int:
    return max(spec.na, spec.nb + spec.nk - 1)


def build_arx_matrix(df_z: pd.DataFrame, spec: ArxSpec, input_cols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    lag = max_lag(spec)
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


def predict_at(y_source: np.ndarray, inputs: list[np.ndarray], t: int, theta: np.ndarray, spec: ArxSpec) -> float:
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


def predict_one_step(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x, y_true = build_arx_matrix(df_z, spec, input_cols)
    return np.clip(x @ theta, clip[0], clip[1]), y_true


def simulate_free_run(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_sim = y.copy()
    inputs = [df_z[col].to_numpy(dtype=float) for col in input_cols]
    lag = max_lag(spec)

    for t in range(lag, len(y)):
        y_sim[t] = float(np.clip(predict_at(y_sim, inputs, t, theta, spec), clip[0], clip[1]))

    return y_sim[lag:], y[lag:]


def simulate_chunked(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    clip: tuple[float, float],
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    y_pred = y.copy()
    inputs = [df_z[col].to_numpy(dtype=float) for col in input_cols]
    lag = max_lag(spec)

    for start in range(lag, len(y), horizon_steps):
        end = min(len(y), start + horizon_steps)
        y_work = y.copy()
        for t in range(start, end):
            y_work[t] = float(np.clip(predict_at(y_work, inputs, t, theta, spec), clip[0], clip[1]))
            y_pred[t] = y_work[t]

    return y_pred[lag:], y[lag:]

