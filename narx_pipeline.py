"""
narx_pipeline.py
================
Polynomial NARX (Nonlinear ARX) pipeline.

Extends the linear ARX baseline by adding polynomial features
(squared terms, cross-terms) to the regression matrix while
keeping OLS estimation.  This allows the model to capture
non-linear relationships such as:
  - y(t-1)^2  → non-linear autoregressive dynamics
  - Drip(t-1) * y(t-1) → interaction between watering and soil state

All heavy-lifting functions (data loading, splitting, OLS, metrics,
residual diagnostics) are reused from arx_pipeline.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arx_pipeline import (
    DataConfig,
    ModelConfig,
    SplitConfig,
    build_regression_matrix,
    build_true_theta,
    compute_ar_roots,
    compute_metrics,
    estimate_ols,
    evaluate_slice,
    extract_true_params_from_module,
    load_or_generate_data,
    parameter_reference_map,
    residual_diagnostics,
    split_time_series,
    summarize_dataset_behavior,
    summarize_parameters,
    simulate_arx,
    simulate_arx_n_step,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NARXModelConfig:
    """Configuration for a Polynomial NARX model."""
    na: int = 2
    nb: int = 2
    nk: int = 1
    poly_degree: int = 2
    include_intercept: bool = True          # NARX benefits from intercept
    input_cols: tuple[str, ...] = tuple(["Temperature", "Humidity", "Light", "Drip", "Mist", "Fan"])
    output_col: str = "Soil_Moisture"
    simulation_clip: tuple[float, float] | None = (0.0, 100.0)

    # Which cross-terms to include (None = auto: all degree-2 combinations)
    cross_term_mode: str = "selective"  # "all", "selective", "none"

    @property
    def linear_config(self) -> ModelConfig:
        """Return a matching linear ARX config for comparison."""
        return ModelConfig(
            na=self.na, nb=self.nb, nk=self.nk,
            include_intercept=False,
            input_cols=self.input_cols,
            output_col=self.output_col,
            simulation_clip=self.simulation_clip,
        )

    @property
    def max_lag(self) -> int:
        return max(self.na, self.nb + self.nk - 1)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _build_linear_features(y: np.ndarray, inputs: list[np.ndarray],
                           t: int, config: NARXModelConfig) -> list[float]:
    """Build the linear part of the feature vector at time t."""
    row: list[float] = []
    for lag in range(1, config.na + 1):
        row.append(float(y[t - lag]))
    for u in inputs:
        for lag in range(config.nk, config.nk + config.nb):
            row.append(float(u[t - lag]))
    return row


def _get_linear_feature_names(config: NARXModelConfig) -> list[str]:
    """Generate names for the linear regressors."""
    names: list[str] = []
    for lag in range(1, config.na + 1):
        names.append(f"y(t-{lag})")
    for col in config.input_cols:
        for lag in range(config.nk, config.nk + config.nb):
            names.append(f"{col}(t-{lag})")
    return names


def _build_poly_features(linear_row: list[float],
                          linear_names: list[str],
                          config: NARXModelConfig) -> tuple[list[float], list[str]]:
    """
    Build polynomial features from a linear feature vector.

    For degree=2:
      - Squared terms: x_i^2
      - Cross terms (selective): y-lags × u-lags, u_on/off × continuous
    """
    poly_feats: list[float] = []
    poly_names: list[str] = []

    if config.poly_degree < 2:
        return poly_feats, poly_names

    n_linear = len(linear_row)

    if config.cross_term_mode == "all":
        # All C(n,2) + n combinations
        for i, j in combinations_with_replacement(range(n_linear), 2):
            poly_feats.append(linear_row[i] * linear_row[j])
            if i == j:
                poly_names.append(f"{linear_names[i]}^2")
            else:
                poly_names.append(f"{linear_names[i]}*{linear_names[j]}")

    elif config.cross_term_mode == "selective":
        # Squared terms for y-lags and continuous inputs only
        y_indices = list(range(config.na))
        continuous_input_names = {"Temperature", "Humidity", "Light"}
        binary_input_names = {"Drip", "Mist", "Fan"}

        cont_indices = []
        bin_indices = []
        for idx, name in enumerate(linear_names):
            # check if this name starts with a continuous input
            for cin in continuous_input_names:
                if name.startswith(cin):
                    cont_indices.append(idx)
                    break
            for bin_name in binary_input_names:
                if name.startswith(bin_name):
                    bin_indices.append(idx)
                    break

        # 1) Squared terms for y-lags
        for i in y_indices:
            poly_feats.append(linear_row[i] ** 2)
            poly_names.append(f"{linear_names[i]}^2")

        # 2) Squared terms for continuous inputs
        for i in cont_indices:
            poly_feats.append(linear_row[i] ** 2)
            poly_names.append(f"{linear_names[i]}^2")

        # 3) Cross-terms: y(t-lag) × each binary input
        for yi in y_indices:
            for bi in bin_indices:
                poly_feats.append(linear_row[yi] * linear_row[bi])
                poly_names.append(f"{linear_names[yi]}*{linear_names[bi]}")

        # 4) Cross-terms: y(t-lag) × each continuous input (first lag only to avoid explosion)
        for yi in y_indices:
            for ci in cont_indices[:len(config.input_cols)]:  # limit
                poly_feats.append(linear_row[yi] * linear_row[ci])
                poly_names.append(f"{linear_names[yi]}*{linear_names[ci]}")

    elif config.cross_term_mode == "none":
        # Only squared terms
        for i in range(n_linear):
            poly_feats.append(linear_row[i] ** 2)
            poly_names.append(f"{linear_names[i]}^2")

    return poly_feats, poly_names


def get_narx_feature_names(config: NARXModelConfig) -> list[str]:
    """Get all feature names (linear + polynomial + intercept)."""
    linear_names = _get_linear_feature_names(config)
    # Use dummy values to get poly names (only names matter)
    dummy = [1.0] * len(linear_names)
    _, poly_names = _build_poly_features(dummy, linear_names, config)
    all_names = linear_names + poly_names
    if config.include_intercept:
        all_names.append("intercept")
    return all_names


def build_narx_regression_matrix(
    df: pd.DataFrame,
    config: NARXModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full NARX regression matrix with polynomial features."""
    y = df[config.output_col].astype(float).to_numpy()
    inputs = [df[col].astype(float).to_numpy() for col in config.input_cols]

    max_lag = config.max_lag
    n_eff = len(y) - max_lag
    if n_eff <= 0:
        raise ValueError("Not enough rows for the configured lags")

    linear_names = _get_linear_feature_names(config)
    all_names = get_narx_feature_names(config)
    n_features = len(all_names)

    x_mat = np.zeros((n_eff, n_features), dtype=float)
    y_vec = np.zeros(n_eff, dtype=float)

    for row_idx in range(n_eff):
        t = row_idx + max_lag
        linear_row = _build_linear_features(y, inputs, t, config)
        poly_row, _ = _build_poly_features(linear_row, linear_names, config)
        full_row = linear_row + poly_row
        if config.include_intercept:
            full_row.append(1.0)
        x_mat[row_idx] = full_row
        y_vec[row_idx] = float(y[t])

    return x_mat, y_vec


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_narx(
    df_sim: pd.DataFrame,
    theta: np.ndarray,
    config: NARXModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-run (open-loop) NARX simulation."""
    y = df_sim[config.output_col].astype(float).to_numpy().copy()
    inputs = [df_sim[col].astype(float).to_numpy() for col in config.input_cols]
    max_lag = config.max_lag
    y_sim = y.copy()
    linear_names = _get_linear_feature_names(config)

    for t in range(max_lag, len(y)):
        linear_row = _build_linear_features(y_sim, inputs, t, config)
        poly_row, _ = _build_poly_features(linear_row, linear_names, config)
        full_row = linear_row + poly_row
        if config.include_intercept:
            full_row.append(1.0)
        y_next = float(np.dot(full_row, theta))
        if config.simulation_clip is not None:
            y_next = float(np.clip(y_next, config.simulation_clip[0], config.simulation_clip[1]))
        y_sim[t] = y_next

    return y_sim[max_lag:], y[max_lag:]


def simulate_narx_n_step(
    df_sim: pd.DataFrame,
    theta: np.ndarray,
    n_steps: int,
    config: NARXModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """N-step (periodic reset) NARX prediction."""
    y = df_sim[config.output_col].astype(float).to_numpy().copy()
    inputs = [df_sim[col].astype(float).to_numpy() for col in config.input_cols]
    max_lag = config.max_lag
    y_sim = y.copy()
    linear_names = _get_linear_feature_names(config)

    for t in range(max_lag, len(y)):
        origin = max(max_lag - 1, t - n_steps)
        y_hist = y[origin - max_lag + 1: origin + 1].copy()
        predicted_t = float("nan")

        for step_t in range(origin + 1, t + 1):
            # Build y-vector from history buffer
            linear_row: list[float] = []
            for lag in range(1, config.na + 1):
                linear_row.append(float(y_hist[-lag]))
            for u in inputs:
                for lag in range(config.nk, config.nk + config.nb):
                    linear_row.append(float(u[step_t - lag]))

            poly_row, _ = _build_poly_features(linear_row, linear_names, config)
            full_row = linear_row + poly_row
            if config.include_intercept:
                full_row.append(1.0)

            y_next = float(np.dot(full_row, theta))
            if config.simulation_clip is not None:
                y_next = float(np.clip(y_next, config.simulation_clip[0], config.simulation_clip[1]))
            y_hist = np.append(y_hist[1:], y_next)
            if step_t == t:
                predicted_t = y_next

        y_sim[t] = predicted_t

    return y_sim[max_lag:], y[max_lag:]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_narx_slice(
    name: str,
    df_slice: pd.DataFrame,
    theta: np.ndarray,
    config: NARXModelConfig,
    n_step: int = 12,
    arx_theta: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full evaluation of a NARX model on a data slice."""
    x_mat, y_vec = build_narx_regression_matrix(df_slice, config)
    y_pred = x_mat @ theta
    y_sim, y_true_sim = simulate_narx(df_slice, theta, config)
    y_n_step, _ = simulate_narx_n_step(df_slice, theta, n_steps=n_step, config=config)

    linear_config = config.linear_config

    result: dict[str, Any] = {
        "name": name,
        "metrics_1step": compute_metrics(y_vec, y_pred, len(theta)),
        "metrics_n_step": compute_metrics(y_true_sim, y_n_step, len(theta)),
        "metrics_sim": compute_metrics(y_true_sim, y_sim, len(theta)),
        "arrays": {
            "y_true_1step": y_vec,
            "y_pred_1step": y_pred,
            "y_true_sim": y_true_sim,
            "y_pred_sim": y_sim,
            "y_pred_n_step": y_n_step,
        },
        "residual_diagnostics": residual_diagnostics(y_vec - y_pred, df_slice, linear_config),
        "behavior": summarize_dataset_behavior(df_slice, name),
    }

    # ARX baseline comparison
    if arx_theta is not None:
        arx_sim, arx_true = simulate_arx(df_slice, arx_theta, linear_config)
        result["arx_baseline_sim"] = compute_metrics(arx_true, arx_sim, len(arx_theta))
        arx_x, arx_y = build_regression_matrix(df_slice, linear_config)
        arx_pred = arx_x @ arx_theta
        result["arx_baseline_1step"] = compute_metrics(arx_y, arx_pred, len(arx_theta))

    return result


# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------

def narx_model_selection(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    base_config: NARXModelConfig,
    na_list: list[int] | None = None,
    nb_list: list[int] | None = None,
    nk_list: list[int] | None = None,
    degree_list: list[int] | None = None,
    cross_modes: list[str] | None = None,
) -> pd.DataFrame:
    """Search over NARX configurations."""
    na_list = na_list or [1, 2, 3]
    nb_list = nb_list or [1, 2]
    nk_list = nk_list or [1]
    degree_list = degree_list or [1, 2]
    cross_modes = cross_modes or ["selective"]

    rows: list[dict[str, Any]] = []

    for na in na_list:
        for nb in nb_list:
            for nk in nk_list:
                for degree in degree_list:
                    for cross_mode in cross_modes:
                        candidate = NARXModelConfig(
                            na=na, nb=nb, nk=nk,
                            poly_degree=degree,
                            include_intercept=base_config.include_intercept,
                            input_cols=base_config.input_cols,
                            output_col=base_config.output_col,
                            simulation_clip=base_config.simulation_clip,
                            cross_term_mode=cross_mode,
                        )
                        try:
                            x_tr, y_tr = build_narx_regression_matrix(df_train, candidate)
                            theta, _, _ = estimate_ols(x_tr, y_tr)
                            ev = evaluate_narx_slice("Validation", df_val, theta, candidate, n_step=12)
                        except Exception as exc:
                            rows.append({
                                "na": na, "nb": nb, "nk": nk,
                                "degree": degree, "cross_mode": cross_mode,
                                "error": str(exc),
                            })
                            continue

                        m1 = ev["metrics_1step"]
                        ms = ev["metrics_sim"]
                        rows.append({
                            "na": na, "nb": nb, "nk": nk,
                            "degree": degree,
                            "cross_mode": cross_mode,
                            "n_params": len(theta),
                            "RMSE_1step": m1["RMSE"],
                            "FIT_1step": m1["FIT"],
                            "R2_1step": m1["R2"],
                            "AIC_1step": m1["AIC"],
                            "BIC_1step": m1["BIC"],
                            "RMSE_sim": ms["RMSE"],
                            "FIT_sim": ms["FIT"],
                            "R2_sim": ms["R2"],
                        })

    df_results = pd.DataFrame(rows)
    if "error" in df_results.columns:
        df_results = df_results[df_results["error"].isna()]
    return df_results.sort_values(
        ["RMSE_sim", "RMSE_1step", "AIC_1step", "n_params"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Run full pipeline
# ---------------------------------------------------------------------------

def run_narx_pipeline(
    data_config: DataConfig | None = None,
    split_config: SplitConfig | None = None,
    narx_config: NARXModelConfig | None = None,
) -> dict[str, Any]:
    """Run the full NARX pipeline end-to-end."""
    data_cfg = data_config or DataConfig()
    split_cfg = split_config or SplitConfig()
    narx_cfg = narx_config or NARXModelConfig()
    arx_cfg = narx_cfg.linear_config

    # --- Data ---
    df, true_params, data_source = load_or_generate_data(data_cfg)
    df_train, df_val, df_test = split_time_series(df, split_cfg)

    # --- NARX estimation ---
    x_train, y_train = build_narx_regression_matrix(df_train, narx_cfg)
    theta_hat, cov_hat, sigma2_hat = estimate_ols(x_train, y_train)

    # --- ARX baseline for comparison ---
    arx_x_train, arx_y_train = build_regression_matrix(df_train, arx_cfg)
    arx_theta, arx_cov, arx_sigma2 = estimate_ols(arx_x_train, arx_y_train)
    arx_true_theta = build_true_theta(true_params, arx_cfg)

    feature_names = get_narx_feature_names(narx_cfg)

    dataset_overview = {
        "rows": int(len(df)),
        "timestamp_start": str(df["Timestamp"].iloc[0]),
        "timestamp_end": str(df["Timestamp"].iloc[-1]),
        "months_present": sorted(int(v) for v in pd.Series(df["Month"]).dropna().unique()),
        "seasons_present": sorted(str(v) for v in pd.Series(df["Season"]).dropna().unique()),
        "narx_n_features": len(feature_names),
        "arx_n_features": len(arx_cfg.param_names),
        "condition_number_narx": float(np.linalg.cond(x_train.T @ x_train)),
        "condition_number_arx": float(np.linalg.cond(arx_x_train.T @ arx_x_train)),
        "rank_narx": int(np.linalg.matrix_rank(x_train)),
    }

    results: dict[str, Any] = {
        "data_source": data_source,
        "data_config": data_cfg,
        "split_config": split_cfg,
        "narx_config": narx_cfg,
        "arx_config": arx_cfg,
        "df_full": df,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "true_params": true_params,
        "dataset_overview": dataset_overview,
        "feature_names": feature_names,
        "theta_hat": theta_hat,
        "cov_hat": cov_hat,
        "sigma2": float(sigma2_hat),
        "arx_theta": arx_theta,
        "arx_sigma2": float(arx_sigma2),
        # Evaluations
        "val": evaluate_narx_slice("Validation", df_val, theta_hat, narx_cfg,
                                    n_step=12, arx_theta=arx_theta),
        "test": evaluate_narx_slice("Test", df_test, theta_hat, narx_cfg,
                                     n_step=12, arx_theta=arx_theta),
        # ARX parameter analysis (for reference)
        "arx_parameter_summary": summarize_parameters(arx_theta, arx_cov, arx_cfg, true_params),
        "arx_ar_roots": compute_ar_roots(arx_theta, arx_cfg),
    }

    # Model search
    results["model_selection"] = narx_model_selection(
        df_train=df_train,
        df_val=df_val,
        base_config=narx_cfg,
    )

    # ARX baseline eval for clean comparison
    if arx_true_theta is not None:
        arx_true_sim_val, arx_true_y_val = simulate_arx(df_val, arx_true_theta, arx_cfg)
        results["arx_deterministic_ceiling_val"] = compute_metrics(arx_true_y_val, arx_true_sim_val, len(arx_true_theta))
        arx_true_sim_test, arx_true_y_test = simulate_arx(df_test, arx_true_theta, arx_cfg)
        results["arx_deterministic_ceiling_test"] = compute_metrics(arx_true_y_test, arx_true_sim_test, len(arx_true_theta))

    return results


def print_narx_summary(results: dict[str, Any]) -> None:
    """Print CLI summary of NARX results."""
    ov = results["dataset_overview"]
    print("=== NARX Pipeline Results ===")
    print(f"Data source         : {results['data_source']}")
    print(f"Total rows          : {ov['rows']}")
    print(f"NARX features       : {ov['narx_n_features']}")
    print(f"ARX features        : {ov['arx_n_features']}")
    print(f"Cond(NARX X'X)      : {ov['condition_number_narx']:.2f}")
    print(f"Cond(ARX X'X)       : {ov['condition_number_arx']:.2f}")
    print()

    print(f"{'Model':<16} | {'Slice':<10} | {'FIT_1step':>10} | {'FIT_12step':>11} | {'FIT_sim':>9}")
    print("-" * 68)
    for key in ["val", "test"]:
        ev = results[key]
        print(
            f"{'NARX':<16} | {ev['name']:<10} | "
            f"{ev['metrics_1step']['FIT']:>10.3f} | "
            f"{ev['metrics_n_step']['FIT']:>11.3f} | "
            f"{ev['metrics_sim']['FIT']:>9.3f}"
        )
        if "arx_baseline_sim" in ev:
            print(
                f"{'ARX (baseline)':<16} | {ev['name']:<10} | "
                f"{ev['arx_baseline_1step']['FIT']:>10.3f} | "
                f"{'n/a':>11} | "
                f"{ev['arx_baseline_sim']['FIT']:>9.3f}"
            )
    print()


if __name__ == "__main__":
    results = run_narx_pipeline()
    print_narx_summary(results)
