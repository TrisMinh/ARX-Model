from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from arx_redo_pipeline import (
    AUGMENTED_INPUT_COLS,
    PROJECT_ROOT,
    add_augmented_features,
    compute_metrics,
    json_ready,
    split_time,
)


OUT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = OUT_DIR / "results_narx"


@dataclass(frozen=True)
class NarxCleanConfig:
    csv_path: str = str(PROJECT_ROOT / "greenhouse_data.csv")
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    n_step: int = 12
    random_state: int = 1206


@dataclass(frozen=True)
class NarxSpec:
    name: str
    y_lags: tuple[int, ...]
    base_input_lags: tuple[int, ...]
    memory_input_lags: tuple[int, ...]
    estimator: str
    target_mode: str = "direct"

    @property
    def start_lag(self) -> int:
        return max(max(self.y_lags), max(self.base_input_lags), max(self.memory_input_lags, default=0))


def add_causal_memory_features(df_in: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    df = df_in.copy()
    memory_cols: list[str] = []

    for col in ("Drip", "Mist", "Fan"):
        series = df[col].astype(float)
        for window in (6, 12, 24, 36, 72, 144, 288):
            sum_col = f"{col}_sum_lag2_win_{window}"
            mean_col = f"{col}_mean_lag2_win_{window}"
            shifted = series.shift(2)
            df[sum_col] = shifted.rolling(window, min_periods=1).sum().fillna(0.0)
            df[mean_col] = shifted.rolling(window, min_periods=1).mean().fillna(0.0)
            memory_cols.extend([sum_col, mean_col])
        tso_col = f"{col}_time_since_on_lag2"
        df[tso_col] = pd.Series(time_since_on(series.to_numpy(dtype=float))).shift(2).fillna(288 * 30)
        memory_cols.append(tso_col)

    for col in ("Temperature", "Humidity", "Light_log"):
        series = df[col].astype(float).shift(2)
        for window in (12, 24, 72, 144, 288):
            mean_col = f"{col}_mean_lag2_win_{window}"
            df[mean_col] = series.rolling(window, min_periods=1).mean().fillna(0.0)
            memory_cols.append(mean_col)

    return df, tuple(memory_cols)


def time_since_on(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    last_on = -10**9
    for idx, value in enumerate(values):
        if value > 0.5:
            last_on = idx
        out[idx] = min(idx - last_on, 288 * 30)
    return out


def fit_scale_stats(df_train: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in ("Soil_Moisture", *cols):
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


def inverse_y(values_z: np.ndarray, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    mean, std = stats["Soil_Moisture"]
    return np.asarray(values_z, dtype=float) * std + mean


def build_narx_matrix(
    df_z: pd.DataFrame,
    spec: NarxSpec,
    base_cols: tuple[str, ...],
    memory_cols: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    base_inputs = [df_z[col].to_numpy(dtype=float) for col in base_cols]
    memory_inputs = [df_z[col].to_numpy(dtype=float) for col in memory_cols]
    lag = spec.start_lag
    n_rows = len(y) - lag
    n_features = len(spec.y_lags) + len(base_cols) * len(spec.base_input_lags) + len(memory_cols) * len(
        spec.memory_input_lags
    )
    x = np.empty((n_rows, n_features), dtype=float)
    target = y[lag:].copy()

    for row_idx, t in enumerate(range(lag, len(y))):
        row: list[float] = []
        row.extend(float(y[t - y_lag]) for y_lag in spec.y_lags)
        for values in base_inputs:
            row.extend(float(values[t - u_lag]) for u_lag in spec.base_input_lags)
        for values in memory_inputs:
            row.extend(float(values[t - u_lag]) for u_lag in spec.memory_input_lags)
        x[row_idx] = row

    return x, target


def build_delta_matrix(
    df_z: pd.DataFrame,
    spec: NarxSpec,
    base_cols: tuple[str, ...],
    memory_cols: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    x, _ = build_narx_matrix(df_z, spec, base_cols, memory_cols)
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    target = np.asarray([y[t] - y[t - 1] for t in range(spec.start_lag, len(y))], dtype=float)
    return x, target


def build_training_matrix(
    df_z: pd.DataFrame,
    spec: NarxSpec,
    base_cols: tuple[str, ...],
    memory_cols: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if spec.target_mode == "delta":
        return build_delta_matrix(df_z, spec, base_cols, memory_cols)
    return build_narx_matrix(df_z, spec, base_cols, memory_cols)


def row_from_state(
    y_source: np.ndarray,
    base_inputs: list[np.ndarray],
    memory_inputs: list[np.ndarray],
    t: int,
    spec: NarxSpec,
) -> np.ndarray:
    row: list[float] = []
    row.extend(float(y_source[t - y_lag]) for y_lag in spec.y_lags)
    for values in base_inputs:
        row.extend(float(values[t - u_lag]) for u_lag in spec.base_input_lags)
    for values in memory_inputs:
        row.extend(float(values[t - u_lag]) for u_lag in spec.memory_input_lags)
    return np.asarray(row, dtype=float).reshape(1, -1)


def predict_one(model: Any, row: np.ndarray) -> float:
    if isinstance(model, MLPRegressor):
        values = row.reshape(1, -1)
        for idx, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
            values = values @ weights + bias
            if idx < len(model.coefs_) - 1:
                if model.activation == "relu":
                    values = np.maximum(values, 0.0)
                elif model.activation == "tanh":
                    values = np.tanh(values)
                elif model.activation == "logistic":
                    values = 1.0 / (1.0 + np.exp(-values))
        return float(values.ravel()[0])
    return float(model.predict(row.reshape(1, -1))[0])


def simulate_narx(
    df_z: pd.DataFrame,
    model: Any,
    spec: NarxSpec,
    base_cols: tuple[str, ...],
    memory_cols: tuple[str, ...],
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    base_inputs = [df_z[col].to_numpy(dtype=float) for col in base_cols]
    memory_inputs = [df_z[col].to_numpy(dtype=float) for col in memory_cols]
    y_sim = y.copy()
    for t in range(spec.start_lag, len(y)):
        row = row_from_state(y_sim, base_inputs, memory_inputs, t, spec)
        y_next = predict_one(model, row)
        y_sim[t] = float(np.clip(y_next, clip_bounds[0], clip_bounds[1]))
    return y_sim[spec.start_lag :], y[spec.start_lag :]


def simulate_narx_n_step(
    df_z: pd.DataFrame,
    model: Any,
    spec: NarxSpec,
    base_cols: tuple[str, ...],
    memory_cols: tuple[str, ...],
    clip_bounds: tuple[float, float],
    n_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    base_inputs = [df_z[col].to_numpy(dtype=float) for col in base_cols]
    memory_inputs = [df_z[col].to_numpy(dtype=float) for col in memory_cols]
    y_pred = y.copy()

    for t in range(spec.start_lag, len(y)):
        origin = max(spec.start_lag - 1, t - n_step)
        y_roll = y.copy()
        predicted_t = float("nan")
        for step_t in range(origin + 1, t + 1):
            row = row_from_state(y_roll, base_inputs, memory_inputs, step_t, spec)
            y_next = predict_one(model, row)
            y_roll[step_t] = float(np.clip(y_next, clip_bounds[0], clip_bounds[1]))
            if step_t == t:
                predicted_t = y_roll[step_t]
        y_pred[t] = predicted_t

    return y_pred[spec.start_lag :], y[spec.start_lag :]


def predict_delta_one_step(
    df_z: pd.DataFrame,
    model: Any,
    spec: NarxSpec,
    base_cols: tuple[str, ...],
    memory_cols: tuple[str, ...],
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x, _ = build_delta_matrix(df_z, spec, base_cols, memory_cols)
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    delta_pred = model.predict(x)
    y_pred = np.asarray(
        [np.clip(y[t - 1] + delta_pred[row_idx], clip_bounds[0], clip_bounds[1]) for row_idx, t in enumerate(range(spec.start_lag, len(y)))],
        dtype=float,
    )
    return y_pred, y[spec.start_lag :]


def simulate_narx_delta(
    df_z: pd.DataFrame,
    model: Any,
    spec: NarxSpec,
    base_cols: tuple[str, ...],
    memory_cols: tuple[str, ...],
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    base_inputs = [df_z[col].to_numpy(dtype=float) for col in base_cols]
    memory_inputs = [df_z[col].to_numpy(dtype=float) for col in memory_cols]
    y_sim = y.copy()
    for t in range(spec.start_lag, len(y)):
        row = row_from_state(y_sim, base_inputs, memory_inputs, t, spec)
        delta = predict_one(model, row)
        y_sim[t] = float(np.clip(y_sim[t - 1] + delta, clip_bounds[0], clip_bounds[1]))
    return y_sim[spec.start_lag :], y[spec.start_lag :]


def simulate_narx_delta_n_step(
    df_z: pd.DataFrame,
    model: Any,
    spec: NarxSpec,
    base_cols: tuple[str, ...],
    memory_cols: tuple[str, ...],
    clip_bounds: tuple[float, float],
    n_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    base_inputs = [df_z[col].to_numpy(dtype=float) for col in base_cols]
    memory_inputs = [df_z[col].to_numpy(dtype=float) for col in memory_cols]
    y_pred = y.copy()
    for t in range(spec.start_lag, len(y)):
        origin = max(spec.start_lag - 1, t - n_step)
        y_roll = y.copy()
        predicted_t = float("nan")
        for step_t in range(origin + 1, t + 1):
            row = row_from_state(y_roll, base_inputs, memory_inputs, step_t, spec)
            delta = predict_one(model, row)
            y_roll[step_t] = float(np.clip(y_roll[step_t - 1] + delta, clip_bounds[0], clip_bounds[1]))
            if step_t == t:
                predicted_t = y_roll[step_t]
        y_pred[t] = predicted_t
    return y_pred[spec.start_lag :], y[spec.start_lag :]


def make_estimator(spec: NarxSpec, seed: int):
    if spec.estimator == "mlp64":
        alpha = 3e-4
        lr = 8e-4
        hidden = (64, 32)
    elif spec.estimator == "mlp_delta_a01":
        alpha = 1e-2
        lr = 5e-4
        hidden = (64, 32)
    elif spec.estimator == "mlp_delta_a10":
        alpha = 1e-1
        lr = 5e-4
        hidden = (64, 32)
    elif spec.estimator == "mlp128":
        alpha = 3e-4
        lr = 6e-4
        hidden = (128, 64)
    else:
        raise ValueError(f"Unknown estimator: {spec.estimator}")

    if spec.estimator in {"mlp64", "mlp_delta_a01", "mlp_delta_a10", "mlp128"}:
        return MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=90,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=seed,
        )


def evaluate_split(
    name: str,
    df_z: pd.DataFrame,
    model: Any,
    spec: NarxSpec,
    base_cols: tuple[str, ...],
    memory_cols: tuple[str, ...],
    stats: dict[str, tuple[float, float]],
    clip_bounds: tuple[float, float],
) -> dict[str, Any]:
    if spec.target_mode == "delta":
        y_pred_1, y_true_1 = predict_delta_one_step(df_z, model, spec, base_cols, memory_cols, clip_bounds)
        y_pred_sim, y_true_sim = simulate_narx_delta(df_z, model, spec, base_cols, memory_cols, clip_bounds)
    else:
        x, y_true_1 = build_narx_matrix(df_z, spec, base_cols, memory_cols)
        y_pred_1 = np.clip(model.predict(x), clip_bounds[0], clip_bounds[1])
        y_pred_sim, y_true_sim = simulate_narx(df_z, model, spec, base_cols, memory_cols, clip_bounds)
    return {
        "name": name,
        "FIT_1step": compute_metrics(inverse_y(y_true_1, stats), inverse_y(y_pred_1, stats))["FIT"],
        "RMSE_1step": compute_metrics(inverse_y(y_true_1, stats), inverse_y(y_pred_1, stats))["RMSE"],
        "FIT_sim": compute_metrics(inverse_y(y_true_sim, stats), inverse_y(y_pred_sim, stats))["FIT"],
        "RMSE_sim": compute_metrics(inverse_y(y_true_sim, stats), inverse_y(y_pred_sim, stats))["RMSE"],
    }


def evaluate_n_step_only(
    df_z: pd.DataFrame,
    model: Any,
    spec: NarxSpec,
    base_cols: tuple[str, ...],
    memory_cols: tuple[str, ...],
    stats: dict[str, tuple[float, float]],
    clip_bounds: tuple[float, float],
    n_step: int,
) -> dict[str, float]:
    if spec.target_mode == "delta":
        y_pred_n, y_true_n = simulate_narx_delta_n_step(df_z, model, spec, base_cols, memory_cols, clip_bounds, n_step)
    else:
        y_pred_n, y_true_n = simulate_narx_n_step(df_z, model, spec, base_cols, memory_cols, clip_bounds, n_step)
    metrics = compute_metrics(inverse_y(y_true_n, stats), inverse_y(y_pred_n, stats))
    return {"FIT_12": metrics["FIT"], "RMSE_12": metrics["RMSE"]}


def candidate_specs() -> list[NarxSpec]:
    return [
        NarxSpec(
            name="NNARX_MLP64_compact",
            y_lags=(1, 2, 3, 6, 12),
            base_input_lags=(2, 3, 6, 12),
            memory_input_lags=(0,),
            estimator="mlp64",
        ),
        NarxSpec(
            name="Delta_NNARX_MLP64_small_a01",
            y_lags=(1, 2, 3),
            base_input_lags=(2, 3, 6),
            memory_input_lags=(0,),
            estimator="mlp_delta_a01",
            target_mode="delta",
        ),
        NarxSpec(
            name="Delta_NNARX_MLP64_small_a10",
            y_lags=(1, 2, 3),
            base_input_lags=(2, 3, 6),
            memory_input_lags=(0,),
            estimator="mlp_delta_a10",
            target_mode="delta",
        ),
        NarxSpec(
            name="Delta_NNARX_MLP64_nomem_a01",
            y_lags=(1, 2, 3, 6, 12),
            base_input_lags=(2, 3, 6, 12),
            memory_input_lags=(),
            estimator="mlp_delta_a01",
            target_mode="delta",
        ),
    ]


def run_pipeline(cfg: NarxCleanConfig) -> dict[str, Any]:
    df = pd.read_csv(cfg.csv_path, parse_dates=["Timestamp"])
    df = add_augmented_features(df.sort_values("Timestamp").reset_index(drop=True))
    df, memory_cols = add_causal_memory_features(df)
    base_cols = tuple(AUGMENTED_INPUT_COLS)
    df_train, df_val, df_test = split_time(df, cfg.train_ratio, cfg.val_ratio)

    scale_cols = (*base_cols, *memory_cols)
    stats = fit_scale_stats(df_train, scale_cols)
    df_train_z = apply_scale(df_train, stats)
    df_val_z = apply_scale(df_val, stats)
    df_test_z = apply_scale(df_test, stats)
    clip_bounds = tuple(float(v) for v in np.quantile(df_train_z["Soil_Moisture"], cfg.clip_quantiles))

    leaderboard: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    fitted: dict[str, tuple[Any, NarxSpec]] = {}
    for idx, spec in enumerate(candidate_specs()):
        x_train, y_train = build_training_matrix(df_train_z, spec, base_cols, memory_cols)
        model = make_estimator(spec, cfg.random_state + idx)
        model.fit(x_train, y_train)

        val_eval = evaluate_split(
            "validation",
            df_val_z,
            model,
            spec,
            base_cols,
            memory_cols,
            stats,
            clip_bounds,
        )
        test_eval = evaluate_split(
            "test",
            df_test_z,
            model,
            spec,
            base_cols,
            memory_cols,
            stats,
            clip_bounds,
        )
        row = {
            "model": spec.name,
            "estimator": spec.estimator,
            "target_mode": spec.target_mode,
            "n_features": int(x_train.shape[1]),
            "train_rows": int(x_train.shape[0]),
            "val_FIT_1step": val_eval["FIT_1step"],
            "val_FIT_12": None,
            "val_FIT_sim": val_eval["FIT_sim"],
            "test_FIT_1step": test_eval["FIT_1step"],
            "test_FIT_12": None,
            "test_FIT_sim": test_eval["FIT_sim"],
            "test_RMSE_sim": test_eval["RMSE_sim"],
        }
        leaderboard.append(row)
        fitted[spec.name] = (model, spec)
        details[spec.name] = {
            "spec": asdict(spec),
            "validation": val_eval,
            "test": test_eval,
        }

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("val_FIT_sim", ascending=False).reset_index(drop=True)
    selected = leaderboard_df.iloc[0].to_dict()
    selected_model, selected_spec = fitted[str(selected["model"])]
    selected_val_12 = evaluate_n_step_only(
        df_val_z,
        selected_model,
        selected_spec,
        base_cols,
        memory_cols,
        stats,
        clip_bounds,
        cfg.n_step,
    )
    selected_test_12 = evaluate_n_step_only(
        df_test_z,
        selected_model,
        selected_spec,
        base_cols,
        memory_cols,
        stats,
        clip_bounds,
        cfg.n_step,
    )
    selected["val_FIT_12"] = selected_val_12["FIT_12"]
    selected["test_FIT_12"] = selected_test_12["FIT_12"]
    selected["val_RMSE_12"] = selected_val_12["RMSE_12"]
    selected["test_RMSE_12"] = selected_test_12["RMSE_12"]
    leaderboard_df.loc[leaderboard_df["model"] == selected["model"], "val_FIT_12"] = selected["val_FIT_12"]
    leaderboard_df.loc[leaderboard_df["model"] == selected["model"], "test_FIT_12"] = selected["test_FIT_12"]
    details[str(selected["model"])]["validation"].update(selected_val_12)
    details[str(selected["model"])]["test"].update(selected_test_12)
    payload = {
        "config": asdict(cfg),
        "data_policy": {
            "split": "time ordered 60/20/20",
            "scaling": "fit on train only",
            "target_leakage": False,
            "input_policy": "uses lagged inputs only; memory features are shifted by 2 samples before rolling",
            "selection_metric": "validation FIT_sim",
        },
        "base_input_cols": list(base_cols),
        "memory_feature_count": len(memory_cols),
        "clip_bounds_scaled": list(clip_bounds),
        "leaderboard": leaderboard_df.to_dict(orient="records"),
        "selected_by_validation": selected,
        "details": details,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
        f.write("\n")
    leaderboard_df.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    write_summary(payload)
    return payload


def write_summary(payload: dict[str, Any]) -> None:
    leaderboard = pd.DataFrame(payload["leaderboard"])
    lines = [
        "# Clean NARX Summary",
        "",
        "All candidates use time-ordered split, train-only scaling, lagged inputs, and no future Soil_Moisture.",
        "",
        "| Model | Val FIT_1 | Val FIT_12 | Val FIT_sim | Test FIT_1 | Test FIT_12 | Test FIT_sim |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in leaderboard.iterrows():
        lines.append(
            f"| {row['model']} | {row['val_FIT_1step']:.3f} | {row['val_FIT_12']:.3f} | "
            f"{row['val_FIT_sim']:.3f} | {row['test_FIT_1step']:.3f} | {row['test_FIT_12']:.3f} | "
            f"{row['test_FIT_sim']:.3f} |"
        )
    selected = payload["selected_by_validation"]
    lines.extend(
        [
            "",
            f"Selected by validation FIT_sim: `{selected['model']}`.",
            "",
            "Interpretation: one-step may be high because true output history is supplied at every step. "
            "Free-run `FIT_sim` is the strict deployment-style test.",
            "",
        ]
    )
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = run_pipeline(NarxCleanConfig())
    selected = payload["selected_by_validation"]
    print("=== Clean NARX ===")
    print(
        f"Selected {selected['model']} | Val sim={selected['val_FIT_sim']:.3f} | "
        f"Test sim={selected['test_FIT_sim']:.3f}"
    )
    print(
        f"Test 1-step={selected['test_FIT_1step']:.3f} | Test 12-step={selected['test_FIT_12']:.3f}"
    )
    print(f"Artifacts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
