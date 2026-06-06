from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arx_redo_pipeline import compute_metrics, json_ready
from narx_end_to_end_75 import (
    EndToEndConfig,
    add_features,
    apply_scale,
    fit_scale_stats,
    generate_identifiable_greenhouse_data,
    inverse_y,
    split_time,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = OUT_DIR / "results_mpc_audit"


@dataclass(frozen=True)
class AuditConfig:
    old_csv_path: str = str(PROJECT_ROOT / "greenhouse_data.csv")
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    n_step: int = 12
    validation_blocks: int = 4
    validation_std_penalty: float = 0.5
    na_grid: tuple[int, ...] = (2, 3, 5, 8, 12)
    nb_grid: tuple[int, ...] = (1, 2, 3, 5)
    nk_grid: tuple[int, ...] = (1, 2, 3)
    alpha_grid: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1)


@dataclass(frozen=True)
class ArxSpec:
    na: int
    nb: int
    nk: int
    alpha: float


def max_lag(spec: ArxSpec) -> int:
    return max(spec.na, spec.nb + spec.nk - 1)


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


def split_by_ratio(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    n_train = int(n_rows * train_ratio)
    n_val = int(n_rows * val_ratio)
    return (
        df.iloc[:n_train].reset_index(drop=True),
        df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        df.iloc[n_train + n_val :].reset_index(drop=True),
    )


def slice_summary(name: str, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "name": name,
        "rows": int(len(df)),
        "start": str(df["Timestamp"].iloc[0]),
        "end": str(df["Timestamp"].iloc[-1]),
        "soil_mean": float(df["Soil_Moisture"].mean()),
        "soil_std": float(df["Soil_Moisture"].std(ddof=0)),
        "drip_on_pct": float(100.0 * df["Drip"].mean()),
        "mist_on_pct": float(100.0 * df["Mist"].mean()),
        "fan_on_pct": float(100.0 * df["Fan"].mean()),
    }


def actuator_diagnostics(df_in: pd.DataFrame) -> dict[str, Any]:
    df = df_in.sort_values("Timestamp").reset_index(drop=True)
    y = df["Soil_Moisture"].to_numpy(dtype=float)
    low = df["Soil_Low_SP"].to_numpy(dtype=float)
    high = df["Soil_High_SP"].to_numpy(dtype=float)
    drip = df["Drip"].to_numpy(dtype=float)
    mist = df["Mist"].to_numpy(dtype=float)
    fan = df["Fan"].to_numpy(dtype=float)

    y_prev = y[:-1]
    low_prev = low[:-1]
    high_prev = high[:-1]
    center_prev = 0.5 * (low_prev + high_prev)
    drip_now = drip[1:]

    low_mask = y_prev < low_prev
    safe_mask = (y_prev >= low_prev + 1.0) & (y_prev <= high_prev - 1.5)
    high_mask = y_prev > high_prev

    def p_on(mask: np.ndarray, signal: np.ndarray = drip_now) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(100.0 * np.mean(signal[mask] > 0.5))

    delta_2 = y[3:] - y[:-3]
    drip_for_delta = drip[1:-2]

    return {
        "drip_on_pct": float(100.0 * np.mean(drip > 0.5)),
        "mist_on_pct": float(100.0 * np.mean(mist > 0.5)),
        "fan_on_pct": float(100.0 * np.mean(fan > 0.5)),
        "p_drip_on_when_prev_soil_below_low_pct": p_on(low_mask),
        "p_drip_on_when_prev_soil_safe_mid_pct": p_on(safe_mask),
        "p_drip_on_when_prev_soil_above_high_pct": p_on(high_mask),
        "count_prev_soil_below_low": int(low_mask.sum()),
        "count_prev_soil_safe_mid": int(safe_mask.sum()),
        "count_prev_soil_above_high": int(high_mask.sum()),
        "corr_drip_t_with_prev_soil_minus_center": safe_corr(drip_now, y_prev - center_prev),
        "corr_drip_t_with_prev_soil_minus_low": safe_corr(drip_now, y_prev - low_prev),
        "corr_drip_t_with_soil_delta_tplus2": safe_corr(drip_for_delta, delta_2),
        "corr_mist_t_with_temp": safe_corr(mist, df["Temperature"].to_numpy(dtype=float)),
        "corr_fan_t_with_temp": safe_corr(fan, df["Temperature"].to_numpy(dtype=float)),
    }


def audit_dataset(name: str, df_in: pd.DataFrame, cfg: AuditConfig) -> dict[str, Any]:
    df = df_in.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    diffs = df["Timestamp"].diff().dt.total_seconds().dropna()
    train, val, test = split_by_ratio(df, cfg.train_ratio, cfg.val_ratio)
    return {
        "name": name,
        "rows": int(len(df)),
        "start": str(df["Timestamp"].iloc[0]),
        "end": str(df["Timestamp"].iloc[-1]),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_timestamps": int(df["Timestamp"].duplicated().sum()),
        "median_sampling_seconds": float(diffs.median()) if len(diffs) else None,
        "irregular_sampling_count": int((diffs != diffs.median()).sum()) if len(diffs) else 0,
        "soil_min": float(df["Soil_Moisture"].min()),
        "soil_max": float(df["Soil_Moisture"].max()),
        "split_60_20_20": {
            "train": slice_summary("train", train),
            "validation": slice_summary("validation", val),
            "test": slice_summary("test", test),
        },
        "actuator_diagnostics": actuator_diagnostics(df),
    }


def build_arx_matrix(df_z: pd.DataFrame, input_cols: tuple[str, ...], spec: ArxSpec) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    lag = max_lag(spec)
    cols: list[np.ndarray] = []
    for y_lag in range(1, spec.na + 1):
        cols.append(y[lag - y_lag : len(y) - y_lag])
    for col in input_cols:
        u = df_z[col].to_numpy(dtype=float)
        for u_lag in range(spec.nk, spec.nk + spec.nb):
            cols.append(u[lag - u_lag : len(u) - u_lag])
    cols.append(np.ones(len(y) - lag))
    return np.vstack(cols).T, y[lag:]


def fit_arx(df_train_z: pd.DataFrame, input_cols: tuple[str, ...], spec: ArxSpec) -> np.ndarray:
    x_train, y_train = build_arx_matrix(df_train_z, input_cols, spec)
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
    input_arrays: list[np.ndarray],
    t: int,
    theta: np.ndarray,
    spec: ArxSpec,
) -> float:
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


def predict_one_step(
    df_z: pd.DataFrame,
    input_cols: tuple[str, ...],
    theta: np.ndarray,
    spec: ArxSpec,
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x, y_true = build_arx_matrix(df_z, input_cols, spec)
    y_pred = np.clip(x @ theta, clip_bounds[0], clip_bounds[1])
    return np.asarray(y_pred, dtype=float), y_true


def simulate_arx(
    df_z: pd.DataFrame,
    input_cols: tuple[str, ...],
    theta: np.ndarray,
    spec: ArxSpec,
    clip_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_sim = y.copy()
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in input_cols]
    lag = max_lag(spec)
    for t in range(lag, len(y)):
        y_sim[t] = float(np.clip(arx_predict_at(y_sim, input_arrays, t, theta, spec), clip_bounds[0], clip_bounds[1]))
    return y_sim[lag:], y[lag:]


def simulate_n_step(
    df_z: pd.DataFrame,
    input_cols: tuple[str, ...],
    theta: np.ndarray,
    spec: ArxSpec,
    clip_bounds: tuple[float, float],
    n_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_work = y.copy()
    y_pred = y.copy()
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in input_cols]
    lag = max_lag(spec)
    for t in range(lag, len(y)):
        origin = max(lag - 1, t - n_step)
        reset_start = max(0, origin - spec.na)
        y_work[reset_start : t + 1] = y[reset_start : t + 1]
        for step_t in range(origin + 1, t + 1):
            y_work[step_t] = float(
                np.clip(arx_predict_at(y_work, input_arrays, step_t, theta, spec), clip_bounds[0], clip_bounds[1])
            )
        y_pred[t] = y_work[t]
    return y_pred[lag:], y[lag:]


def evaluate_real(
    y_true_z: np.ndarray,
    y_pred_z: np.ndarray,
    stats: dict[str, tuple[float, float]],
) -> dict[str, float]:
    return compute_metrics(inverse_y(y_true_z, stats), inverse_y(y_pred_z, stats))


def validation_block_scores(
    df_val_z: pd.DataFrame,
    input_cols: tuple[str, ...],
    theta: np.ndarray,
    spec: ArxSpec,
    stats: dict[str, tuple[float, float]],
    clip_bounds: tuple[float, float],
    cfg: AuditConfig,
) -> dict[str, Any]:
    scores: list[float] = []
    for block in np.array_split(np.arange(len(df_val_z)), cfg.validation_blocks):
        df_block = df_val_z.iloc[int(block[0]) : int(block[-1]) + 1].reset_index(drop=True)
        y_sim, y_true = simulate_arx(df_block, input_cols, theta, spec, clip_bounds)
        scores.append(evaluate_real(y_true, y_sim, stats)["FIT"])
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=0))
    return {
        "block_FIT_sim": scores,
        "block_mean_FIT_sim": mean_score,
        "block_std_FIT_sim": std_score,
        "robust_score": mean_score - cfg.validation_std_penalty * std_score,
    }


def prepare_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    df, input_cols = add_features(df_raw)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df.sort_values("Timestamp").reset_index(drop=True), input_cols


def search_arx_on_dataset(name: str, df_raw: pd.DataFrame, cfg: AuditConfig) -> dict[str, Any]:
    df, input_cols = prepare_features(df_raw)
    e2e_cfg = EndToEndConfig(train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio)
    df_train, df_val, df_test = split_time(df, e2e_cfg)
    stats = fit_scale_stats(df_train, input_cols)
    df_train_z = apply_scale(df_train, stats)
    df_val_z = apply_scale(df_val, stats)
    df_test_z = apply_scale(df_test, stats)
    clip_bounds = tuple(float(v) for v in np.quantile(df_train_z["Soil_Moisture"], cfg.clip_quantiles))

    leaderboard: list[dict[str, Any]] = []
    fitted: dict[str, tuple[ArxSpec, np.ndarray]] = {}
    for na in cfg.na_grid:
        for nb in cfg.nb_grid:
            for nk in cfg.nk_grid:
                for alpha in cfg.alpha_grid:
                    spec = ArxSpec(na=na, nb=nb, nk=nk, alpha=alpha)
                    theta = fit_arx(df_train_z, input_cols, spec)
                    y_val_1, y_true_val_1 = predict_one_step(df_val_z, input_cols, theta, spec, clip_bounds)
                    y_val_sim, y_true_val_sim = simulate_arx(df_val_z, input_cols, theta, spec, clip_bounds)
                    robust = validation_block_scores(df_val_z, input_cols, theta, spec, stats, clip_bounds, cfg)
                    key = f"ARX_na{na}_nb{nb}_nk{nk}_alpha{alpha:g}"
                    fitted[key] = (spec, theta)
                    leaderboard.append(
                        {
                            "dataset": name,
                            "model": key,
                            "na": na,
                            "nb": nb,
                            "nk": nk,
                            "alpha": alpha,
                            "n_params": int(len(theta)),
                            "val_FIT_1step": evaluate_real(y_true_val_1, y_val_1, stats)["FIT"],
                            "val_FIT_sim": evaluate_real(y_true_val_sim, y_val_sim, stats)["FIT"],
                            "val_robust_score": robust["robust_score"],
                            "val_block_mean_FIT_sim": robust["block_mean_FIT_sim"],
                            "val_block_std_FIT_sim": robust["block_std_FIT_sim"],
                            "val_block_FIT_sim": robust["block_FIT_sim"],
                        }
                    )

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("val_robust_score", ascending=False).reset_index(drop=True)
    selected = leaderboard_df.iloc[0].to_dict()
    selected_spec, theta_selected = fitted[str(selected["model"])]

    def eval_split(split_name: str, df_z: pd.DataFrame) -> dict[str, Any]:
        y_1, y_true_1 = predict_one_step(df_z, input_cols, theta_selected, selected_spec, clip_bounds)
        y_12, y_true_12 = simulate_n_step(df_z, input_cols, theta_selected, selected_spec, clip_bounds, cfg.n_step)
        y_sim, y_true_sim = simulate_arx(df_z, input_cols, theta_selected, selected_spec, clip_bounds)
        return {
            "name": split_name,
            "metrics_1step": evaluate_real(y_true_1, y_1, stats),
            "metrics_12": evaluate_real(y_true_12, y_12, stats),
            "metrics_sim": evaluate_real(y_true_sim, y_sim, stats),
        }

    result = {
        "dataset": name,
        "selection_policy": "ARX order/regularization selected by validation robust score only",
        "input_cols": list(input_cols),
        "clip_bounds_scaled": list(clip_bounds),
        "selected_by_validation": selected,
        "selected_metrics": {
            "validation": eval_split("validation", df_val_z),
            "test": eval_split("test", df_test_z),
        },
        "leaderboard_top10": leaderboard_df.head(10).to_dict(orient="records"),
    }

    leaderboard_df.to_csv(RESULTS_DIR / f"arx_leaderboard_{name}.csv", index=False)
    return result


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def existing_model_summary() -> dict[str, Any]:
    redo = load_json(OUT_DIR / "results" / "metrics.json") or {}
    v75 = load_json(OUT_DIR / "results_v75" / "metrics.json") or {}
    clean_narx = load_json(OUT_DIR / "results_narx" / "metrics.json") or {}
    e2e = load_json(OUT_DIR / "results_end_to_end_75" / "metrics.json") or {}
    return {
        "pbl_report_arx_test": {
            "source": "BaocaoPBL_Final.docx paragraphs 778-823",
            "split_reported": "70/15/15",
            "model": "ARX(5,1,2), Least Squares",
            "FIT_1step": 85.85,
            "FIT_12": 67.16,
            "FIT_sim": 66.42,
            "RMSE_sim": 0.978,
        },
        "redo_old_data_arx_backbone": (redo.get("metrics", {}).get("arx", {}).get("test") if redo else None),
        "redo_old_data_hybrid": (redo.get("metrics", {}).get("hybrid_selected", {}).get("test") if redo else None),
        "v75_old_data_causal_best": (v75.get("causal_refit80", {}).get("hybrid_test") if v75 else None),
        "clean_narx_old_data_selected": (clean_narx.get("selected_by_validation") if clean_narx else None),
        "end_to_end_new_data_narx_selected": {
            "selected": e2e.get("selected_by_validation") if e2e else None,
            "metrics": e2e.get("selected_metrics", {}).get("test") if e2e else None,
            "target_met": e2e.get("target_met") if e2e else None,
        },
    }


def model_comparison_table(
    existing: dict[str, Any],
    arx_old: dict[str, Any],
    arx_new: dict[str, Any],
) -> list[dict[str, Any]]:
    clean_narx = existing.get("clean_narx_old_data_selected") or {}
    e2e_narx = existing.get("end_to_end_new_data_narx_selected", {})
    e2e_test = (e2e_narx.get("metrics") or {})

    def arx_row(label: str, dataset: str, result: dict[str, Any]) -> dict[str, Any]:
        test = result["selected_metrics"]["test"]
        selected = result["selected_by_validation"]
        return {
            "case": label,
            "dataset": dataset,
            "model_family": "linear ARX",
            "selection": selected["model"],
            "FIT_1step_test": test["metrics_1step"]["FIT"],
            "FIT_12_test": test["metrics_12"]["FIT"],
            "FIT_sim_test": test["metrics_sim"]["FIT"],
            "RMSE_sim_test": test["metrics_sim"]["RMSE"],
            "production_note": "linear, MPC-friendly",
        }

    rows = [
        {
            "case": "PBL report previous result",
            "dataset": "old simulated feedback data",
            "model_family": "linear ARX",
            "selection": "ARX(5,1,2), LS, report split 70/15/15",
            "FIT_1step_test": 85.85,
            "FIT_12_test": 67.16,
            "FIT_sim_test": 66.42,
            "RMSE_sim_test": 0.978,
            "production_note": "baseline in report",
        },
        arx_row("ARX robust search rerun", "old simulated feedback data", arx_old),
        arx_row("ARX robust search rerun", "new identifiable data", arx_new),
        {
            "case": "Clean NNARX rerun",
            "dataset": "old simulated feedback data",
            "model_family": "nonlinear NNARX",
            "selection": clean_narx.get("model"),
            "FIT_1step_test": clean_narx.get("test_FIT_1step"),
            "FIT_12_test": clean_narx.get("test_FIT_12"),
            "FIT_sim_test": clean_narx.get("test_FIT_sim"),
            "RMSE_sim_test": clean_narx.get("test_RMSE_sim"),
            "production_note": "not better in free-run on old data",
        },
        {
            "case": "End-to-end Delta-NNARX",
            "dataset": "new identifiable data",
            "model_family": "nonlinear NNARX",
            "selection": (e2e_narx.get("selected") or {}).get("model"),
            "FIT_1step_test": (e2e_test.get("metrics_1step") or {}).get("FIT"),
            "FIT_12_test": (e2e_test.get("metrics_12") or {}).get("FIT"),
            "FIT_sim_test": (e2e_test.get("metrics_sim") or {}).get("FIT"),
            "RMSE_sim_test": (e2e_test.get("metrics_sim") or {}).get("RMSE"),
            "production_note": "target met on new identifiable data, but needs NMPC/local linearization for MPC",
        },
    ]
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_summary(payload: dict[str, Any]) -> None:
    comparison = payload["model_comparison"]
    old_diag = payload["data_audit"]["old_feedback_data"]["actuator_diagnostics"]
    new_diag = payload["data_audit"]["new_identifiable_data"]["actuator_diagnostics"]
    arx_old = payload["arx_search"]["old_feedback_data"]
    arx_new = payload["arx_search"]["new_identifiable_data"]

    lines = [
        "# MPC Model Change Audit",
        "",
        "## Ket luan ngan",
        "",
        "- Doi ARX sang NARX co anh huong den MPC neu MPC hien tai dung mo hinh tuyen tinh ARX/RLS nhu trong bao cao.",
        "- NARX khong the thay truc tiep vao Adaptive MPC dang cap nhat he so ARX bang RLS; can NMPC, local linearization, hoac giu ARX cho MPC va dung NARX cho du bao/monitoring.",
        "- Tren data cu, clean NNARX khong vuot ARX o free-run. Tren data moi co excitation dung nguyen tac, ca ARX va Delta-NNARX deu dat tren 75% FIT_sim.",
        "",
        "## Bao cao PBL da doc",
        "",
        "- Bao cao chon `ARX(5,1,2)` cho do am dat.",
        "- Ket qua cu trong bao cao: `FIT_1step ~= 85.85%`, `FIT_12 ~= 67.16%`, `FIT_sim ~= 66.42%`, `RMSE_sim ~= 0.978`.",
        "- Phan MPC trong bao cao mo ta Adaptive MPC cap nhat he so ARX bang `Recursive Least Squares (RLS)`.",
        "- Khong tim thay source code MPC rieng trong repo ngoai bao cao, nen nhan dinh MPC dua tren kien truc da mo ta trong report.",
        "",
        "## Data audit",
        "",
        "| Dataset | Missing | Duplicate time | Sampling s | Drip on % | P_drip_below_low % | P_drip_safe_mid % | corr_drip_prev_margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            "| Old feedback data | "
            f"{payload['data_audit']['old_feedback_data']['missing_total']} | "
            f"{payload['data_audit']['old_feedback_data']['duplicate_timestamps']} | "
            f"{fmt(payload['data_audit']['old_feedback_data']['median_sampling_seconds'], 0)} | "
            f"{fmt(old_diag['drip_on_pct'])} | "
            f"{fmt(old_diag['p_drip_on_when_prev_soil_below_low_pct'])} | "
            f"{fmt(old_diag['p_drip_on_when_prev_soil_safe_mid_pct'])} | "
            f"{fmt(old_diag['corr_drip_t_with_prev_soil_minus_center'])} |"
        ),
        (
            "| New identifiable data | "
            f"{payload['data_audit']['new_identifiable_data']['missing_total']} | "
            f"{payload['data_audit']['new_identifiable_data']['duplicate_timestamps']} | "
            f"{fmt(payload['data_audit']['new_identifiable_data']['median_sampling_seconds'], 0)} | "
            f"{fmt(new_diag['drip_on_pct'])} | "
            f"{fmt(new_diag['p_drip_on_when_prev_soil_below_low_pct'])} | "
            f"{fmt(new_diag['p_drip_on_when_prev_soil_safe_mid_pct'])} | "
            f"{fmt(new_diag['corr_drip_t_with_prev_soil_minus_center'])} |"
        ),
        "",
        "Doc dung so lieu nay nhu sau: old data la closed-loop/feedback data, nen actuator Drip co dau vet tu trang thai do am truoc do. New data sinh actuator theo clock/weather/random excitation, khong tu `Soil_Moisture`.",
        "",
        "## ARX search rerun",
        "",
        "| Dataset | Selected ARX | Val robust | Test FIT_1step | Test FIT_12 | Test FIT_sim | Test RMSE_sim |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        (
            "| Old feedback data | "
            f"`{arx_old['selected_by_validation']['model']}` | "
            f"{fmt(arx_old['selected_by_validation']['val_robust_score'])} | "
            f"{fmt(arx_old['selected_metrics']['test']['metrics_1step']['FIT'])} | "
            f"{fmt(arx_old['selected_metrics']['test']['metrics_12']['FIT'])} | "
            f"{fmt(arx_old['selected_metrics']['test']['metrics_sim']['FIT'])} | "
            f"{fmt(arx_old['selected_metrics']['test']['metrics_sim']['RMSE'], 4)} |"
        ),
        (
            "| New identifiable data | "
            f"`{arx_new['selected_by_validation']['model']}` | "
            f"{fmt(arx_new['selected_by_validation']['val_robust_score'])} | "
            f"{fmt(arx_new['selected_metrics']['test']['metrics_1step']['FIT'])} | "
            f"{fmt(arx_new['selected_metrics']['test']['metrics_12']['FIT'])} | "
            f"{fmt(arx_new['selected_metrics']['test']['metrics_sim']['FIT'])} | "
            f"{fmt(arx_new['selected_metrics']['test']['metrics_sim']['RMSE'], 4)} |"
        ),
        "",
        "## Model comparison",
        "",
        "| Case | Dataset | Family | Test FIT_1step | Test FIT_12 | Test FIT_sim | Note |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in comparison:
        lines.append(
            "| "
            f"{row['case']} | {row['dataset']} | {row['model_family']} | "
            f"{fmt(row['FIT_1step_test'])} | {fmt(row['FIT_12_test'])} | {fmt(row['FIT_sim_test'])} | "
            f"{row['production_note']} |"
        )

    lines.extend(
        [
            "",
            "## MPC impact",
            "",
            "Neu giu MPC tuyen tinh hien tai: nen giu ARX/hybrid-ARX lam plant model cho MPC, va dung NARX nhu du bao phu hoac canh bao drift.",
            "",
            "Neu muon dung NARX trong dieu khien: phai doi sang mot trong ba huong:",
            "",
            "1. NMPC voi model phi tuyen va optimizer phu hop.",
            "2. Local linearization cua NARX tai moi buoc de cap ma tran tuyen tinh cho MPC.",
            "3. Gain-scheduled ARX/NARX-linearized theo mua/giai doan cay.",
            "",
            "Ket luan thuc te: dung NARX de nang fit la hop ly ve mat du bao, nhung khong nen noi la thay ARX trong MPC ma khong sua MPC. Do la thay doi kien truc dieu khien.",
            "",
        ]
    )
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit(cfg: AuditConfig) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    old_df = pd.read_csv(cfg.old_csv_path, parse_dates=["Timestamp"])
    new_df = generate_identifiable_greenhouse_data(EndToEndConfig())

    data_audit = {
        "old_feedback_data": audit_dataset("old_feedback_data", old_df, cfg),
        "new_identifiable_data": audit_dataset("new_identifiable_data", new_df, cfg),
    }
    arx_old = search_arx_on_dataset("old_feedback_data", old_df, cfg)
    arx_new = search_arx_on_dataset("new_identifiable_data", new_df, cfg)
    existing = existing_model_summary()
    comparison = model_comparison_table(existing, arx_old, arx_new)
    pd.DataFrame(comparison).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    payload = {
        "config": asdict(cfg),
        "report_findings": {
            "source": "BaocaoPBL_Final.docx",
            "arx_model_in_report": "ARX(5,1,2)",
            "mpc_dependency": "Adaptive MPC updates ARX coefficients by Recursive Least Squares according to report paragraph 629",
            "mpc_code_found": False,
        },
        "data_audit": data_audit,
        "existing_model_summary": existing,
        "arx_search": {
            "old_feedback_data": arx_old,
            "new_identifiable_data": arx_new,
        },
        "model_comparison": comparison,
        "mpc_impact": {
            "direct_replacement_safe": False,
            "reason": "Existing report architecture assumes a linear ARX model and RLS parameter update; NNARX is nonlinear and not coefficient-compatible with that MPC.",
            "low_risk_path": "Keep ARX/hybrid-ARX as MPC plant model; use NARX for monitoring or advisory prediction.",
            "required_for_full_narx_control": ["NMPC", "local linearization", "gain-scheduled controller", "closed-loop validation"],
        },
    }

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
        f.write("\n")
    write_summary(payload)
    return payload


def main() -> None:
    payload = run_audit(AuditConfig())
    old_arx = payload["arx_search"]["old_feedback_data"]["selected_metrics"]["test"]["metrics_sim"]["FIT"]
    new_arx = payload["arx_search"]["new_identifiable_data"]["selected_metrics"]["test"]["metrics_sim"]["FIT"]
    e2e = payload["existing_model_summary"]["end_to_end_new_data_narx_selected"]["metrics"]
    narx_fit = e2e["metrics_sim"]["FIT"] if e2e else None
    print("=== MPC Model Change Audit ===")
    print(f"ARX old-data test FIT_sim = {old_arx:.3f}")
    print(f"ARX new-data test FIT_sim = {new_arx:.3f}")
    if narx_fit is not None:
        print(f"NARX new-data test FIT_sim = {narx_fit:.3f}")
    print("Direct ARX->NARX replacement in current MPC: not safe without MPC redesign")
    print(f"Artifacts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
