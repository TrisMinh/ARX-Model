from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import arx_backbone_pipeline as base


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class ResidualConfig:
    days: int = 16
    sampling_seconds: int = 20
    seed: int = 305031
    residual_y_lags: tuple[int, ...] = (1, 2, 3, 6, 12)
    residual_input_lags: tuple[int, ...] = (2, 3, 6, 12, 24)
    shrink_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    validation_blocks: int = 4
    validation_std_penalty: float = 0.5
    random_state: int = 909


def json_ready(value: Any) -> Any:
    return base.json_ready(value)


def residual_features(
    df_z: pd.DataFrame,
    y_arx_sim_z: np.ndarray,
    spec: base.ArxSpec,
    input_cols: tuple[str, ...],
    cfg: ResidualConfig,
) -> np.ndarray:
    lag = base.arx_max_lag(spec)
    y_source = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_source[lag:] = y_arx_sim_z
    input_arrays = [df_z[col].to_numpy(dtype=float) for col in input_cols]

    n_rows = len(df_z) - lag
    n_cols = 1 + len(cfg.residual_y_lags) + len(input_cols) * len(cfg.residual_input_lags)
    x = np.empty((n_rows, n_cols), dtype=float)

    for row_idx, t in enumerate(range(lag, len(df_z))):
        row: list[float] = [float(y_arx_sim_z[row_idx])]
        row.extend(float(y_source[max(0, t - y_lag)]) for y_lag in cfg.residual_y_lags)
        for values in input_arrays:
            row.extend(float(values[max(0, t - u_lag)]) for u_lag in cfg.residual_input_lags)
        x[row_idx] = row
    return x


def candidate_models(cfg: ResidualConfig) -> list[tuple[str, Any]]:
    return [
        ("ridge_alpha0.1", Ridge(alpha=0.1)),
        ("ridge_alpha1", Ridge(alpha=1.0)),
        ("ridge_alpha10", Ridge(alpha=10.0)),
        (
            "hgb_leaf15_l2_0.01",
            HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.03,
                max_leaf_nodes=15,
                l2_regularization=0.01,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=cfg.random_state,
            ),
        ),
        (
            "hgb_leaf31_l2_0.05",
            HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.03,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=cfg.random_state + 1,
            ),
        ),
    ]


def inverse_metrics(y_true_z: np.ndarray, y_pred_z: np.ndarray, stats: dict[str, tuple[float, float]]) -> dict[str, float]:
    return base.fit_metrics(base.inverse_y(y_true_z, stats), base.inverse_y(y_pred_z, stats))


def robust_block_score(
    y_true_z: np.ndarray,
    y_pred_z: np.ndarray,
    stats: dict[str, tuple[float, float]],
    cfg: ResidualConfig,
) -> dict[str, Any]:
    scores: list[float] = []
    indices = np.arange(len(y_true_z))
    for block in np.array_split(indices, cfg.validation_blocks):
        yt = y_true_z[block]
        yp = y_pred_z[block]
        scores.append(inverse_metrics(yt, yp, stats)["FIT"])
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=0))
    return {
        "val_block_FIT_sim": scores,
        "val_block_mean_FIT_sim": mean_score,
        "val_block_std_FIT_sim": std_score,
        "val_robust_score": mean_score - cfg.validation_std_penalty * std_score,
    }


def run_experiment(cfg: ResidualConfig, data_csv: str | None = None) -> dict[str, Any]:
    base_cfg = base.OutdoorConfig(days=cfg.days, sampling_seconds=cfg.sampling_seconds, seed=cfg.seed)
    raw_df = base.load_raw_data(base_cfg, data_csv)
    df = base.add_features(raw_df)
    train, val, test = base.split_time(df, base_cfg)

    stats = base.fit_scale_stats(train, base.INSIDE_INPUT_COLS)
    train_z = base.apply_scale(train, stats)
    val_z = base.apply_scale(val, stats)
    test_z = base.apply_scale(test, stats)
    clip = tuple(float(v) for v in np.quantile(train_z["Soil_Moisture"], (0.005, 0.995)))

    arx = base.run_arx_search("inside_only", base.INSIDE_INPUT_COLS, train_z, val_z, test_z, stats, clip, base_cfg)
    spec = arx["spec"]
    theta = arx["theta"]

    y_arx_train_z, y_true_train_z = base.simulate_arx(train_z, theta, spec, base.INSIDE_INPUT_COLS, clip)
    y_arx_val_z, y_true_val_z = base.simulate_arx(val_z, theta, spec, base.INSIDE_INPUT_COLS, clip)
    y_arx_test_z, y_true_test_z = base.simulate_arx(test_z, theta, spec, base.INSIDE_INPUT_COLS, clip)

    x_train = residual_features(train_z, y_arx_train_z, spec, base.INSIDE_INPUT_COLS, cfg)
    x_val = residual_features(val_z, y_arx_val_z, spec, base.INSIDE_INPUT_COLS, cfg)
    x_test = residual_features(test_z, y_arx_test_z, spec, base.INSIDE_INPUT_COLS, cfg)
    y_res_train = y_true_train_z - y_arx_train_z

    rows: list[dict[str, Any]] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    arx_val_metrics = inverse_metrics(y_true_val_z, y_arx_val_z, stats)
    arx_test_metrics = inverse_metrics(y_true_test_z, y_arx_test_z, stats)
    arx_robust = robust_block_score(y_true_val_z, y_arx_val_z, stats, cfg)
    rows.append(
        {
            "candidate": "ARX_backbone_no_residual",
            "residual_model": "none",
            "shrink": 0.0,
            "val_FIT_sim": arx_val_metrics["FIT"],
            "test_FIT_sim": arx_test_metrics["FIT"],
            "test_RMSE_sim": arx_test_metrics["RMSE"],
            "test_Bias_sim": arx_test_metrics["Bias"],
            "n_residual_features": int(x_train.shape[1]),
            **arx_robust,
        }
    )
    predictions["ARX_backbone_no_residual"] = (y_arx_val_z, y_arx_test_z, np.zeros_like(y_arx_test_z))

    for model_name, model in candidate_models(cfg):
        model.fit(x_train, y_res_train)
        corr_val = np.asarray(model.predict(x_val), dtype=float)
        corr_test = np.asarray(model.predict(x_test), dtype=float)
        for shrink in cfg.shrink_grid:
            y_hybrid_val_z = np.clip(y_arx_val_z + shrink * corr_val, clip[0], clip[1])
            y_hybrid_test_z = np.clip(y_arx_test_z + shrink * corr_test, clip[0], clip[1])
            val_metrics = inverse_metrics(y_true_val_z, y_hybrid_val_z, stats)
            test_metrics = inverse_metrics(y_true_test_z, y_hybrid_test_z, stats)
            robust = robust_block_score(y_true_val_z, y_hybrid_val_z, stats, cfg)
            candidate_id = f"{model_name}|{shrink:g}"
            rows.append(
                {
                    "candidate": candidate_id,
                    "residual_model": model_name,
                    "shrink": float(shrink),
                    "val_FIT_sim": val_metrics["FIT"],
                    "test_FIT_sim": test_metrics["FIT"],
                    "test_RMSE_sim": test_metrics["RMSE"],
                    "test_Bias_sim": test_metrics["Bias"],
                    "n_residual_features": int(x_train.shape[1]),
                    **robust,
                }
            )
            predictions[candidate_id] = (y_hybrid_val_z, y_hybrid_test_z, shrink * corr_test)

    leaderboard = pd.DataFrame(rows).sort_values("val_robust_score", ascending=False).reset_index(drop=True)
    selected = leaderboard.iloc[0].to_dict()
    selected_id = str(selected["candidate"])
    _, y_selected_test_z, selected_corr_test_z = predictions[selected_id]
    selected_test_metrics = inverse_metrics(y_true_test_z, y_selected_test_z, stats)

    lag = base.arx_max_lag(spec)
    results = {
        "config": asdict(cfg),
        "data_policy": {
            "source_data": str(Path(data_csv).resolve()) if data_csv is not None else "simulated_physical_protocol",
            "split": "chia theo thời gian 70/15/15",
            "target_leakage": False,
            "residual_features": "chỉ dùng ARX simulated trajectory và input lag quá khứ",
            "selection_metric": "validation robust score = mean(block FIT_sim) - 0.5*std(block FIT_sim)",
            "test_policy": "test chỉ báo cáo sau khi chọn bằng validation",
        },
        "arx_backbone": {
            "selected_by_validation": arx["selected_by_validation"],
            "test": arx["test"],
        },
        "residual_policy": {
            "model_family": "Hybrid ARX residual correction",
            "equation": "y_hybrid = y_arx_sim + shrink * residual_model(features)",
            "residual_target_train": "y_true_train - y_arx_sim_train",
            "residual_y_lags": list(cfg.residual_y_lags),
            "residual_input_lags": list(cfg.residual_input_lags),
            "uses_true_future_soil_moisture": False,
            "n_residual_features": int(x_train.shape[1]),
        },
        "selected_by_validation": selected,
        "metrics": {
            "arx_backbone_test_sim": arx_test_metrics,
            "hybrid_selected_test_sim": selected_test_metrics,
            "test_gain_FIT_sim": selected_test_metrics["FIT"] - arx_test_metrics["FIT"],
            "test_gain_RMSE_sim": arx_test_metrics["RMSE"] - selected_test_metrics["RMSE"],
        },
        "leaderboard_top20": leaderboard.head(20).to_dict(orient="records"),
    }

    test_predictions = pd.DataFrame(
        {
            "Timestamp": test["Timestamp"].iloc[lag:].to_numpy(),
            "y_true": base.inverse_y(y_true_test_z, stats),
            "y_arx_sim": base.inverse_y(y_arx_test_z, stats),
            "y_hybrid_selected_sim": base.inverse_y(y_selected_test_z, stats),
            "residual_correction_scaled": selected_corr_test_z,
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(results), f, indent=2)
        f.write("\n")
    leaderboard.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    test_predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    write_summary(results)
    return results


def write_summary(results: dict[str, Any]) -> None:
    arx = results["metrics"]["arx_backbone_test_sim"]
    hybrid = results["metrics"]["hybrid_selected_test_sim"]
    selected = results["selected_by_validation"]
    gain = results["metrics"]["test_gain_FIT_sim"]
    lines = [
        "# Hybrid ARX Residual Final",
        "",
        "Bản này dùng ARX 16 input làm backbone và thêm residual correction. Đây là bản bảo vệ chính nếu chọn hướng Hybrid ARX residual.",
        "",
        "## Nguyên tắc không leakage",
        "",
        "- Residual chỉ train trên train split.",
        "- Chọn residual model và shrink bằng validation robust score.",
        "- Test chỉ dùng để báo cáo cuối.",
        "- Feature residual không dùng `Soil_Moisture` thật tương lai; chỉ dùng quỹ đạo `y_arx_sim` và input lag quá khứ.",
        "- `shrink=0` nằm trong grid, nên nếu residual không giúp thì pipeline có quyền chọn không sửa gì.",
        "",
        "## Kết quả",
        "",
        "| Model | Test FIT_sim | Test RMSE | Test Bias |",
        "| --- | ---: | ---: | ---: |",
        f"| ARX backbone | {arx['FIT']:.3f} | {arx['RMSE']:.4f} | {arx['Bias']:.4f} |",
        f"| Hybrid selected | {hybrid['FIT']:.3f} | {hybrid['RMSE']:.4f} | {hybrid['Bias']:.4f} |",
        "",
        f"- Candidate selected by validation: `{selected['candidate']}`.",
        f"- Test gain FIT_sim: `{gain:.3f}` điểm.",
        "",
    ]
    if gain > 0.25:
        lines.append("Kết luận: residual có cải thiện thật trên test. Bản bảo vệ chính nên gọi đúng là Hybrid ARX residual correction, không gọi là ARX thuần.")
    elif gain < -0.25:
        lines.append("Kết luận: residual làm kém hơn trên test; không nên đưa vào bản final.")
    else:
        lines.append("Kết luận: residual gần như không cải thiện; giữ ARX thuần hợp lý hơn.")
    lines.append("")
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy bản bảo vệ Hybrid ARX residual trên data/protocol mới.")
    parser.add_argument("--data-csv", default=None, help="CSV thật. Nếu bỏ trống, dùng data mô phỏng protocol mới.")
    parser.add_argument("--days", type=int, default=16)
    parser.add_argument("--sampling-seconds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=305031)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ResidualConfig(days=args.days, sampling_seconds=args.sampling_seconds, seed=args.seed)
    results = run_experiment(cfg, args.data_csv)
    arx = results["metrics"]["arx_backbone_test_sim"]
    hybrid = results["metrics"]["hybrid_selected_test_sim"]
    selected = results["selected_by_validation"]
    print("=== Hybrid ARX residual final ===")
    print(f"ARX backbone:    FIT_sim={arx['FIT']:.3f} RMSE={arx['RMSE']:.4f}")
    print(f"Hybrid selected: FIT_sim={hybrid['FIT']:.3f} RMSE={hybrid['RMSE']:.4f}")
    print(f"Selected: {selected['candidate']} | val_robust={selected['val_robust_score']:.3f}")
    print(f"Đã lưu kết quả vào {RESULTS_DIR}")


if __name__ == "__main__":
    main()
