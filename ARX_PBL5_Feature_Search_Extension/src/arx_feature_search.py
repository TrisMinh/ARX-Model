from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import arx_backbone_pipeline as base


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class FeatureSearchConfig:
    days: int = 16
    sampling_seconds: int = 20
    seed: int = 305031
    min_robust_gain: float = 0.05


def json_ready(value: Any) -> Any:
    return base.json_ready(value)


def rolling_mean(values: pd.Series, samples: int) -> pd.Series:
    return values.rolling(samples, min_periods=1).mean()


def add_candidate_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    drip = df["Drip"].astype(float)
    mist = df["Mist"].astype(float)
    fan = df["Fan"].astype(float)
    light_log = df["Light_log"].astype(float)
    dryness = df["Indoor_Dryness"].astype(float)
    vpd = df["VPD_Proxy_In"].astype(float)
    temp = df["Temperature_In"].astype(float)
    humi = df["Humidity_In"].astype(float)

    df["Drip_Start"] = ((drip > 0.5) & (drip.shift(1, fill_value=0.0) <= 0.5)).astype(float)
    df["Mist_Start"] = ((mist > 0.5) & (mist.shift(1, fill_value=0.0) <= 0.5)).astype(float)
    df["Fan_Start"] = ((fan > 0.5) & (fan.shift(1, fill_value=0.0) <= 0.5)).astype(float)

    df["Drip_Roll_1min"] = rolling_mean(drip, 3)
    df["Drip_Roll_3min"] = rolling_mean(drip, 9)
    df["Drip_Roll_5min"] = rolling_mean(drip, 15)
    df["Mist_Roll_1min"] = rolling_mean(mist, 3)
    df["Mist_Roll_3min"] = rolling_mean(mist, 9)
    df["Fan_Roll_1min"] = rolling_mean(fan, 3)
    df["Fan_Roll_3min"] = rolling_mean(fan, 9)
    df["Fan_Roll_5min"] = rolling_mean(fan, 15)

    df["Fan_x_Indoor_Dryness"] = fan * dryness
    df["Fan_x_Humidity_In"] = fan * humi
    df["Fan_x_VPD"] = fan * vpd
    df["Fan_x_TempIn"] = fan * temp
    df["Mist_x_Indoor_Dryness"] = mist * dryness
    df["Mist_x_TempIn"] = mist * temp
    df["Mist_x_VPD"] = mist * vpd
    df["Drip_x_Indoor_Dryness"] = drip * dryness
    df["Drip_x_Light_log"] = drip * light_log
    df["Drip_x_VPD"] = drip * vpd

    df["LightLog_x_Dryness"] = light_log * dryness
    df["LightLog_x_VPD"] = light_log * vpd
    df["VPD_sq"] = vpd**2
    df["Light_log_sq"] = light_log**2
    df["TempIn_sq"] = temp**2
    df["Humidity_In_sq"] = humi**2

    df["VPD_x_Hour_sin"] = vpd * df["Hour_sin"].astype(float)
    df["VPD_x_Hour_cos"] = vpd * df["Hour_cos"].astype(float)
    df["LightLog_x_Hour_sin"] = light_log * df["Hour_sin"].astype(float)
    df["LightLog_x_Hour_cos"] = light_log * df["Hour_cos"].astype(float)

    return df


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "actuator_event_edges": ("Drip_Start", "Mist_Start", "Fan_Start"),
    "water_memory_roll": ("Drip_Roll_1min", "Drip_Roll_3min", "Drip_Roll_5min", "Mist_Roll_1min", "Mist_Roll_3min"),
    "fan_memory_roll": ("Fan_Roll_1min", "Fan_Roll_3min", "Fan_Roll_5min"),
    "fan_drying_interactions": ("Fan_x_Indoor_Dryness", "Fan_x_Humidity_In", "Fan_x_VPD", "Fan_x_TempIn"),
    "mist_context_interactions": ("Mist_x_Indoor_Dryness", "Mist_x_TempIn", "Mist_x_VPD"),
    "drip_context_interactions": ("Drip_x_Indoor_Dryness", "Drip_x_Light_log", "Drip_x_VPD"),
    "solar_drying_shape": ("LightLog_x_Dryness", "LightLog_x_VPD", "VPD_sq", "Light_log_sq"),
    "sensor_quadratic_shape": ("TempIn_sq", "Humidity_In_sq"),
    "daily_interactions": ("VPD_x_Hour_sin", "VPD_x_Hour_cos", "LightLog_x_Hour_sin", "LightLog_x_Hour_cos"),
}


SCREEN_SPECS = (
    base.ArxSpec(12, 3, 2, 0.1),
    base.ArxSpec(12, 6, 1, 10.0),
    base.ArxSpec(24, 6, 3, 1.0),
)


def run_arx_spec_search(
    label: str,
    input_cols: tuple[str, ...],
    df_train_z: pd.DataFrame,
    df_val_z: pd.DataFrame,
    df_test_z: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    clip: tuple[float, float],
    cfg: base.OutdoorConfig,
    specs: tuple[base.ArxSpec, ...],
    include_control_horizon: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fitted: dict[str, tuple[base.ArxSpec, np.ndarray]] = {}
    for spec in specs:
        theta = base.fit_arx(df_train_z, spec, input_cols)
        val_1, val_true_1 = base.predict_arx_one_step(df_val_z, theta, spec, input_cols, clip)
        val_sim, val_true_sim = base.simulate_arx(df_val_z, theta, spec, input_cols, clip)
        robust = base.validation_blocks_arx(df_val_z, theta, spec, input_cols, stats, clip, cfg)
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
                "val_FIT_1step": base.fit_metrics(base.inverse_y(val_true_1, stats), base.inverse_y(val_1, stats))["FIT"],
                "val_FIT_sim": base.fit_metrics(base.inverse_y(val_true_sim, stats), base.inverse_y(val_sim, stats))["FIT"],
                **robust,
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values("val_robust_score", ascending=False).reset_index(drop=True)
    selected = leaderboard.iloc[0].to_dict()
    spec, theta = fitted[str(selected["model"])]
    if include_control_horizon:
        validation = base.evaluate_arx(df_val_z, theta, spec, input_cols, stats, clip, cfg, include_control_horizon=True)
        test_eval = base.evaluate_arx(df_test_z, theta, spec, input_cols, stats, clip, cfg, include_control_horizon=True)
    else:
        y_val, yt_val = base.simulate_arx(df_val_z, theta, spec, input_cols, clip)
        y_test, yt_test = base.simulate_arx(df_test_z, theta, spec, input_cols, clip)
        validation = {"metrics_sim": base.fit_metrics(base.inverse_y(yt_val, stats), base.inverse_y(y_val, stats))}
        test_eval = {"metrics_sim": base.fit_metrics(base.inverse_y(yt_test, stats), base.inverse_y(y_test, stats))}
    return {
        "input_set": label,
        "input_cols": list(input_cols),
        "leaderboard": leaderboard,
        "selected_by_validation": selected,
        "spec": spec,
        "theta": theta,
        "validation": validation,
        "test": test_eval,
    }


def fit_dataset(
    label: str,
    input_cols: tuple[str, ...],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: base.OutdoorConfig,
    specs: tuple[base.ArxSpec, ...] | None = None,
    include_control_horizon: bool = True,
) -> dict[str, Any]:
    stats = base.fit_scale_stats(train, input_cols)
    train_z = base.apply_scale(train, stats)
    val_z = base.apply_scale(val, stats)
    test_z = base.apply_scale(test, stats)
    clip = tuple(float(v) for v in np.quantile(train_z["Soil_Moisture"], (0.005, 0.995)))
    if specs is None:
        result = base.run_arx_search(label, input_cols, train_z, val_z, test_z, stats, clip, cfg)
        search_mode = "full"
    else:
        result = run_arx_spec_search(
            label,
            input_cols,
            train_z,
            val_z,
            test_z,
            stats,
            clip,
            cfg,
            specs,
            include_control_horizon=include_control_horizon,
        )
        search_mode = "screen"
    selected = result["selected_by_validation"]
    test_metrics = result["test"]
    return {
        "label": label,
        "search_mode": search_mode,
        "input_cols": list(input_cols),
        "n_input_cols": len(input_cols),
        "selected_model": selected["model"],
        "val_FIT_sim": result["validation"]["metrics_sim"]["FIT"],
        "val_robust_score": selected["val_robust_score"],
        "val_block_mean_FIT_sim": selected["val_block_mean_FIT_sim"],
        "val_block_std_FIT_sim": selected["val_block_std_FIT_sim"],
        "test_FIT_sim": test_metrics["metrics_sim"]["FIT"],
        "test_FIT_60": test_metrics.get("metrics_60", {}).get("FIT", float("nan")),
        "test_RMSE_sim": test_metrics["metrics_sim"]["RMSE"],
        "result": result,
    }


def compact_row(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row["label"],
        "search_mode": row["search_mode"],
        "n_input_cols": row["n_input_cols"],
        "selected_model": row["selected_model"],
        "val_robust_score": row["val_robust_score"],
        "delta_val_robust": row["val_robust_score"] - baseline["val_robust_score"],
        "val_FIT_sim": row["val_FIT_sim"],
        "test_FIT_sim": row["test_FIT_sim"],
        "delta_test_FIT_sim": row["test_FIT_sim"] - baseline["test_FIT_sim"],
        "test_FIT_60": row["test_FIT_60"],
        "test_RMSE_sim": row["test_RMSE_sim"],
    }


def run_feature_search(cfg: FeatureSearchConfig, data_csv: str | None = None) -> dict[str, Any]:
    base_cfg = base.OutdoorConfig(days=cfg.days, sampling_seconds=cfg.sampling_seconds, seed=cfg.seed)
    raw_df = base.load_raw_data(base_cfg, data_csv)
    df = add_candidate_features(base.add_features(raw_df))
    train, val, test = base.split_time(df, base_cfg)

    baseline_cols = tuple(base.INSIDE_INPUT_COLS)
    baseline = fit_dataset("baseline_17", baseline_cols, train, val, test, base_cfg)
    baseline_screen = fit_dataset(
        "baseline_17_screen",
        baseline_cols,
        train,
        val,
        test,
        base_cfg,
        SCREEN_SPECS,
        include_control_horizon=False,
    )

    single_rows: list[dict[str, Any]] = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        cols = tuple(dict.fromkeys((*baseline_cols, *group_cols)))
        row = fit_dataset(
            f"screen_base_plus_{group_name}",
            cols,
            train,
            val,
            test,
            base_cfg,
            SCREEN_SPECS,
            include_control_horizon=False,
        )
        single_rows.append(compact_row(row, baseline_screen))

    selected_groups: list[str] = []
    current_cols = baseline_cols
    current = baseline_screen
    greedy_steps: list[dict[str, Any]] = [
        {
            "step": 0,
            "action": "baseline",
            **compact_row(current, baseline_screen),
        }
    ]

    remaining = set(FEATURE_GROUPS)
    step = 1
    while remaining:
        candidates: list[dict[str, Any]] = []
        for group_name in sorted(remaining):
            cols = tuple(dict.fromkeys((*current_cols, *FEATURE_GROUPS[group_name])))
            row = fit_dataset(
                f"screen_greedy_add_{group_name}",
                cols,
                train,
                val,
                test,
                base_cfg,
                SCREEN_SPECS,
                include_control_horizon=False,
            )
            row["candidate_group"] = group_name
            candidates.append(row)

        best = max(candidates, key=lambda item: item["val_robust_score"])
        gain = best["val_robust_score"] - current["val_robust_score"]
        if gain < cfg.min_robust_gain:
            break

        group_name = str(best["candidate_group"])
        selected_groups.append(group_name)
        remaining.remove(group_name)
        current_cols = tuple(dict.fromkeys((*current_cols, *FEATURE_GROUPS[group_name])))
        current = best
        greedy_steps.append(
            {
                "step": step,
                "action": f"add_{group_name}",
                **compact_row(current, baseline_screen),
            }
        )
        step += 1

    final_cols = tuple(current_cols)
    final_selected = baseline
    if selected_groups:
        final_selected = fit_dataset("full_confirm_selected_features", final_cols, train, val, test, base_cfg)

    full_confirm_rows = [compact_row(baseline, baseline)]
    if selected_groups:
        full_confirm_rows.append(compact_row(final_selected, baseline))

    payload = {
        "config": asdict(cfg),
        "data_policy": {
            "source_data": str(Path(data_csv).resolve()) if data_csv is not None else "simulated_physical_protocol",
            "selection_policy": "chọn feature bằng validation robust score, không chọn theo test",
            "target_leakage": False,
            "candidate_rule": "chỉ dùng cảm biến trong, actuator, thời gian và biến suy ra từ quá khứ/hiện tại",
        },
        "baseline": compact_row(baseline, baseline),
        "baseline_screen": compact_row(baseline_screen, baseline_screen),
        "single_group_leaderboard": sorted(single_rows, key=lambda item: item["val_robust_score"], reverse=True),
        "greedy_steps": greedy_steps,
        "full_confirm": full_confirm_rows,
        "selected_groups": selected_groups,
        "selected_input_cols": final_selected["input_cols"],
        "selected_summary": compact_row(final_selected, baseline),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(payload["single_group_leaderboard"]).to_csv(RESULTS_DIR / "single_group_leaderboard.csv", index=False)
    pd.DataFrame(greedy_steps).to_csv(RESULTS_DIR / "greedy_steps.csv", index=False)
    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
        f.write("\n")
    write_summary(payload)
    return payload


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_summary(payload: dict[str, Any]) -> None:
    baseline = payload["baseline"]
    selected = payload["selected_summary"]
    lines = [
        "# Dò Feature Cho ARX Thuần",
        "",
        "## Nguyên tắc",
        "",
        "- Chỉ thử biến đo được hoặc suy ra từ cảm biến trong, actuator và thời gian.",
        "- Không dùng `Soil_Moisture_True`, trạng thái ẩn mô phỏng, cảm biến ngoài hoặc dữ liệu tương lai.",
        "- Feature được chọn bằng validation robust score; test chỉ dùng để báo cáo sau khi đã chọn.",
        "",
        "## Kết quả ngắn",
        "",
        f"- Baseline: `FIT_sim = {fmt(baseline['test_FIT_sim'])}`, validation robust `{fmt(baseline['val_robust_score'])}`.",
        f"- Bản chọn theo validation: `FIT_sim = {fmt(selected['test_FIT_sim'])}`, validation robust `{fmt(selected['val_robust_score'])}`.",
        f"- Chênh lệch test so với baseline: `{fmt(selected['delta_test_FIT_sim'])}` điểm.",
        "",
        "## Nhóm feature được chọn",
        "",
    ]
    if payload["selected_groups"]:
        for group in payload["selected_groups"]:
            cols = ", ".join(FEATURE_GROUPS[group])
            lines.append(f"- `{group}`: {cols}")
    else:
        lines.append("- Không có nhóm feature nào vượt ngưỡng validation robust.")

    lines.extend(
        [
            "",
            "## Cảnh báo",
            "",
            "Nếu một feature làm test tăng nhưng validation robust giảm, không đưa vào bản final vì đó là chọn theo test và dễ bị xem là leakage phương pháp.",
            "",
        ]
    )
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dò feature vật lý cho ARX final.")
    parser.add_argument("--data-csv", default=None, help="Đường dẫn CSV log thật. Nếu bỏ trống, dùng dữ liệu mô phỏng có kiểm soát.")
    parser.add_argument("--days", type=int, default=16)
    parser.add_argument("--sampling-seconds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=305031)
    parser.add_argument("--min-robust-gain", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_feature_search(
        FeatureSearchConfig(
            days=args.days,
            sampling_seconds=args.sampling_seconds,
            seed=args.seed,
            min_robust_gain=args.min_robust_gain,
        ),
        args.data_csv,
    )
    selected = payload["selected_summary"]
    print("=== ARX feature search ===")
    print(
        f"Selected FIT_sim={selected['test_FIT_sim']:.3f}, "
        f"val_robust={selected['val_robust_score']:.3f}, "
        f"groups={payload['selected_groups']}"
    )
    print(f"Đã lưu kết quả vào {RESULTS_DIR}")


if __name__ == "__main__":
    main()
