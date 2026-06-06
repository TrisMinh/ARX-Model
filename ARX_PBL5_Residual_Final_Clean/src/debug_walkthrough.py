from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

import arx_backbone_pipeline as base
import arx_residual_experiment as residual


def log_step(title: str, value: Any, note: str = "", max_rows: int = 3) -> None:
    """In ra kết quả trung gian sau mỗi bước để người mới dễ nhìn luồng dữ liệu."""
    print("\n" + "=" * 88)
    print(f"STEP: {title}")
    if note:
        print(f"NOTE: {note}")

    if isinstance(value, pd.DataFrame):
        print(f"type=DataFrame shape={value.shape}")
        print(f"columns={list(value.columns)}")
        print(value.head(max_rows).to_string(index=False))
        return

    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        preview = flat[: min(8, flat.size)]
        print(f"type=ndarray shape={value.shape} dtype={value.dtype}")
        print(f"preview={np.array2string(preview, precision=4, separator=', ')}")
        return

    if isinstance(value, dict):
        text = json.dumps(base.json_ready(value), ensure_ascii=False, indent=2)
        if len(text) > 3500:
            text = text[:3500] + "\n... <cắt bớt log cho dễ đọc>"
        print(text)
        return

    print(repr(value))


def run_walkthrough(args: argparse.Namespace) -> None:
    cfg = base.OutdoorConfig(days=args.days, sampling_seconds=args.sampling_seconds, seed=args.seed)
    log_step("01. Tạo OutdoorConfig", cfg.__dict__, "Các tham số nền cho dữ liệu, split và ARX.")

    raw_df = base.load_raw_data(cfg, args.data_csv)
    log_step("02. load_raw_data", raw_df, "Nếu không truyền CSV thật, hàm sinh dữ liệu mô phỏng có kiểm soát.")

    df = base.add_features(raw_df)
    log_step("03. add_features", df[list(base.INSIDE_INPUT_COLS) + ["Soil_Moisture"]], "Thêm feature hợp lệ từ sensor, actuator và timestamp.")

    train, val, test = base.split_time(df, cfg)
    log_step(
        "04. split_time",
        {"train_rows": len(train), "val_rows": len(val), "test_rows": len(test)},
        "Chia theo thời gian, không shuffle.",
    )

    stats = base.fit_scale_stats(train, base.INSIDE_INPUT_COLS)
    log_step("05. fit_scale_stats", {k: stats[k] for k in list(stats)[:6]}, "Scaler chỉ fit trên train.")

    train_z = base.apply_scale(train, stats)
    val_z = base.apply_scale(val, stats)
    test_z = base.apply_scale(test, stats)
    log_step("06. apply_scale train", train_z[["Soil_Moisture", "Temperature_In", "Humidity_In"]])

    clip = tuple(float(v) for v in np.quantile(train_z["Soil_Moisture"], (0.005, 0.995)))
    log_step("07. Tính clip", {"clip_low": clip[0], "clip_high": clip[1]}, "Giới hạn dự đoán để mô phỏng không trôi quá xa.")

    if args.full_search:
        arx = base.run_arx_search("inside_only", base.INSIDE_INPUT_COLS, train_z, val_z, test_z, stats, clip, cfg)
        spec = arx["spec"]
        theta = arx["theta"]
        log_step("08. run_arx_search selected", arx["selected_by_validation"], "Đây là cấu hình ARX được validation chọn.")
    else:
        spec = base.ArxSpec(na=12, nb=6, nk=1, alpha=10.0)
        theta = base.fit_arx(train_z, spec, base.INSIDE_INPUT_COLS)
        log_step(
            "08. fit_arx với spec final hiện tại",
            {"spec": spec.name, "theta_len": len(theta), "theta_preview": theta[:8]},
            "Dùng nhanh spec đang là final để log không mất nhiều thời gian.",
        )

    x_train, y_train = base.build_arx_matrix(train_z, spec, base.INSIDE_INPUT_COLS)
    log_step("09. build_arx_matrix X", x_train, "Mỗi dòng là một thời điểm, mỗi cột là một lag/output/input/intercept.")
    log_step("10. build_arx_matrix y", y_train, "Vector mục tiêu sau khi bỏ các dòng chưa đủ lag.")

    y_one, yt_one = base.predict_arx_one_step(test_z, theta, spec, base.INSIDE_INPUT_COLS, clip)
    log_step("11. predict_arx_one_step metrics", base.fit_metrics(base.inverse_y(yt_one, stats), base.inverse_y(y_one, stats)))

    y_sim, yt_sim = base.simulate_arx(test_z, theta, spec, base.INSIDE_INPUT_COLS, clip)
    log_step("12. simulate_arx metrics", base.fit_metrics(base.inverse_y(yt_sim, stats), base.inverse_y(y_sim, stats)))

    y_60, yt_60 = base.simulate_arx_n_step(test_z, theta, spec, base.INSIDE_INPUT_COLS, clip, cfg.n_step_control)
    log_step("13. simulate_arx_n_step 60 metrics", base.fit_metrics(base.inverse_y(yt_60, stats), base.inverse_y(y_60, stats)))

    residual_cfg = residual.ResidualConfig(days=args.days, sampling_seconds=args.sampling_seconds, seed=args.seed)
    log_step("14. Tạo ResidualConfig", residual_cfg.__dict__, "Cấu hình lag residual, shrink và model phụ.")

    y_arx_train_z, y_true_train_z = base.simulate_arx(train_z, theta, spec, base.INSIDE_INPUT_COLS, clip)
    y_arx_val_z, y_true_val_z = base.simulate_arx(val_z, theta, spec, base.INSIDE_INPUT_COLS, clip)
    y_arx_test_z, y_true_test_z = base.simulate_arx(test_z, theta, spec, base.INSIDE_INPUT_COLS, clip)
    x_res_train = residual.residual_features(train_z, y_arx_train_z, spec, base.INSIDE_INPUT_COLS, residual_cfg)
    x_res_val = residual.residual_features(val_z, y_arx_val_z, spec, base.INSIDE_INPUT_COLS, residual_cfg)
    x_res_test = residual.residual_features(test_z, y_arx_test_z, spec, base.INSIDE_INPUT_COLS, residual_cfg)
    log_step("15. residual_features train", x_res_train, "Feature để học sai số còn lại của ARX.")
    log_step("16. residual_features split shapes", {"train": x_res_train.shape, "val": x_res_val.shape, "test": x_res_test.shape})

    y_res_train = y_true_train_z - y_arx_train_z
    models = dict(residual.candidate_models(residual_cfg))
    model = models["hgb_leaf15_l2_0.01"]
    model.fit(x_res_train, y_res_train)
    corr_val = np.asarray(model.predict(x_res_val), dtype=float)
    corr_test = np.asarray(model.predict(x_res_test), dtype=float)
    shrink = 0.25
    y_hybrid_val_z = np.clip(y_arx_val_z + shrink * corr_val, clip[0], clip[1])
    y_hybrid_test_z = np.clip(y_arx_test_z + shrink * corr_test, clip[0], clip[1])
    log_step("17. robust_block_score residual", residual.robust_block_score(y_true_val_z, y_hybrid_val_z, stats, residual_cfg))
    log_step("18. Hybrid residual test metrics", residual.inverse_metrics(y_true_test_z, y_hybrid_test_z, stats))

    print("\nHOÀN TẤT WALKTHROUGH.")
    print("Muốn chạy đúng pipeline final đầy đủ:")
    print(r"python -B .\ARX_PBL5_Residual_Final_Clean\src\arx_residual_experiment.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log từng bước build ARX + Hybrid residual.")
    parser.add_argument("--data-csv", default=None, help="CSV thật. Bỏ trống thì dùng dữ liệu mô phỏng.")
    parser.add_argument("--days", type=int, default=16)
    parser.add_argument("--sampling-seconds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=305031)
    parser.add_argument("--full-search", action="store_true", help="Chạy full ARX search thay vì dùng spec final hiện tại.")
    return parser.parse_args()


def main() -> None:
    run_walkthrough(parse_args())


if __name__ == "__main__":
    main()
