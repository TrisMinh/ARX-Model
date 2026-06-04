from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_cell(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def print_bordered_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    if not rows:
        return

    widths: dict[str, int] = {}
    for col in columns:
        max_cell = max(len(_fmt_cell(row.get(col, ""))) for row in rows)
        widths[col] = max(len(col), max_cell)

    def sep(char: str = "-") -> str:
        return "+" + "+".join(char * (widths[col] + 2) for col in columns) + "+"

    print(sep("-"))
    print("| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |")
    print(sep("="))
    for row in rows:
        print(
            "| "
            + " | ".join(_fmt_cell(row.get(col, "")).ljust(widths[col]) for col in columns)
            + " |"
        )
        print(sep("-"))


def load_model(model_path: Path) -> dict:
    with model_path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    if model.get("model") != "ARX":
        raise ValueError("Only ARX model artifacts are supported")
    return model


class ARXRealtimePredictor:
    def __init__(self, model: dict):
        cfg = model["model_config"]
        self.na = int(cfg["na"])
        self.nb = int(cfg["nb"])
        self.nk = int(cfg["nk"])
        self.include_intercept = bool(cfg.get("include_intercept", False))
        self.input_cols = list(cfg["input_cols"])
        self.output_col = str(cfg["output_col"])
        self.theta = np.asarray(model["theta_hat"], dtype=float)

        self.max_u_lag = self.nk + self.nb - 1
        if self.max_u_lag <= 0:
            raise ValueError("Invalid lag setup in model")

        self.y_hist: deque[float] = deque(maxlen=self.na)
        self.u_hist: dict[str, deque[float]] = {
            col: deque(maxlen=self.max_u_lag) for col in self.input_cols
        }

    def _ensure_ready(self) -> None:
        if len(self.y_hist) < self.na:
            raise RuntimeError("Not enough output history to predict")
        for col in self.input_cols:
            if len(self.u_hist[col]) < self.max_u_lag:
                raise RuntimeError(f"Not enough input history for column: {col}")

    def warm_start_from_history(self, df_history: pd.DataFrame) -> None:
        required = [self.output_col, *self.input_cols]
        missing = [c for c in required if c not in df_history.columns]
        if missing:
            raise ValueError(f"History data missing columns: {missing}")

        if len(df_history) < max(self.na, self.max_u_lag):
            raise ValueError(
                "History CSV does not have enough rows for lag initialization"
            )

        y_tail = df_history[self.output_col].astype(float).tail(self.na)
        self.y_hist.clear()
        self.y_hist.extend(y_tail.to_list())

        for col in self.input_cols:
            u_tail = df_history[col].astype(float).tail(self.max_u_lag)
            self.u_hist[col].clear()
            self.u_hist[col].extend(u_tail.to_list())

    def _build_regressor(self) -> np.ndarray:
        self._ensure_ready()

        x: list[float] = []
        for lag in range(1, self.na + 1):
            x.append(float(self.y_hist[-lag]))

        for col in self.input_cols:
            hist = self.u_hist[col]
            if len(hist) < self.max_u_lag:
                raise RuntimeError(f"Not enough input history for column: {col}")
            for lag in range(self.nk, self.nk + self.nb):
                x.append(float(hist[-lag]))

        if self.include_intercept:
            x.append(1.0)

        return np.asarray(x, dtype=float)

    def _build_regressor_from_state(
        self,
        y_hist: deque[float],
        u_hist: dict[str, deque[float]],
    ) -> np.ndarray:
        x: list[float] = []
        for lag in range(1, self.na + 1):
            x.append(float(y_hist[-lag]))

        for col in self.input_cols:
            hist = u_hist[col]
            for lag in range(self.nk, self.nk + self.nb):
                x.append(float(hist[-lag]))

        if self.include_intercept:
            x.append(1.0)

        return np.asarray(x, dtype=float)

    def append_input(self, input_row: dict[str, float]) -> None:
        for col in self.input_cols:
            if col not in input_row:
                raise ValueError(f"Missing input column: {col}")
            self.u_hist[col].append(float(input_row[col]))

    def predict_current_step(self) -> float:
        x = self._build_regressor()
        return float(np.dot(x, self.theta))

    def update_output_state(self, y_value: float) -> None:
        self.y_hist.append(float(y_value))

    def forecast_n_steps_constant_input(
        self,
        current_input: dict[str, float],
        steps: int,
        clip_range: tuple[float, float] | None = None,
    ) -> list[float]:
        if steps <= 0:
            return []

        self._ensure_ready()
        y_hist = deque(self.y_hist, maxlen=self.na)
        u_hist = {
            col: deque(self.u_hist[col], maxlen=self.max_u_lag)
            for col in self.input_cols
        }

        preds: list[float] = []
        for _ in range(steps):
            for col in self.input_cols:
                u_hist[col].append(float(current_input[col]))
            x = self._build_regressor_from_state(y_hist, u_hist)
            y_pred = float(np.dot(x, self.theta))
            if clip_range is not None:
                y_pred = float(np.clip(y_pred, clip_range[0], clip_range[1]))
            y_hist.append(y_pred)
            preds.append(y_pred)
        return preds

    def ingest_input_and_predict(self, input_row: dict[str, float]) -> float:
        self.append_input(input_row)
        y_pred = self.predict_current_step()
        self.update_output_state(y_pred)
        return y_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime ARX prediction from streaming inputs"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path("greenhouse_observations_100.csv"),
        help="CSV containing new observations with input columns",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("arx_model.json"),
        help="Path to ARX model artifact JSON",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("greenhouse_data.csv"),
        help="CSV used to initialize lag history",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forecast horizon steps from current point (default: 12)",
    )
    parser.add_argument(
        "--sample-minutes",
        type=int,
        default=5,
        help="Minutes per data step for future timestamp labeling (default: 5)",
    )
    parser.add_argument(
        "--no-measured-soil",
        action="store_true",
        help="Ignore Soil_Moisture in input and run pure free-run state update",
    )
    parser.add_argument(
        "--clip-to-setpoint",
        action="store_true",
        help="Clip horizon forecasts to current [Soil_Low_SP, Soil_High_SP] when available",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between predictions (default: 2.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("realtime_predictions.csv"),
        help="Where to save prediction results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_model(args.model)
    predictor = ARXRealtimePredictor(model)

    df_history = pd.read_csv(args.history)
    predictor.warm_start_from_history(df_history)

    df_in = pd.read_csv(args.input_csv)
    missing_inputs = [c for c in predictor.input_cols if c not in df_in.columns]
    if missing_inputs:
        raise ValueError(f"Input CSV missing required columns: {missing_inputs}")

    has_measured_soil = predictor.output_col in df_in.columns
    use_measured = bool(has_measured_soil and not args.no_measured_soil)

    rows_out: list[dict[str, float | int | str]] = []

    has_setpoint_band = {"Soil_Low_SP", "Soil_High_SP"}.issubset(df_in.columns)

    print("Starting realtime ARX prediction...")
    print(f"Rows to process: {len(df_in)}")
    print(f"Interval: {args.interval:.2f}s")
    print(f"Horizon: {args.horizon} steps")
    print(f"Use measured soil state: {use_measured}")

    for idx, row in df_in.iterrows():
        input_row = {col: float(row[col]) for col in predictor.input_cols}
        predictor.append_input(input_row)
        y_pred_now = predictor.predict_current_step()

        if use_measured:
            y_state = float(row[predictor.output_col])
        else:
            y_state = y_pred_now
        predictor.update_output_state(y_state)

        clip_range: tuple[float, float] | None = None
        if args.clip_to_setpoint and has_setpoint_band:
            clip_range = (float(row["Soil_Low_SP"]), float(row["Soil_High_SP"]))

        future_preds = predictor.forecast_n_steps_constant_input(
            current_input=input_row,
            steps=args.horizon,
            clip_range=clip_range,
        )

        out_row = {
            "step": int(idx + 1),
            "timestamp": str(row["Timestamp"]) if "Timestamp" in df_in.columns else "",
            "measured_soil": float(row[predictor.output_col]) if has_measured_soil else np.nan,
            "pred_now": y_pred_now,
            "state_soil_used": y_state,
            "pred_h12": future_preds[-1] if future_preds else np.nan,
            **input_row,
        }

        if "Timestamp" in df_in.columns:
            current_ts = pd.to_datetime(row["Timestamp"])
            out_row["target_ts_h12"] = str(
                current_ts + pd.Timedelta(minutes=args.sample_minutes * args.horizon)
            )
        for h, pred_h in enumerate(future_preds, start=1):
            out_row[f"pred_h{h}"] = pred_h

        rows_out.append(out_row)

        row_view = {
            "step": int(idx + 1),
            "ts": out_row["timestamp"],
            "measured": out_row["measured_soil"],
            "pred_now": y_pred_now,
            "pred_h6": future_preds[5] if len(future_preds) >= 6 else np.nan,
            "pred_h12": future_preds[11] if len(future_preds) >= 12 else np.nan,
        }

        if has_setpoint_band and len(future_preds) >= 12:
            sp_low = float(row["Soil_Low_SP"])
            sp_high = float(row["Soil_High_SP"])
            if future_preds[11] > sp_high + 2.0:
                row_view["warn"] = "h12 > SP_high"
            elif future_preds[11] < sp_low - 2.0:
                row_view["warn"] = "h12 < SP_low"
            else:
                row_view["warn"] = "OK"

        columns = ["step", "ts", "measured", "pred_now", "pred_h6", "pred_h12"]
        if "warn" in row_view:
            columns.append("warn")
        print_bordered_table([row_view], columns)

        if idx < len(df_in) - 1 and args.interval > 0:
            time.sleep(args.interval)

    df_out = pd.DataFrame(rows_out)
    pd.DataFrame(df_out).to_csv(args.output, index=False)
    print(f"Saved predictions to: {args.output}")

    preview_cols = [
        "step",
        "timestamp",
        "measured_soil",
        "pred_now",
        "pred_h1",
        "pred_h6",
        "pred_h12",
    ]
    preview_cols = [c for c in preview_cols if c in df_out.columns]
    print("\nPreview table:")
    preview_rows = df_out[preview_cols].head(10).to_dict(orient="records")
    print_bordered_table(preview_rows, preview_cols)


if __name__ == "__main__":
    main()
