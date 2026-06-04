from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "greenhouse_data.csv"
OUT = PROJECT_ROOT / "greenhouse_data_missing.csv"
REPORT = PROJECT_ROOT / "greenhouse_data_missing_report.csv"


def add_missing_block(df: pd.DataFrame, col: str, start: int, length: int) -> None:
    end = min(start + length, len(df))
    df.loc[start:end - 1, col] = np.nan


def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["Timestamp"])
    df_missing = df.copy()

    # Deterministic missing pattern. Keep Timestamp intact so the time-series order remains valid.
    missing_plan = [
        ("Soil_Moisture", 1200, 18, "continuous sensor block"),
        ("Temperature", 4800, 24, "continuous sensor block"),
        ("Humidity", 9600, 30, "continuous sensor block"),
        ("Light", 14400, 36, "continuous sensor block"),
        ("Soil_Low_SP", 18000, 12, "setpoint block"),
        ("Soil_High_SP", 18000, 12, "setpoint block"),
        ("Drip", 22000, 20, "binary actuator block"),
        ("Mist", 26000, 16, "binary actuator block"),
        ("Fan", 30000, 16, "binary actuator block"),
        ("Month", 34000, 10, "time-context block"),
        ("Season", 38000, 10, "categorical-context block"),
    ]
    for col, start, length, _ in missing_plan:
        add_missing_block(df_missing, col, start, length)

    report_rows = []
    for col, _, _, kind in missing_plan:
        report_rows.append(
            {
                "column": col,
                "kind": kind,
                "missing_count": int(df_missing[col].isna().sum()),
                "missing_pct": 100.0 * float(df_missing[col].isna().mean()),
            }
        )
    report = pd.DataFrame(report_rows)

    df_missing.to_csv(OUT, index=False)
    report.to_csv(REPORT, index=False)
    print(f"Wrote {OUT}")
    print(f"Wrote {REPORT}")
    print(report)


if __name__ == "__main__":
    main()
