from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Literal

import pandas as pd

from arx5s_clean.data.collection.step_00_schema import MODEL_COLS, normalize_model_columns

DataSource = Literal["legacy", "real", "mau_cu", "thuc_te"]
NormalizedSource = Literal["legacy", "real"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def normalize_source(source: DataSource) -> NormalizedSource:
    if source in ("mau_cu", "legacy"):
        return "legacy"
    if source in ("thuc_te", "real"):
        return "real"
    raise ValueError("source must be one of: legacy, real, mau_cu, thuc_te")


def real_data_dir(real_dir: Path | None = None) -> Path:
    project_dir = project_root() / "arx_5s_clean"
    if real_dir is None:
        return project_dir / "data" / "01_raw_sessions_real"
    return real_dir if real_dir.is_absolute() else project_dir / real_dir


def real_csv_paths(real_dir: Path | None = None) -> list[Path]:
    files = sorted(real_data_dir(real_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No real CSV files found in {real_data_dir(real_dir)}")
    return files


def load_model_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_model_columns(df)


def load_source_data(source: DataSource, real_dir: Path | None = None) -> pd.DataFrame:
    if normalize_source(source) == "legacy":
        csv_text = subprocess.check_output(
            ["git", "show", "HEAD:arx_5s_clean/data/mini_greenhouse_5s_data.csv"],
            cwd=project_root(),
            text=True,
        )
        return normalize_model_columns(pd.read_csv(io.StringIO(csv_text)))

    frames = [load_model_csv(path) for path in real_csv_paths(real_dir)]
    out = pd.concat(frames, ignore_index=True)
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    return out.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True).loc[:, MODEL_COLS]
