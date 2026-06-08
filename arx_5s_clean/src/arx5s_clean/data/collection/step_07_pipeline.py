from __future__ import annotations

from pathlib import Path

import pandas as pd

from arx5s_clean.data.collection.step_00_source_data import DataSource
from arx5s_clean.data.collection.step_01_raw_sessions import build_raw_sessions
from arx5s_clean.data.collection.step_04_preprocess import clean_all_sessions
from arx5s_clean.data.collection.step_05_augmentation import build_augmented_training_data


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def clear_csv_outputs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.csv"):
        path.unlink()


def run(days: int, seed: int, source: DataSource = "legacy", real_dir: Path | None = None) -> None:
    root = project_root()
    data_dir = root / "data"
    raw_dir = data_dir / "01_raw_sessions"
    processed_dir = data_dir / "02_cleaned_sessions"

    clear_csv_outputs(raw_dir)
    clear_csv_outputs(processed_dir)

    raw_paths = build_raw_sessions(raw_dir, seed, source, real_dir)
    raw_df = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True)
    raw_df.to_csv(raw_dir / "00_raw_tong_hop.csv", index=False)

    cleaned_data = clean_all_sessions(raw_paths, processed_dir)
    final_df = build_augmented_training_data(cleaned_data, days, seed, source)

    final_df.to_csv(data_dir / "mini_greenhouse_5s_data.csv", index=False)
