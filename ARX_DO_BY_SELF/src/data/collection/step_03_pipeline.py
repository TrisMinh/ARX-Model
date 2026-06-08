from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.collection.step_00_data_io import (
    build_reference_data_files,
    format_model_data,
    input_csv_paths,
    load_reference_data,
    load_source_data,
    project_root,
)
from data.collection.step_01_clean_data import clean_all_data_files
from data.collection.step_02_generate_data import (
    build_collection_session_files,
    build_reference_training_data,
    build_training_data,
)


# Xóa CSV output cũ trong một thư mục.
def clear_csv_outputs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.rglob("*.csv"):
        path.unlink()


# Chạy toàn bộ pipeline build data.
def run(days: int, seed: int, source_dir: Path | None = None) -> None:
    root = project_root()
    data_dir = root / "data"
    input_dir = data_dir / "_01_data"
    clean_dir = data_dir / "_02_clean_data"
    data_path = data_dir / "mini_greenhouse_5s_data.csv"

    clear_csv_outputs(clean_dir)

    use_reference_data = False
    if source_dir is None:
        raw_paths = input_csv_paths(input_dir)
    elif data_path.exists() and source_dir == data_path.parent:
        raw_paths = build_reference_data_files(input_dir, data_path)
        use_reference_data = True
    else:
        clear_csv_outputs(input_dir)
        for child in input_dir.iterdir():
            if child.is_dir():
                for path in child.rglob("*"):
                    if path.is_file():
                        path.unlink()
                child.rmdir()
        source_data = load_source_data(source_dir)
        raw_paths = build_collection_session_files(input_dir, source_data, seed)

    if not raw_paths and data_path.exists():
        raw_paths = build_reference_data_files(input_dir, data_path)
        use_reference_data = True

    raw_df = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True)
    raw_df["Timestamp"] = pd.to_datetime(raw_df["Timestamp"], errors="coerce")
    raw_df = raw_df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    format_model_data(raw_df).to_csv(input_dir / "00_raw_tong_hop.csv", index=False)

    cleaned_data = clean_all_data_files(raw_paths, clean_dir)
    if use_reference_data:
        final_df = build_reference_training_data(load_reference_data(data_path), days)
    else:
        final_df = build_training_data(cleaned_data, days, seed)
    format_model_data(final_df).to_csv(data_path, index=False)
