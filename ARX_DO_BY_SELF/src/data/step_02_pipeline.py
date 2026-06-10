from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.step_00_data_io import (
    build_data_files,
    build_reference_data_files,
    format_model_data,
    input_csv_paths,
    project_root,
)
from data.step_01_clean_data import clean_all_data_files


# Xóa CSV output cũ trong một thư mục.
def clear_csv_outputs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.rglob("*.csv"):
        path.unlink()


# Xóa file/folder output có thể tái tạo trong thư mục raw đầu vào.
def clear_data_tree(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in sorted(directory.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


# Chạy toàn bộ pipeline build data.
def run(source_dir: Path | None = None) -> None:
    root = project_root()
    data_dir = root / "data"
    input_dir = data_dir / "_01_data"
    clean_dir = data_dir / "_02_clean_data"
    data_path = data_dir / "mini_greenhouse_5s_data.csv"
    if source_dir is not None and not source_dir.is_absolute():
        source_dir = root / source_dir

    clear_csv_outputs(clean_dir)

    if source_dir is None:
        raw_paths = input_csv_paths(input_dir)
    elif data_path.exists() and source_dir == data_path.parent:
        clear_data_tree(input_dir)
        raw_paths = build_reference_data_files(input_dir, data_path)
    else:
        clear_data_tree(input_dir)
        source_data_path = source_dir / "mini_greenhouse_5s_data.csv"
        if source_data_path.exists():
            raw_paths = build_reference_data_files(input_dir, source_data_path)
        else:
            raw_paths = build_data_files(input_dir, source_dir)

    if not raw_paths and data_path.exists():
        raw_paths = build_reference_data_files(input_dir, data_path)

    raw_df = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True)
    raw_df["Timestamp"] = pd.to_datetime(raw_df["Timestamp"], errors="coerce")
    raw_df = raw_df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    format_model_data(raw_df).to_csv(input_dir / "00_raw_tong_hop.csv", index=False)

    cleaned_data = clean_all_data_files(raw_paths, clean_dir)
    format_model_data(cleaned_data).to_csv(data_path, index=False)
