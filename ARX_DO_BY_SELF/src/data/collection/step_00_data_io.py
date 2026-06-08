from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Các cột raw bắt buộc phải có trong file thu thập.
MODEL_COLS: tuple[str, ...] = (
    "Timestamp",
    "Temperature",
    "Humidity",
    "Light",
    "Soil_Moisture",
    "Drip",
    "Mist",
    "Fan",
)

SAMPLE_SECONDS = 5

SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "Temperature": (15.0, 45.0),
    "Humidity": (30.0, 100.0),
    "Light": (0.0, 1200.0),
    "Soil_Moisture": (0.0, 100.0),
}

ACTUATOR_COLS: tuple[str, ...] = ("Drip", "Mist", "Fan")


# Làm tròn sensor cho giống dữ liệu đo thực tế.
def format_model_data(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_model_columns(df)
    for col in SENSOR_RANGES:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(1)
    for col in ACTUATOR_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = (out[col] >= 0.5).astype(float)
    return out.loc[:, MODEL_COLS]


# Lấy thư mục gốc của project ARX_DO_BY_SELF.
def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Lấy thư mục chứa data đầu vào.
def input_data_dir(input_dir: Path | None = None) -> Path:
    if input_dir is None:
        return project_root() / "data" / "_01_data"
    return input_dir if input_dir.is_absolute() else project_root() / input_dir


# Lấy danh sách CSV trong thư mục data đầu vào.
def input_csv_paths(input_dir: Path | None = None) -> list[Path]:
    root = input_data_dir(input_dir)
    return sorted(path for path in root.rglob("*.csv") if not path.name.startswith("00_"))


# Kiểm tra và trả về đúng 8 cột raw cần dùng.
def normalize_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in MODEL_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")
    return df.loc[:, MODEL_COLS].copy()


# Đọc một CSV và kiểm tra đúng 8 cột model cần.
def load_model_csv(path: Path) -> pd.DataFrame:
    return normalize_model_columns(pd.read_csv(path))


# Đọc data 5s chuẩn đang dùng làm mốc so sánh.
def load_reference_data(data_path: Path) -> pd.DataFrame:
    df = load_model_csv(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True).loc[:, MODEL_COLS]


# Thêm lỗi nhỏ giống quá trình thu data để bước clean có ý nghĩa.
def inject_collection_artifacts(df: pd.DataFrame, data_name: str) -> pd.DataFrame:
    out = df.copy()
    timestamp_shifts: dict[str, list[tuple[int, int]]] = {
        "morning_anchor": [(1, 1), (2, -1), (3, 2), (4, -3), (5, 1), (6, -2), (7, 2), (8, -1), (9, 1), (10, -2), (12, 1), (86, -2), (214, 1), (410, -1), (620, 2), (900, -1)],
        "noon_anchor": [(1, -1), (2, 2), (3, -2), (4, 1), (5, -1), (6, 2), (7, -3), (8, 1), (9, -1), (10, 2), (15, -1), (120, 2), (250, -2), (410, 1), (620, -1), (980, 2)],
        "afternoon_anchor": [(1, 2), (2, -2), (3, 1), (4, -1), (5, 2), (6, -3), (7, 1), (8, -1), (9, 2), (10, -2), (25, 1), (300, -2), (450, 1), (620, 2), (860, -1), (1110, 1)],
        "night_anchor": [(1, -2), (2, 1), (3, -1), (4, 2), (5, -3), (6, 1), (7, -1), (8, 2), (9, -2), (10, 1), (35, -1), (500, 2), (610, -2), (740, 1), (980, -1), (1180, 2)],
    }

    for idx, seconds in timestamp_shifts.get(data_name, []):
        if 0 <= idx < len(out):
            out.loc[out.index[idx], "Timestamp"] = pd.to_datetime(out.loc[out.index[idx], "Timestamp"]) + pd.Timedelta(
                seconds=seconds
            )

    if data_name == "morning_anchor":
        out.loc[out.index[86], "Light"] = np.nan
        out = out.drop(out.index[[214, 215]])
    elif data_name == "noon_anchor":
        out.loc[out.index[120:126], ["Temperature", "Humidity"]] = np.nan
        out = out.drop(out.index[[410]])
    elif data_name == "afternoon_anchor":
        out.loc[out.index[300:302], "Soil_Moisture"] = np.nan
        duplicate = out.iloc[[620]].copy()
        out = pd.concat([out.iloc[:621], duplicate, out.iloc[621:]], ignore_index=True)
    elif data_name == "night_anchor":
        out.loc[out.index[500], "Fan"] = np.nan
        out = out.drop(out.index[[740, 741, 742]])
    return out.loc[:, MODEL_COLS]


# Tạo tên file data đầu vào.
def data_output_name(index: int, source_stem: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in source_stem)
    if safe.endswith("_raw"):
        return f"{index:02d}_{safe}.csv"
    return f"{index:02d}_{safe}_raw.csv"


# Đọc CSV gốc và copy sang thư mục data đầu vào của pipeline.
def build_data_files(output_dir: Path, source_dir: Path | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for idx, source_path in enumerate(input_csv_paths(source_dir), start=1):
        raw = format_model_data(load_model_csv(source_path))
        path = output_dir / data_output_name(idx, source_path.stem)
        raw.to_csv(path, index=False)
        paths.append(path)

    if not paths:
        raise FileNotFoundError(f"No CSV files found in {input_data_dir(source_dir)}")
    return paths


# Đọc toàn bộ CSV thật rồi ghép thành một bảng nguồn.
def load_source_data(source_dir: Path | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_path in input_csv_paths(source_dir):
        frames.append(format_model_data(load_model_csv(source_path)))

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {input_data_dir(source_dir)}")

    data = pd.concat(frames, ignore_index=True)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    data = data.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return data.loc[:, MODEL_COLS]


# Tách data chuẩn thành các file data đầu vào giống bản 5s chuẩn.
def build_reference_data_files(output_dir: Path, data_path: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data = load_reference_data(data_path)

    windows = (
        ("morning_anchor", "2026-01-01 07:00:00", "2026-01-01 09:00:00"),
        ("noon_anchor", "2026-01-01 11:30:00", "2026-01-01 13:30:00"),
        ("afternoon_anchor", "2026-01-01 15:00:00", "2026-01-01 17:00:00"),
        ("night_anchor", "2026-01-01 20:00:00", "2026-01-01 22:00:00"),
    )

    paths: list[Path] = []
    for name, start, end in windows:
        df = source_data[
            (source_data["Timestamp"] >= pd.Timestamp(start)) & (source_data["Timestamp"] < pd.Timestamp(end))
        ].copy()
        df = inject_collection_artifacts(df.reset_index(drop=True), name)
        path = output_dir / f"{name}_raw.csv"
        format_model_data(df).to_csv(path, index=False)
        paths.append(path)
    return paths
