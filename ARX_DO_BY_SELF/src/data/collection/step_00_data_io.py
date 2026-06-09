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
        missing = out[col].isna()
        out[col] = (out[col] >= 0.5).astype(float)
        out.loc[missing, col] = np.nan
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
    n_rows = len(out)
    if n_rows == 0:
        return out.loc[:, MODEL_COLS]

    def spread_indices(count: int, low_frac: float, high_frac: float) -> list[int]:
        if count <= 0:
            return []
        if n_rows == 1:
            return [0]
        low = int(round((n_rows - 1) * low_frac))
        high = int(round((n_rows - 1) * high_frac))
        return sorted({int(idx) for idx in np.linspace(low, high, count).round().clip(0, n_rows - 1)})

    try:
        block_id = int(data_name.split("_", 1)[0])
    except ValueError:
        block_id = 0

    timestamp_offset = (1, 2, 3, 4, 1, 2, 3, 4, 1)[block_id % 9]
    out["Timestamp"] = pd.to_datetime(out["Timestamp"]) + pd.Timedelta(
        seconds=timestamp_offset
    )

    first_ts = pd.to_datetime(out["Timestamp"].iloc[0])

    def add_missing_points(points: list[tuple[float, str]]) -> None:
        for frac, col in points:
            idx = spread_indices(1, frac, frac)[0]
            out.loc[out.index[idx], col] = np.nan

    def add_missing_span(frac: float, length: int, cols: tuple[str, ...]) -> None:
        start = spread_indices(1, frac, frac)[0]
        end = min(start + length, n_rows)
        if start < end:
            out.loc[out.index[start:end], list(cols)] = np.nan

    missing_profile = (first_ts.day + block_id * 5) % 12
    if missing_profile == 1:
        add_missing_points([(0.18, "Temperature")])
    elif missing_profile == 2:
        add_missing_points([(0.22, "Humidity"), (0.74, "Light")])
    elif missing_profile == 3:
        add_missing_points([(0.16, "Temperature"), (0.48, "Humidity"), (0.82, "Light")])
    elif missing_profile == 4:
        add_missing_span(0.35, 3, ("Light",))
    elif missing_profile == 5:
        add_missing_span(0.42, 3, ("Temperature", "Humidity"))
    elif missing_profile == 6:
        add_missing_points([(0.30, "Fan"), (0.63, "Mist")])
    elif missing_profile == 7:
        add_missing_span(0.56, 4, ("Soil_Moisture",))
        add_missing_points([(0.78, "Light")])
    elif missing_profile == 9:
        add_missing_span(0.50, 2, ("Humidity",))
        add_missing_points([(0.84, "Temperature")])
    elif missing_profile == 10:
        add_missing_points([(0.46, "Soil_Moisture"), (0.70, "Drip")])
    elif missing_profile == 11:
        add_missing_span(0.24, 4, ("Temperature", "Humidity"))

    duplicate_profile = (first_ts.day * 3 + block_id) % 11
    duplicate_count = 2 if duplicate_profile == 7 else 1 if duplicate_profile in (0, 4, 9) else 0
    for dup_no in range(duplicate_count):
        duplicate_idx = spread_indices(1, 0.28 + 0.22 * dup_no, 0.28 + 0.22 * dup_no)[0]
        duplicate = out.iloc[[duplicate_idx]].copy()
        out = pd.concat([out.iloc[: duplicate_idx + 1], duplicate, out.iloc[duplicate_idx + 1 :]], ignore_index=True)
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


# Tách data chuẩn thành các file data đầu vào theo ngày và theo buổi.
def build_reference_data_files(output_dir: Path, data_path: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data = load_reference_data(data_path)
    paths: list[Path] = []
    windows = (
        ("00_midnight_raw.csv", "00_midnight", 0.0, 7.0),
        ("01_morning_raw.csv", "01_morning", 7.0, 9.0),
        ("02_late_morning_raw.csv", "02_late_morning", 9.0, 11.5),
        ("03_noon_raw.csv", "03_noon", 11.5, 13.5),
        ("04_early_afternoon_raw.csv", "04_early_afternoon", 13.5, 15.0),
        ("05_afternoon_raw.csv", "05_afternoon", 15.0, 17.0),
        ("06_evening_raw.csv", "06_evening", 17.0, 20.0),
        ("07_night_raw.csv", "07_night", 20.0, 22.0),
        ("08_late_night_raw.csv", "08_late_night", 22.0, 24.0),
    )

    for day, day_df in source_data.groupby(source_data["Timestamp"].dt.strftime("%Y-%m-%d"), sort=True):
        day_dir = output_dir / str(day)
        day_dir.mkdir(parents=True, exist_ok=True)
        timestamp = pd.to_datetime(day_df["Timestamp"])
        day_start = pd.Timestamp(day)

        for file_name, artifact_name, start_hour, end_hour in windows:
            start = day_start + pd.to_timedelta(start_hour, unit="h")
            end = day_start + pd.to_timedelta(end_hour, unit="h")
            session = day_df[(timestamp >= start) & (timestamp < end)].reset_index(drop=True)
            if session.empty:
                continue
            path = day_dir / file_name
            raw_session = inject_collection_artifacts(session, artifact_name)
            format_model_data(raw_session).to_csv(path, index=False)
            paths.append(path)
    return paths
