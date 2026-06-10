from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Các cột bắt buộc trong mỗi file dữ liệu thu thập.
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
# Ngưỡng hợp lệ dùng để phát hiện outlier sensor trong bước clean.
SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "Temperature": (15.0, 45.0),
    "Humidity": (30.0, 100.0),
    "Light": (0.0, 1200.0),
    "Soil_Moisture": (0.0, 100.0),
}

ACTUATOR_COLS: tuple[str, ...] = ("Drip", "Mist", "Fan")


# Chuẩn hóa format cột: sensor dạng số làm tròn, actuator về 0/1.
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
    return Path(__file__).resolve().parents[2]


# Lấy thư mục chứa các file raw đầu vào.
def input_data_dir(input_dir: Path | None = None) -> Path:
    if input_dir is None:
        return project_root() / "data" / "_01_data"
    return input_dir if input_dir.is_absolute() else project_root() / input_dir


# Lấy danh sách CSV raw, bỏ qua file raw tổng hợp nếu đã có.
def input_csv_paths(input_dir: Path | None = None) -> list[Path]:
    root = input_data_dir(input_dir)
    return sorted(path for path in root.rglob("*.csv") if path.name != "00_raw_tong_hop.csv")


# Kiểm tra đủ cột bắt buộc, bỏ cột thừa và sắp xếp lại đúng thứ tự.
def normalize_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in MODEL_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")
    return df.loc[:, MODEL_COLS].copy()


# Đọc một CSV và kiểm tra đúng format cột.
def load_model_csv(path: Path) -> pd.DataFrame:
    return normalize_model_columns(pd.read_csv(path))


# Đọc file dữ liệu tổng 5 giây khi cần tách lại theo ngày/phiên.
def load_reference_data(data_path: Path) -> pd.DataFrame:
    df = load_model_csv(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True).loc[:, MODEL_COLS]


# Chuẩn hóa tên file raw khi nhập các CSV rời từ thư mục nguồn.
def data_output_name(index: int, source_stem: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in source_stem)
    if safe.endswith("_raw"):
        return f"{index:02d}_{safe}.csv"
    return f"{index:02d}_{safe}_raw.csv"


# Chuẩn hóa các CSV rời và ghi vào thư mục raw đầu vào của pipeline.
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


# Đọc toàn bộ CSV nguồn rồi ghép thành một bảng theo thời gian.
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


# Tùy chọn: tách file tổng 5 giây thành raw theo ngày và phiên thu.
def build_reference_data_files(output_dir: Path, data_path: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data = load_reference_data(data_path)
    paths: list[Path] = []
    windows = (
        ("00_midnight_raw.csv", 0.0, 7.0),
        ("01_morning_raw.csv", 7.0, 9.0),
        ("02_late_morning_raw.csv", 9.0, 11.5),
        ("03_noon_raw.csv", 11.5, 13.5),
        ("04_early_afternoon_raw.csv", 13.5, 15.0),
        ("05_afternoon_raw.csv", 15.0, 17.0),
        ("06_evening_raw.csv", 17.0, 20.0),
        ("07_night_raw.csv", 20.0, 22.0),
        ("08_late_night_raw.csv", 22.0, 24.0),
    )

    for day, day_df in source_data.groupby(source_data["Timestamp"].dt.strftime("%Y-%m-%d"), sort=True):
        day_dir = output_dir / str(day)
        day_dir.mkdir(parents=True, exist_ok=True)
        timestamp = pd.to_datetime(day_df["Timestamp"])
        day_start = pd.Timestamp(day)

        for file_name, start_hour, end_hour in windows:
            start = day_start + pd.to_timedelta(start_hour, unit="h")
            end = day_start + pd.to_timedelta(end_hour, unit="h")
            session = day_df[(timestamp >= start) & (timestamp < end)].reset_index(drop=True)
            if session.empty:
                continue
            path = day_dir / file_name
            format_model_data(session).to_csv(path, index=False)
            paths.append(path)
    return paths
