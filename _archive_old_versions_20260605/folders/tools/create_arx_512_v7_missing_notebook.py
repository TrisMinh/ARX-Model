from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell


SRC = Path("ARX_Model_Version 512/ARX_512_V6 copy.ipynb")
OUT = Path("ARX_Model_Version 512/ARX_512_V7.ipynb")


def replace_code_cell_source(nb, old: str, new: str) -> None:
    for cell in nb.cells:
        if cell.cell_type == "code" and old in str(cell.source):
            cell.source = str(cell.source).replace(old, new)
            return
    raise RuntimeError(f"Could not find code snippet: {old[:80]}")


def main() -> None:
    nb = nbformat.read(SRC, as_version=4)

    # Update notebook identity.
    first = str(nb.cells[0].source)
    first = first.replace("# ARX 512 Baseline V6", "# ARX 512 Baseline V7")
    first = first.replace(
        "ARX(5,1,2), 16 augmented inputs, Ridge alpha search, z-score normalization, intercept, free-run clip.",
        "ARX(5,1,2), 16 augmented inputs, missing-data handling, Ridge alpha search, z-score normalization, intercept, free-run clip.",
    )
    nb.cells[0].source = first

    replace_code_cell_source(nb, "VERSION_NUMBER = 6", "VERSION_NUMBER = 7")
    replace_code_cell_source(
        nb,
        'VERSION_NAME = "v6_512_augmented_zscore_intercept_ridge_clip"',
        'VERSION_NAME = "v7_512_missing_augmented_zscore_intercept_ridge_clip"',
    )

    # Replace the load-data cell with an equivalent one that includes missing-data handling.
    for cell in nb.cells:
        if cell.cell_type == "code" and "df_full, true_params, data_source = load_or_generate_data(DATA_CONFIG)" in str(cell.source):
            cell.source = r'''
DATA_CONFIG = DataConfig(
    csv_path=PROJECT_ROOT / "greenhouse_data.csv",
    generator_script_path=PROJECT_ROOT / "data_generator.py",
    force_regenerate_from_script=False,
    auto_save_generated_csv=True,
)

SPLIT_CONFIG = SplitConfig(train_ratio=0.75, val_ratio=0.15)

BASELINE_INPUT_COLS = (
    "Temperature",
    "Humidity",
    "Light",
    "Drip",
    "Mist",
    "Fan",
)

AUGMENTED_INPUT_COLS = (
    *BASELINE_INPUT_COLS,
    "Light_log",
    "Temp_x_Humi",
    "Temp_x_Light",
    "Humi_x_Light",
    "SP_Center",
    "SP_Width",
    "Month_sin",
    "Month_cos",
    "Season_sin",
    "Season_cos",
)

df_full_raw, true_params, data_source = load_or_generate_data(DATA_CONFIG)

NUMERIC_SENSOR_COLS = [
    "Soil_Moisture",
    "Temperature",
    "Humidity",
    "Light",
    "Soil_Low_SP",
    "Soil_High_SP",
]
BINARY_CONTROL_COLS = ["Drip", "Mist", "Fan"]
TIME_CONTEXT_COLS = ["Month"]
CATEGORICAL_COLS = ["Season"]
MISSING_CHECK_COLS = NUMERIC_SENSOR_COLS + BINARY_CONTROL_COLS + TIME_CONTEXT_COLS + CATEGORICAL_COLS


def missing_report(df_in: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for col in MISSING_CHECK_COLS:
        if col not in df_in.columns:
            continue
        n_missing = int(df_in[col].isna().sum())
        rows.append({
            "stage": label,
            "column": col,
            "missing_count": n_missing,
            "missing_pct": 100.0 * n_missing / max(len(df_in), 1),
        })
    return pd.DataFrame(rows)


def handle_missing_data(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)

    # Cảm biến liên tục: nội suy tuyến tính theo thứ tự thời gian.
    continuous_cols = [col for col in NUMERIC_SENSOR_COLS if col in df.columns]
    if continuous_cols:
        df[continuous_cols] = (
            df[continuous_cols]
            .astype(float)
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    # Tín hiệu điều khiển rời rạc: giữ trạng thái gần nhất rồi ép về 0/1.
    binary_cols = [col for col in BINARY_CONTROL_COLS if col in df.columns]
    for col in binary_cols:
        df[col] = df[col].ffill().bfill().fillna(0).astype(float).round().clip(0, 1)

    # Ngữ cảnh thời gian: điền gần nhất, sau đó ép kiểu phù hợp.
    if "Month" in df.columns:
        df["Month"] = df["Month"].ffill().bfill().fillna(1).astype(int)
    if "Season" in df.columns:
        df["Season"] = df["Season"].ffill().bfill().fillna("unknown")

    return df


missing_before = missing_report(df_full_raw, "before_handling")
df_full = handle_missing_data(df_full_raw)
missing_after = missing_report(df_full, "after_handling")
missing_summary = pd.concat([missing_before, missing_after], ignore_index=True)

df_train_raw, df_val_raw, df_test_raw = split_time_series(df_full, SPLIT_CONFIG)

overview = pd.DataFrame([
    summarize_dataset_behavior(df_full, "Full"),
    summarize_dataset_behavior(df_train_raw, "Train"),
    summarize_dataset_behavior(df_val_raw, "Validation"),
    summarize_dataset_behavior(df_test_raw, "Test"),
])

print("Nguồn dữ liệu:", data_source)
print("Số dòng:", len(df_full), "| Train:", len(df_train_raw), "| Validation:", len(df_val_raw), "| Test:", len(df_test_raw))
display(missing_summary)
overview
'''.strip() + "\n"
            break
    else:
        raise RuntimeError("Could not find load-data cell")

    # Insert a short explanatory markdown before the load-data section if not already present.
    insert_at = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and "## 2. Load data" in str(cell.source):
            insert_at = i + 1
            break
    if insert_at is None:
        raise RuntimeError("Could not find load data markdown")
    nb.cells.insert(
        insert_at,
        new_markdown_cell(
            """### 2.1 Kiểm tra và xử lý missing data

Dữ liệu hiện tại được sinh tự động nên kỳ vọng không có missing. Tuy nhiên, V7 bổ sung bước kiểm tra và xử lý missing để pipeline phù hợp hơn khi chuyển sang dữ liệu cảm biến thật. Biến cảm biến liên tục được nội suy tuyến tính; biến điều khiển 0/1 được điền theo trạng thái gần nhất và ép về 0/1."""
        ),
    )

    # Add missing handling metadata to artifact cell.
    for cell in nb.cells:
        if cell.cell_type == "code" and '"feature_engineering": {' in str(cell.source) and '"normalization": {' in str(cell.source):
            src = str(cell.source)
            src = src.replace(
                '"notes": "ARX(5,1,2), 16 augmented inputs, Ridge alpha search, z-score normalization, intercept, free-run clip.",',
                '"notes": "ARX(5,1,2), 16 augmented inputs, missing-data handling, Ridge alpha search, z-score normalization, intercept, free-run clip.",',
            )
            marker = '    "feature_engineering": {'
            insertion = '''    "missing_data_handling": {
        "enabled": True,
        "continuous_strategy": "linear interpolation + ffill/bfill",
        "binary_strategy": "ffill/bfill then round and clip to 0/1",
        "categorical_strategy": "ffill/bfill",
        "summary": missing_summary.to_dict(orient="records"),
    },
'''
            if '"missing_data_handling": {' not in src:
                src = src.replace(marker, insertion + marker)
            cell.source = src
            break

    OUT.parent.mkdir(exist_ok=True)
    nbformat.write(nb, OUT)
    print(f"Wrote {OUT.resolve()}")


if __name__ == "__main__":
    main()
