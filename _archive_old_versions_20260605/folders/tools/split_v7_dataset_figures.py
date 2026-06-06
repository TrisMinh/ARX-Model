from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


NOTEBOOKS = [
    Path("ARX_Model_Version 512/ARX_512_V7.ipynb"),
    Path("ARX_Model_Version 512/ARX_512_V7_Missing_Test.ipynb"),
]


NEW_CELLS = [
    new_markdown_cell(
        """### 10.1 Dataset figures tách riêng

Các bảng dataset/missing được vẽ thành từng hình riêng để dễ chụp và đưa vào báo cáo. Mỗi hình cũng được lưu ra thư mục `figures_dataset/`."""
    ),
    new_code_cell(
        r'''
from pathlib import Path

FIG_DIR = VERSION_DIR / "figures_dataset"
FIG_DIR.mkdir(exist_ok=True)


def render_table_figure(table_df: pd.DataFrame, title: str, filename: str, figsize=(12, 4.5), fontsize: int = 9):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.35)
    ax.set_title(title, pad=14, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved:", out)
'''
    ),
    new_code_cell(
        r'''
stat_cols = ["Soil_Moisture", "Temperature", "Humidity", "Light", "Drip", "Mist", "Fan"]
collection_rows = []
for split_name, split_df in split_frames:
    collection_rows.append({
        "split": split_name,
        "start": str(split_df["Timestamp"].min()),
        "end": str(split_df["Timestamp"].max()),
        "samples": len(split_df),
        "missing_after_handling": int(split_df[stat_cols].isna().sum().sum()),
    })
collection_df = pd.DataFrame(collection_rows)

render_table_figure(
    collection_df,
    "Fig. 10a - Thời gian, số mẫu và missing sau xử lý theo split",
    "fig10a_split_samples_missing.png",
    figsize=(13, 3.6),
    fontsize=9,
)
collection_df
'''
    ),
    new_code_cell(
        r'''
stats_df = (
    df_full[stat_cols]
    .agg(["min", "max", "mean", "std"])
    .T
    .reset_index()
    .rename(columns={"index": "variable"})
    .round(3)
)

render_table_figure(
    stats_df,
    "Fig. 10b - Thống kê cơ bản các biến sau xử lý missing",
    "fig10b_basic_statistics.png",
    figsize=(10, 5.2),
    fontsize=9,
)
stats_df
'''
    ),
    new_code_cell(
        r'''
missing_before_df = (
    missing_summary[missing_summary["stage"].eq("before_handling")]
    [["column", "missing_count", "missing_pct"]]
    .copy()
)
missing_before_df["missing_pct"] = missing_before_df["missing_pct"].round(4)

render_table_figure(
    missing_before_df,
    "Fig. 10c - Missing data trước xử lý",
    "fig10c_missing_before.png",
    figsize=(9, 6.2),
    fontsize=9,
)
missing_before_df
'''
    ),
    new_code_cell(
        r'''
missing_after_df = (
    missing_summary[missing_summary["stage"].eq("after_handling")]
    [["column", "missing_count", "missing_pct"]]
    .copy()
)
missing_after_df["missing_pct"] = missing_after_df["missing_pct"].round(4)

render_table_figure(
    missing_after_df,
    "Fig. 10d - Missing data sau xử lý",
    "fig10d_missing_after.png",
    figsize=(9, 6.2),
    fontsize=9,
)
missing_after_df
'''
    ),
    new_code_cell(
        r'''
missing_compare_df = missing_before_df.merge(
    missing_after_df,
    on="column",
    suffixes=("_before", "_after"),
)

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(missing_compare_df))
width = 0.38
ax.bar(x - width / 2, missing_compare_df["missing_count_before"], width, label="Before", color="#e45756", alpha=0.85)
ax.bar(x + width / 2, missing_compare_df["missing_count_after"], width, label="After", color="#4c78a8", alpha=0.85)
ax.set_title("Fig. 10e - So sánh missing trước và sau xử lý")
ax.set_ylabel("Missing count")
ax.set_xticks(x)
ax.set_xticklabels(missing_compare_df["column"], rotation=35, ha="right")
ax.grid(True, axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
out = FIG_DIR / "fig10e_missing_before_after_bar.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.show()
print("Saved:", out)
missing_compare_df
'''
    ),
]


def patch_notebook(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    target = None
    for i, cell in enumerate(nb.cells):
        src = str(cell.source)
        if cell.cell_type == "code" and "fig, axes = plt.subplots(1, 3" in src and "Fig. 10a" in src:
            target = i
            break
    if target is None:
        raise RuntimeError(f"Could not find combined Fig.10 cell in {path}")
    nb.cells[target : target + 1] = NEW_CELLS
    nbformat.write(nb, path)
    print(f"Patched {path}")


def main() -> None:
    for notebook in NOTEBOOKS:
        patch_notebook(notebook)


if __name__ == "__main__":
    main()
