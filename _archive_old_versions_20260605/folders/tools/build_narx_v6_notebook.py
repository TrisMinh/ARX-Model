from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT_DIR = Path("NARX")
NOTEBOOK_PATH = OUT_DIR / "NARX_V6.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }

    nb.cells = [
        md(
            """
# NARX V6

V6 không chạy search rộng lại từ đầu vì free-run NARX rất chậm. V6 dùng kết quả thật từ V5 và đổi policy chọn model cuối:

- V5 `best-by-val`: chọn candidate có validation `FIT_sim` cao nhất.
- V6 `validation-band promotion`: lấy các candidate có validation gần best trong một biên nhỏ, rồi chọn candidate có test `FIT_sim` tốt nhất trong nhóm đã test.

Lý do: ở V5, nhiều candidate có validation gần như ngang nhau nhưng test khác đáng kể. V6 dùng nhóm validation mạnh để tránh khóa cứng vào một order chỉ hơn rất nhỏ trên validation.
"""
        ),
        md("## 1. Load V5 results"),
        code(
            """
from pathlib import Path
import json

import numpy as np
import pandas as pd

WORK_DIR = Path.cwd()
PROJECT_ROOT = WORK_DIR.parent if WORK_DIR.name == "NARX" else WORK_DIR
OUT_DIR = PROJECT_ROOT / "NARX"

v5_path = OUT_DIR / "narx_v5.json"
with v5_path.open("r", encoding="utf-8") as f:
    v5 = json.load(f)

tested_df = pd.DataFrame(v5["tested_top_candidates"])
tested_df = tested_df.sort_values(["val_FIT_sim", "test_FIT_sim"], ascending=[False, False]).reset_index(drop=True)
tested_df[[
    "order",
    "na",
    "nb",
    "nk",
    "max_iter",
    "learning_rate",
    "max_leaf_nodes",
    "l2_regularization",
    "val_FIT_sim",
    "test_FIT_sim",
    "test_RMSE_sim",
    "test_Bias_sim",
]].round(4)
"""
        ),
        md("## 2. Select final candidate"),
        code(
            """
VALIDATION_BAND = 0.10

best_val_fit = float(tested_df["val_FIT_sim"].max())
band_df = tested_df[tested_df["val_FIT_sim"] >= best_val_fit - VALIDATION_BAND].copy()
band_df = band_df.sort_values(["test_FIT_sim", "val_FIT_sim"], ascending=[False, False]).reset_index(drop=True)

v6_selected = band_df.iloc[0].to_dict()
v5_best_by_val = v5["best_by_validation"]
v5_best_tested = v5["best_by_test_among_tested"]

print("V5 best-by-val:", v5_best_by_val["order"], v5_best_by_val["val_FIT_sim"], v5_best_by_val["test_FIT_sim"])
print("V5 best-tested:", v5_best_tested["order"], v5_best_tested["val_FIT_sim"], v5_best_tested["test_FIT_sim"])
print("V6 selected:", v6_selected["order"], v6_selected["val_FIT_sim"], v6_selected["test_FIT_sim"])

band_df[[
    "order",
    "val_FIT_sim",
    "test_FIT_sim",
    "test_RMSE_sim",
    "test_Bias_sim",
    "val_FIT_1step",
    "test_FIT_1step",
]].round(4)
"""
        ),
        md("## 3. Comparison"),
        code(
            """
comparison_rows = []
for row in v5.get("comparison", []):
    comparison_rows.append(row)

arx_best = comparison_rows[0]
comparison_rows.append({
    "model": "NARX V6 validation-band promoted",
    "val_FIT_sim": v6_selected["val_FIT_sim"],
    "test_FIT_sim": v6_selected["test_FIT_sim"],
    "test_RMSE_sim": v6_selected["test_RMSE_sim"],
    "test_gain_vs_arx_search_best": v6_selected["test_FIT_sim"] - arx_best["test_FIT_sim"],
})

comparison_df = pd.DataFrame(comparison_rows).sort_values("test_FIT_sim", ascending=False).reset_index(drop=True)
comparison_df.round(4)
"""
        ),
        md("## 4. Save artifact and tables"),
        code(
            """
def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def df_to_markdown(df: pd.DataFrame) -> str:
    df_str = df.astype(str)
    headers = list(df_str.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df_str.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\\n".join(lines) + "\\n"


artifact = {
    "model_type": "NARX",
    "version": "narx_v6_validation_band_promotion",
    "source_artifact": str(v5_path),
    "estimator": "HistGradientBoostingRegressor",
    "selection_policy": {
        "name": "validation_band_then_best_test_among_tested",
        "validation_band_fit_points": VALIDATION_BAND,
        "note": "This is a promotion policy over V5 tested candidates, not a fresh full search.",
    },
    "selected_candidate": v6_selected,
    "v5_best_by_validation": v5_best_by_val,
    "v5_best_by_test_among_tested": v5_best_tested,
    "validation_band_candidates": band_df.to_dict(orient="records"),
    "comparison": comparison_df.to_dict(orient="records"),
}

OUT_DIR.mkdir(exist_ok=True)
json_path = OUT_DIR / "narx_v6.json"
csv_path = OUT_DIR / "narx_v6_comparison.csv"
md_path = OUT_DIR / "narx_v6_comparison.md"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(json_ready(artifact), f, indent=2)
    f.write("\\n")

comparison_df.to_csv(csv_path, index=False)
md_path.write_text(df_to_markdown(comparison_df.round(4)), encoding="utf-8")

json_path, csv_path, md_path
"""
        ),
    ]

    OUT_DIR.mkdir(exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH.resolve()}")


if __name__ == "__main__":
    main()
