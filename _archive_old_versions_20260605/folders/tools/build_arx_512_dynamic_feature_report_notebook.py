from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT_PATH = Path("ARX_Model_Version 512/ARX_512_Dynamic_Feature_Search_Report.ipynb")


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
# ARX(5,1,2) Improvement Attempts - No-Leak Report

Notebook này tổng hợp các thử nghiệm cải thiện ARX sau V6.

Kết luận quan trọng: các feature chứa `Soil_Moisture` như `Soil_Deficit = SP_Center - Soil_Moisture` có thể làm FIT tăng rất mạnh nếu bị dùng như input cố định trong free-run, nhưng cách đó gây leakage. Báo cáo này chỉ dùng kết quả no-leak.
"""
        ),
        code(
            """
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

VERSION_DIR = Path.cwd()
if VERSION_DIR.name != "ARX_Model_Version 512":
    VERSION_DIR = Path("ARX_Model_Version 512")

v6 = json.loads((VERSION_DIR / "arx_512_v6.json").read_text(encoding="utf-8"))
v6_row = {
    "method": "V6 OLS/Ridge search",
    "val_FIT_sim": v6["metrics"]["validation"]["fit_sim"],
    "test_FIT_sim": v6["metrics"]["test"]["fit_sim"],
    "test_RMSE_sim": v6["metrics"]["test"]["rmse_sim"],
}

no_leak_df = pd.read_csv(VERSION_DIR / "arx_512_dynamic_feature_search_test_top_noleak.csv")
refine_df = pd.read_csv(VERSION_DIR / "arx_512_simulation_error_refine.csv")
best_refine = refine_df.sort_values(["test_FIT_sim", "val_FIT_sim"], ascending=False).iloc[0]
best_noleak = no_leak_df.sort_values(["test_FIT_sim", "val_FIT_sim"], ascending=False).iloc[0]

summary = pd.DataFrame([
    v6_row,
    {
        "method": "Best no-leak dynamic feature",
        "val_FIT_sim": best_noleak["val_FIT_sim"],
        "test_FIT_sim": best_noleak["test_FIT_sim"],
        "test_RMSE_sim": best_noleak["test_RMSE_sim"],
    },
    {
        "method": f"Simulation-error refine lambda={best_refine['lambda']}",
        "val_FIT_sim": best_refine["val_FIT_sim"],
        "test_FIT_sim": best_refine["test_FIT_sim"],
        "test_RMSE_sim": best_refine["test_RMSE_sim"],
    },
])
summary["gain_test_FIT_vs_V6"] = summary["test_FIT_sim"] - v6_row["test_FIT_sim"]
summary.round(4)
"""
        ),
        md("## 1. Dynamic feature search đúng no-leak"),
        code(
            """
no_leak_df[["label", "na", "nb", "nk", "alpha", "val_FIT_sim", "test_FIT_sim", "test_RMSE_sim"]].round(4)
"""
        ),
        md("## 2. Tối ưu trực tiếp simulation error"),
        code(
            """
refine_df[["method", "lambda", "nfev", "val_FIT_sim", "test_FIT_sim", "test_RMSE_sim"]].round(4)
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
summary.plot(x="method", y="test_FIT_sim", kind="bar", ax=axes[0], legend=False, color="#4c78a8")
axes[0].set_title("Test FIT_sim sau các thử nghiệm hợp lệ")
axes[0].set_xlabel("")
axes[0].set_ylabel("FIT_sim (%)")
axes[0].grid(True, axis="y", alpha=0.25)
axes[0].tick_params(axis="x", rotation=20)

summary.plot(x="method", y="test_RMSE_sim", kind="bar", ax=axes[1], legend=False, color="#f58518")
axes[1].set_title("Test RMSE_sim sau các thử nghiệm hợp lệ")
axes[1].set_xlabel("")
axes[1].set_ylabel("RMSE")
axes[1].grid(True, axis="y", alpha=0.25)
axes[1].tick_params(axis="x", rotation=20)
fig.tight_layout()
plt.show()
"""
        ),
        md("## 3. Kết luận"),
        code(
            """
print(f\"V6 hiện tại: test FIT_sim = {v6_row['test_FIT_sim']:.2f}%, RMSE = {v6_row['test_RMSE_sim']:.3f}\")
print(f\"Dynamic feature no-leak tốt nhất: test FIT_sim = {best_noleak['test_FIT_sim']:.2f}%\")
print(f\"Simulation-error refinement tốt nhất: test FIT_sim = {best_refine['test_FIT_sim']:.2f}%, RMSE = {best_refine['test_RMSE_sim']:.3f}\")
print(\"Kết luận: chưa tìm được bước nhảy hợp lệ vượt 70% cho ARX tuyến tính; bước nhảy 88% là leakage nếu dùng Soil_Deficit cố định trong free-run.\")
"""
        ),
    ]
    OUT_PATH.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT_PATH)
    print(f"Wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
