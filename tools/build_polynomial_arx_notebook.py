from __future__ import annotations

import json
from pathlib import Path


def md_cell(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in lines],
    }


def code_cell(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else f"{line}\n" for line in lines],
    }


def build_notebook() -> dict:
    cells = [
        md_cell(
            "# Polynomial ARX Notebook (Synthetic Nonlinear Benchmark)",
            "",
            "Mục tiêu:",
            "- Tạo dữ liệu nhà kính có phi tuyến bằng file mới `data_generator_poly_arx.py`.",
            "- So sánh ARX tuyến tính và Polynomial ARX trên cùng bộ dữ liệu phi tuyến này.",
            "- Chứng minh bằng 2 chỉ số chính: `FIT_12step` và `FIT_free_run`.",
            "- Đây là benchmark riêng, không phải thay thế cho kết quả trên notebook ARX cũ.",
        ),
        code_cell(
            "import numpy as np",
            "import pandas as pd",
            "",
            "from arx_pipeline import (",
            "    SplitConfig,",
            "    ModelConfig,",
            "    split_time_series,",
            "    build_regression_matrix,",
            "    estimate_ols,",
            "    evaluate_slice,",
            ")",
            "from narx_pipeline import (",
            "    NARXModelConfig,",
            "    build_narx_regression_matrix,",
            "    evaluate_narx_slice,",
            ")",
            "from data_generator_poly_arx import generate_greenhouse_data_poly",
        ),
        md_cell("## 1) Sinh dữ liệu phi tuyến (file mới, không đụng dữ liệu cũ)"),
        code_cell(
            "df_raw, true_params = generate_greenhouse_data_poly(days=365, sampling_seconds=300, seed=42)",
            "df_raw.to_csv('greenhouse_data_poly.csv', index=False)",
            "print(f'Rows: {len(df_raw)}')",
            "print('Saved: greenhouse_data_poly.csv')",
            "print('True nonlinear params:', {k: true_params[k] for k in ['c_temp2', 'c_humi2', 'c_temp_humi']})",
            "df_raw.head()",
        ),
        md_cell("## 2) Min-Max normalization (đúng quy trình Polynomial ARX)"),
        code_cell(
            "df = df_raw.copy()",
            "cols = ['Soil_Moisture', 'Temperature', 'Humidity', 'Light', 'Drip', 'Mist', 'Fan']",
            "minmax = {}",
            "for c in cols:",
            "    cmin = float(df[c].min())",
            "    cmax = float(df[c].max())",
            "    minmax[c] = (cmin, cmax)",
            "    denom = cmax - cmin if (cmax - cmin) != 0 else 1.0",
            "    df[c] = (df[c] - cmin) / denom",
            "",
            "split_cfg = SplitConfig(train_ratio=0.60, val_ratio=0.20)",
            "df_train, df_val, df_test = split_time_series(df, split_cfg)",
            "print('Train/Val/Test:', len(df_train), len(df_val), len(df_test))",
        ),
        md_cell("## 3) ARX baseline"),
        code_cell(
            "arx_cfg = ModelConfig(",
            "    na=2,",
            "    nb=2,",
            "    nk=1,",
            "    include_intercept=False,",
            "    simulation_clip=(0.0, 1.0),",
            ")",
            "X_arx, y_arx = build_regression_matrix(df_train, arx_cfg)",
            "theta_arx, _, _ = estimate_ols(X_arx, y_arx)",
            "val_arx = evaluate_slice('Validation', df_val, theta_arx, arx_cfg, n_step=12)",
            "test_arx = evaluate_slice('Test', df_test, theta_arx, arx_cfg, n_step=12)",
        ),
        md_cell("## 4) Polynomial ARX (bậc 2)", "Dùng thêm bình phương và tích chéo thông qua ma trận hồi quy polynomial."),
        code_cell(
            "poly_cfg = NARXModelConfig(",
            "    na=2,",
            "    nb=2,",
            "    nk=1,",
            "    poly_degree=2,",
            "    include_intercept=True,",
            "    cross_term_mode='all',",
            "    simulation_clip=(0.0, 1.0),",
            ")",
            "X_poly, y_poly = build_narx_regression_matrix(df_train, poly_cfg)",
            "theta_poly, _, _ = estimate_ols(X_poly, y_poly)",
            "val_poly = evaluate_narx_slice('Validation', df_val, theta_poly, poly_cfg, n_step=12, arx_theta=theta_arx)",
            "test_poly = evaluate_narx_slice('Test', df_test, theta_poly, poly_cfg, n_step=12, arx_theta=theta_arx)",
        ),
        md_cell("## 5) So sánh trọng tâm trên benchmark phi tuyến: FIT_12step và FIT_free_run"),
        code_cell(
            "compare = pd.DataFrame([",
            "    {",
            "        'Split': 'Validation',",
            "        'ARX_FIT_12step': val_arx['metrics_n_step']['FIT'],",
            "        'PolyARX_FIT_12step': val_poly['metrics_n_step']['FIT'],",
            "        'ARX_FIT_free_run': val_arx['metrics_sim']['FIT'],",
            "        'PolyARX_FIT_free_run': val_poly['metrics_sim']['FIT'],",
            "    },",
            "    {",
            "        'Split': 'Test',",
            "        'ARX_FIT_12step': test_arx['metrics_n_step']['FIT'],",
            "        'PolyARX_FIT_12step': test_poly['metrics_n_step']['FIT'],",
            "        'ARX_FIT_free_run': test_arx['metrics_sim']['FIT'],",
            "        'PolyARX_FIT_free_run': test_poly['metrics_sim']['FIT'],",
            "    },",
            "])",
            "compare['Delta_12step'] = compare['PolyARX_FIT_12step'] - compare['ARX_FIT_12step']",
            "compare['Delta_free_run'] = compare['PolyARX_FIT_free_run'] - compare['ARX_FIT_free_run']",
            "display(compare.round(4))",
            "",
            "val_ok = (compare.loc[compare['Split']=='Validation', 'Delta_12step'].iloc[0] > 0) and (compare.loc[compare['Split']=='Validation', 'Delta_free_run'].iloc[0] > 0)",
            "test_ok = (compare.loc[compare['Split']=='Test', 'Delta_12step'].iloc[0] > 0) and (compare.loc[compare['Split']=='Test', 'Delta_free_run'].iloc[0] > 0)",
            "",
            "print('Validation improved on both metrics:', val_ok)",
            "print('Test improved on both metrics:', test_ok)",
            "if val_ok and test_ok:",
            "    print('KET LUAN: Tren benchmark phi tuyen nay, Polynomial ARX tot hon ARX tren ca FIT_12step va FIT_free_run.')",
            "else:",
            "    print('KET LUAN: Tren benchmark phi tuyen nay, Polynomial ARX chua vuot ARX dong thoi ca 2 chi so.')",
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "Polynomial_ARX_Model_Notebook.ipynb"
    notebook = build_notebook()
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created notebook: {notebook_path}")


if __name__ == "__main__":
    main()
