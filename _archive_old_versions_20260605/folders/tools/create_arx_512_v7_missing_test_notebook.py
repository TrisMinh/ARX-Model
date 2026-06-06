from __future__ import annotations

from pathlib import Path

import nbformat


SRC = Path("ARX_Model_Version 512/ARX_512_V7.ipynb")
OUT = Path("ARX_Model_Version 512/ARX_512_V7_Missing_Test.ipynb")


def main() -> None:
    nb = nbformat.read(SRC, as_version=4)
    nb.cells[0].source = str(nb.cells[0].source).replace(
        "# ARX 512 Baseline V7",
        "# ARX 512 Baseline V7 - Missing Data Test",
    )
    nb.cells[0].source = str(nb.cells[0].source) + (
        "\n\nNotebook này dùng `greenhouse_data_missing.csv` để kiểm tra bước phát hiện và xử lý missing data."
    )

    for cell in nb.cells:
        src = str(cell.source)
        if 'VERSION_NAME = "v7_512_missing_augmented_zscore_intercept_ridge_clip"' in src:
            cell.source = src.replace(
                'VERSION_NAME = "v7_512_missing_augmented_zscore_intercept_ridge_clip"',
                'VERSION_NAME = "v7_512_missing_data_test"',
            )
        src = str(cell.source)
        if 'csv_path=PROJECT_ROOT / "greenhouse_data.csv"' in src:
            cell.source = src.replace(
                'csv_path=PROJECT_ROOT / "greenhouse_data.csv"',
                'csv_path=PROJECT_ROOT / "greenhouse_data_missing.csv"',
            )
        src = str(cell.source)
        if 'output_path = VERSION_DIR / f"arx_512_v{VERSION_NUMBER}.json"' in src:
            cell.source = src.replace(
                'output_path = VERSION_DIR / f"arx_512_v{VERSION_NUMBER}.json"',
                'output_path = VERSION_DIR / "arx_512_v7_missing_test.json"',
            )

    nbformat.write(nb, OUT)
    print(f"Wrote {OUT.resolve()}")


if __name__ == "__main__":
    main()
