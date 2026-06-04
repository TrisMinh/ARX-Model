from __future__ import annotations

from pathlib import Path

import nbformat


NB_PATH = Path("ARX_Model_Version 512/ARX_512_V7.ipynb")


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    fixed_cells = []
    removed = 0
    for cell in nb.cells:
        src = str(cell.source)
        if cell.cell_type == "code" and 'VERSION_NAME = "v6_512_augmented_zscore_intercept_ridge_clip"' in src:
            removed += 1
            continue
        fixed_cells.append(cell)
    nb.cells = fixed_cells
    nbformat.write(nb, NB_PATH)
    print(f"Removed {removed} stale V6 config cell(s) from {NB_PATH}")


if __name__ == "__main__":
    main()
