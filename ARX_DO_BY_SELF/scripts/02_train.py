from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from config import ExperimentConfig  # noqa: E402
from pipeline import run_pipeline  # noqa: E402


# Đọc tham số train từ terminal.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARX on data/mini_greenhouse_5s_data.csv.")
    parser.add_argument("--grid", choices=("tiny", "quick", "wide"), default="quick")
    return parser.parse_args()


# Chạy train model.
def main() -> None:
    args = parse_args()
    payload = run_pipeline(PROJECT_ROOT, ExperimentConfig(), args.grid)
    model = payload["model"]
    validation = payload["validation"]
    test = payload["test"]
    print(
        f"{model['name']} "
        f"VAL_free_run={validation['metrics_free_run']['FIT']:.3f} "
        f"TEST_free_run={test['metrics_free_run']['FIT']:.3f}"
    )


if __name__ == "__main__":
    main()
