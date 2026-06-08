from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from arx5s_clean.config import ExperimentConfig  # noqa: E402
from arx5s_clean.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARX on data/mini_greenhouse_5s_data.csv.")
    parser.add_argument("--days", type=int, default=12, help="Number of days recorded in the built data.")
    parser.add_argument("--seed", type=int, default=505031, help="Simulation seed.")
    parser.add_argument("--grid", choices=("tiny", "quick", "wide"), default="quick", help="ARX search grid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(days=args.days, seed=args.seed)
    payload = run_pipeline(PROJECT_ROOT, cfg, args.grid)
    model = payload["model"]
    train = payload["train"]
    validation = payload["validation"]
    test = payload["test"]

    print(f"Saved data to {PROJECT_ROOT / 'data'}")
    print(f"Saved results to {PROJECT_ROOT / 'results'}")
    print(
        "ARX 5s clean: "
        f"{model['name']} "
        f"TRAIN_5min={train['metrics_5min_chunked']['FIT']:.3f}, "
        f"VAL_5min={validation['metrics_5min_chunked']['FIT']:.3f}, "
        f"TEST_5min={test['metrics_5min_chunked']['FIT']:.3f}, "
        f"VAL_free_run={validation['metrics_free_run']['FIT']:.3f}, "
        f"TEST_free_run={test['metrics_free_run']['FIT']:.3f}"
    )


if __name__ == "__main__":
    main()
