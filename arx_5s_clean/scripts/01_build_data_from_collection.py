from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from arx5s_clean.data.collection import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5s data from collection-style raw sessions.")
    parser.add_argument("--days", type=int, default=12, help="Number of augmented days to create.")
    parser.add_argument("--seed", type=int, default=505031, help="Random seed.")
    parser.add_argument(
        "--source",
        choices=("legacy", "real", "mau_cu", "thuc_te"),
        default="legacy",
        help="Data source: legacy sample data or real collected CSV.",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=None,
        help="Directory containing real CSV files. Relative paths are resolved from arx_5s_clean/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(days=args.days, seed=args.seed, source=args.source, real_dir=args.real_dir)


if __name__ == "__main__":
    main()
