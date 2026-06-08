from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from data.collection import run  # noqa: E402


# Đọc tham số build data từ terminal.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5s training data from raw CSV.")
    parser.add_argument("--days", type=int, default=12)
    parser.add_argument("--seed", type=int, default=505031)
    parser.add_argument("--raw-dir", type=Path, default=None)
    return parser.parse_args()


# Chạy build data.
def main() -> None:
    args = parse_args()
    run(days=args.days, seed=args.seed, source_dir=args.raw_dir)


if __name__ == "__main__":
    main()
