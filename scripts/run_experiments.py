#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corafl.runner import run_all, run_main, run_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CORA-FL validation pilot or locked experiment matrix")
    parser.add_argument("--config", default="configs/confirmatory.yaml")
    parser.add_argument("--stage", choices=["pilot", "main", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    if args.stage == "pilot":
        output = run_pilot(config_path)
    elif args.stage == "main":
        output = run_main(config_path, force=args.force)
    else:
        output = run_all(config_path, force=args.force)
    print(output)


if __name__ == "__main__":
    main()
