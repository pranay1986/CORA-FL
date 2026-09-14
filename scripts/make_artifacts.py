#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corafl.analysis import generate_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all figures and tables from raw logs")
    parser.add_argument("--config", default="configs/development.yaml")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    print(json.dumps(generate_artifacts(config_path), indent=2))


if __name__ == "__main__":
    main()
