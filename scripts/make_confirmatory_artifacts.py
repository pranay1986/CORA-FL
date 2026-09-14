#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corafl.confirmatory_analysis import generate_confirmatory_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate locked CORA-FL confirmatory figures, tables, and statistics")
    parser.add_argument("--config", default="configs/confirmatory.yaml")
    args = parser.parse_args()
    config_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    print(json.dumps(generate_confirmatory_artifacts(config_path), indent=2))


if __name__ == "__main__":
    main()
