#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = {"", "REQUIRED", "TBD", "TODO"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate genuine traffic inputs")
    parser.add_argument("--config", default="configs/publication_traffic.yaml")
    args = parser.parse_args()
    config_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for entry in config["datasets"]:
        name = str(entry["name"])
        path = Path(entry["path"])
        path = path if path.is_absolute() else ROOT / path
        checksum = str(entry.get("checksum_sha256", "REQUIRED"))
        source_url = str(entry.get("source_url", "REQUIRED"))
        license_name = str(entry.get("license", "REQUIRED"))
        if checksum.upper() in PLACEHOLDERS:
            failures.append(f"{name}: checksum_sha256 is not locked")
        if source_url.upper() in PLACEHOLDERS:
            failures.append(f"{name}: source_url is not recorded")
        if license_name.upper() in PLACEHOLDERS:
            failures.append(f"{name}: license is not recorded")
        if not path.is_file():
            failures.append(f"{name}: missing file {path}")
            continue
        if checksum.upper() not in PLACEHOLDERS and sha256(path) != checksum.lower():
            failures.append(f"{name}: SHA-256 mismatch")
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "data" not in archive:
                    raise ValueError("NPZ has no 'data' array")
                data = archive["data"]
                if data.ndim not in (2, 3):
                    raise ValueError(f"expected 2-D/3-D data, received {data.ndim}-D")
                expected = entry.get("expected_sensors")
                if expected is not None and data.shape[1] != int(expected):
                    raise ValueError(f"expected {expected} sensors, received {data.shape[1]}")
                shape = tuple(int(value) for value in data.shape)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        print(f"PASS {name}: {path.name}, shape={shape}, sha256={sha256(path)}")
    if failures:
        print("Traffic publication stage remains blocked:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        print("No synthetic or substitute tensor was generated.", file=sys.stderr)
        raise SystemExit(2)
    print("All traffic inputs are present and checksum-locked.")


if __name__ == "__main__":
    main()
