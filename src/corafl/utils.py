from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_arrays(arrays: list[np.ndarray]) -> str:
    """Hash ndarray metadata and exact C-order bytes."""
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_tree(root: Path, relative_paths: list[Path]) -> str:
    """Hash file names and contents in a deterministic order."""
    digest = hashlib.sha256()
    for relative in sorted(relative_paths, key=lambda item: item.as_posix()):
        path = root / relative
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def make_safe(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): make_safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [make_safe(val) for val in item]
        if isinstance(item, np.generic):
            return make_safe(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted-export"


def environment_record(repo_root: Path) -> dict[str, Any]:
    import matplotlib
    import pandas
    import scipy
    import sklearn
    import yaml

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pyyaml": yaml.__version__,
        "git_commit": git_commit(repo_root),
        "pid": os.getpid(),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
