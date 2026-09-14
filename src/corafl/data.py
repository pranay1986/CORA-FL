from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import sha256_arrays


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    task: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_outputs: int
    source: str
    split_seed: int

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def fingerprint(self) -> str:
        return sha256_arrays(
            [self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test]
        )


def _classification_dataset(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    if name == "digits":
        frame = load_digits()
        source = "scikit-learn bundled Digits dataset"
    elif name == "breast_cancer":
        frame = load_breast_cancer()
        source = "scikit-learn bundled Breast Cancer Wisconsin Diagnostic dataset"
    elif name == "wine":
        frame = load_wine()
        source = "scikit-learn bundled Wine recognition dataset"
    else:
        raise ValueError(f"Unsupported bundled dataset: {name}")
    return frame.data.astype(np.float64), frame.target.astype(np.int64), source


def load_bundled_dataset(name: str, split_seed: int = 20260910) -> DatasetBundle:
    x, y, source = _classification_dataset(name)
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=split_seed, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=2.0 / 3.0, random_state=split_seed + 1, stratify=y[temp_idx]
    )
    scaler = StandardScaler().fit(x[train_idx])
    return DatasetBundle(
        name=name,
        task="classification",
        x_train=scaler.transform(x[train_idx]).astype(np.float64),
        y_train=y[train_idx],
        x_val=scaler.transform(x[val_idx]).astype(np.float64),
        y_val=y[val_idx],
        x_test=scaler.transform(x[test_idx]).astype(np.float64),
        y_test=y[test_idx],
        n_outputs=int(np.max(y) + 1),
        source=source,
        split_seed=split_seed,
    )


def load_traffic_tensor(
    path: str | Path,
    name: str,
    history: int = 12,
    horizons: tuple[int, ...] = (3, 6, 12),
    split_seed: int = 20260910,
) -> tuple[DatasetBundle, dict[str, list[np.ndarray]]]:
    """Load an explicit traffic tensor; never download or synthesize one."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Traffic tensor not found: {input_path}. No substitute or synthetic data are generated."
        )
    with np.load(input_path, allow_pickle=False) as archive:
        if "data" not in archive:
            raise ValueError("Traffic NPZ must contain a 'data' array")
        values = np.asarray(archive["data"], dtype=np.float64)
    if values.ndim == 3:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError("Traffic data must have shape [time, sensors] or [time, sensors, features]")
    if not np.isfinite(values).all():
        raise ValueError("Traffic tensor contains NaN or infinite values; preprocessing must be explicit")

    total_time, n_sensors = values.shape
    train_end = int(total_time * 0.70)
    val_end = int(total_time * 0.80)
    mean = values[:train_end].mean(axis=0)
    std = values[:train_end].std(axis=0)
    std[std < 1e-12] = 1.0
    normalized = (values - mean) / std
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    times: list[int] = []
    sensors: list[int] = []
    max_horizon = max(horizons)
    for sensor in range(n_sensors):
        for time_idx in range(history, total_time - max_horizon):
            xs.append(normalized[time_idx - history : time_idx, sensor])
            ys.append(normalized[[time_idx + h - 1 for h in horizons], sensor])
            times.append(time_idx)
            sensors.append(sensor)
    x_all = np.asarray(xs, dtype=np.float64)
    y_all = np.asarray(ys, dtype=np.float64)
    times_arr = np.asarray(times)
    sensors_arr = np.asarray(sensors)
    masks = {
        "train": times_arr < train_end - max_horizon,
        "val": (times_arr >= train_end) & (times_arr < val_end - max_horizon),
        "test": times_arr >= val_end,
    }
    split_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    client_parts: dict[str, list[np.ndarray]] = {}
    for split, mask in masks.items():
        selected = np.flatnonzero(mask)
        split_arrays[split] = (x_all[selected], y_all[selected])
        local_sensors = sensors_arr[selected]
        client_parts[split] = [np.flatnonzero(local_sensors == sensor) for sensor in range(n_sensors)]
    bundle = DatasetBundle(
        name=name,
        task="regression",
        x_train=split_arrays["train"][0],
        y_train=split_arrays["train"][1],
        x_val=split_arrays["val"][0],
        y_val=split_arrays["val"][1],
        x_test=split_arrays["test"][0],
        y_test=split_arrays["test"][1],
        n_outputs=len(horizons),
        source=str(input_path.resolve()),
        split_seed=split_seed,
    )
    return bundle, client_parts
