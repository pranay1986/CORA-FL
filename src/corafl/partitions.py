from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DatasetBundle
from .utils import sha256_json, stable_seed


@dataclass(frozen=True)
class PartitionBundle:
    mode: str
    train: list[np.ndarray]
    val: list[np.ndarray]
    test: list[np.ndarray]
    class_histograms: np.ndarray
    domains: np.ndarray
    fingerprint: str


def _iid_indices(n_samples: int, n_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.sort(chunk.astype(np.int64)) for chunk in np.array_split(rng.permutation(n_samples), n_clients)]


def _allocate_by_class(y: np.ndarray, class_probabilities: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_clients = class_probabilities.shape[1]
    parts: list[list[int]] = [[] for _ in range(n_clients)]
    for class_id in range(class_probabilities.shape[0]):
        members = rng.permutation(np.flatnonzero(y == class_id))
        probabilities = class_probabilities[class_id]
        counts = rng.multinomial(len(members), probabilities / probabilities.sum())
        cursor = 0
        for client_id, count in enumerate(counts):
            parts[client_id].extend(members[cursor : cursor + count].tolist())
            cursor += count
    return [np.asarray(sorted(indices), dtype=np.int64) for indices in parts]


def _dirichlet_partitions(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, n_clients: int, alpha: float, min_train_samples: int, seed: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_classes = int(max(np.max(y_train), np.max(y_val), np.max(y_test)) + 1)
    for attempt in range(500):
        rng = np.random.default_rng(stable_seed(seed, "dirichlet", attempt))
        probabilities = rng.dirichlet(np.full(n_clients, alpha), size=n_classes)
        train = _allocate_by_class(y_train, probabilities, stable_seed(seed, "train", attempt))
        if min(len(indices) for indices in train) < min_train_samples:
            continue
        val = _allocate_by_class(y_val, probabilities, stable_seed(seed, "val", attempt))
        test = _allocate_by_class(y_test, probabilities, stable_seed(seed, "test", attempt))
        return train, val, test
    raise RuntimeError("Unable to construct a non-IID partition with the requested minimum client size")


def _histograms(y: np.ndarray, parts: list[np.ndarray], n_classes: int) -> np.ndarray:
    result = np.zeros((len(parts), n_classes), dtype=np.float64)
    for client_id, indices in enumerate(parts):
        counts = np.bincount(y[indices], minlength=n_classes).astype(np.float64)
        result[client_id] = counts / max(float(counts.sum()), 1.0)
    return result


def _correlate_shards_by_domain(
    train: list[np.ndarray],
    val: list[np.ndarray],
    test: list[np.ndarray],
    histograms: np.ndarray,
    n_domains: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Place statistically similar shards in the same two-client domain.

    The mapping uses train-only histograms. It models a regional failure domain
    whose colocated clients observe related, rather than randomly interleaved,
    distributions. The current certificate requires two clients per domain.
    """
    n_clients = len(train)
    if n_clients != 2 * n_domains:
        raise ValueError("Domain-correlated partition currently requires two clients per domain")
    candidates: list[tuple[float, int, int]] = []
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            candidates.append((float(np.abs(histograms[i] - histograms[j]).sum()), i, j))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used: set[int] = set()
    groups: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i not in used and j not in used:
            used.update((i, j))
            groups.append((i, j))
    if len(groups) != n_domains:
        raise RuntimeError("Unable to construct domain-correlated shard groups")
    groups.sort()
    order = [groups[d][0] for d in range(n_domains)] + [groups[d][1] for d in range(n_domains)]
    return (
        [train[i] for i in order],
        [val[i] for i in order],
        [test[i] for i in order],
        histograms[np.asarray(order, dtype=np.int64)],
    )


def make_partitions(dataset: DatasetBundle, mode: str, n_clients: int, alpha: float, min_train_samples: int, seed: int, n_domains: int) -> PartitionBundle:
    if dataset.task != "classification":
        raise ValueError("IID/non-IID class partitioning is defined only for classification datasets")
    if mode == "iid":
        train = _iid_indices(len(dataset.y_train), n_clients, stable_seed(seed, dataset.name, "train"))
        val = _iid_indices(len(dataset.y_val), n_clients, stable_seed(seed, dataset.name, "val"))
        test = _iid_indices(len(dataset.y_test), n_clients, stable_seed(seed, dataset.name, "test"))
    elif mode in {"noniid", "noniid_domain"}:
        train, val, test = _dirichlet_partitions(dataset.y_train, dataset.y_val, dataset.y_test, n_clients, alpha, min_train_samples, stable_seed(seed, dataset.name, "noniid"))
    else:
        raise ValueError(f"Unknown partition mode: {mode}")
    hist = _histograms(dataset.y_train, train, dataset.n_outputs)
    if mode == "noniid_domain":
        train, val, test, hist = _correlate_shards_by_domain(train, val, test, hist, n_domains)
    domains = np.arange(n_clients, dtype=np.int64) % n_domains
    fingerprint = sha256_json({"dataset": dataset.name, "mode": mode, "seed": seed, "train": [x.tolist() for x in train], "val": [x.tolist() for x in val], "test": [x.tolist() for x in test], "domains": domains.tolist()})
    return PartitionBundle(mode, train, val, test, hist, domains, fingerprint)


def traffic_partitions(split_parts: dict[str, list[np.ndarray]], n_domains: int) -> PartitionBundle:
    n_clients = len(split_parts["train"])
    if n_clients % n_domains:
        raise ValueError("Number of traffic sensors must be divisible by n_domains")
    domains = np.arange(n_clients, dtype=np.int64) % n_domains
    fingerprint = sha256_json({"train": [x.tolist() for x in split_parts["train"]], "val": [x.tolist() for x in split_parts["val"]], "test": [x.tolist() for x in split_parts["test"]], "domains": domains.tolist()})
    return PartitionBundle("natural_sensor", split_parts["train"], split_parts["val"], split_parts["test"], np.ones((n_clients, 1)), domains, fingerprint)
