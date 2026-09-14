from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import sha256_json, stable_seed


@dataclass(frozen=True)
class NetworkTrace:
    active: np.ndarray
    link_up: np.ndarray
    scenario: str
    seed: int
    outage_domain: int
    outage_start: int
    outage_end: int
    link_loss: float
    degraded_link_loss: float
    degraded_domain_pair: tuple[int, int]
    checksum: str
    path: str


def _trace_checksum(active: np.ndarray, link_up: np.ndarray, metadata: dict[str, object]) -> str:
    return sha256_json({**metadata, "active_hex": active.astype(np.uint8).tobytes().hex(), "link_hex": link_up.astype(np.uint8).tobytes().hex()})


def create_trace(path: Path, n_clients: int, n_domains: int, rounds: int, scenario: str, seed: int, link_loss: float, outage_start_fraction: float, outage_end_fraction: float, degraded_link_loss: float | None = None) -> NetworkTrace:
    active = np.ones((rounds, n_clients), dtype=bool)
    outage_domain = outage_start = outage_end = -1
    degraded = float(link_loss if degraded_link_loss is None else degraded_link_loss)
    degraded_pair = (-1, -1)
    if scenario == "one_domain":
        outage_domain = int(seed % n_domains)
        outage_start = int(np.floor(rounds * outage_start_fraction))
        outage_end = int(np.ceil(rounds * outage_end_fraction))
        domains = np.arange(n_clients) % n_domains
        active[outage_start:outage_end, domains == outage_domain] = False
        if degraded > link_loss:
            degraded_pair = tuple(sorted(((outage_domain + 1) % n_domains, (outage_domain + 2) % n_domains)))
    elif scenario != "nominal":
        raise ValueError(f"Unsupported scenario: {scenario}")
    rng = np.random.default_rng(stable_seed("link-trace", seed, scenario, n_clients, rounds))
    link_up = np.ones((rounds, n_clients, n_clients), dtype=bool)
    if link_loss > 0.0:
        for round_id in range(rounds):
            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    edge_loss = link_loss
                    if degraded_pair != (-1, -1):
                        pair = tuple(sorted((int(i % n_domains), int(j % n_domains))))
                        if pair == degraded_pair:
                            edge_loss = degraded
                    success = bool(rng.random() >= edge_loss)
                    link_up[round_id, i, j] = success
                    link_up[round_id, j, i] = success
    metadata = {"n_clients": n_clients, "n_domains": n_domains, "rounds": rounds, "scenario": scenario, "seed": seed, "outage_domain": outage_domain, "outage_start": outage_start, "outage_end": outage_end, "link_loss": link_loss, "degraded_link_loss": degraded, "degraded_domain_pair": list(degraded_pair)}
    checksum = _trace_checksum(active, link_up, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, active=active, link_up=link_up, metadata=np.asarray([str(metadata)], dtype="U1024"), checksum=np.asarray([checksum], dtype="U64"))
    return NetworkTrace(active, link_up, scenario, seed, outage_domain, outage_start, outage_end, link_loss, degraded, degraded_pair, checksum, str(path.resolve()))


def load_or_create_trace(directory: Path, n_clients: int, n_domains: int, rounds: int, scenario: str, seed: int, link_loss: float, outage_start_fraction: float, outage_end_fraction: float, degraded_link_loss: float | None = None) -> NetworkTrace:
    degraded = float(link_loss if degraded_link_loss is None else degraded_link_loss)
    trace_tag = stable_seed("trace-path-v2", n_clients, n_domains, rounds, scenario, seed, link_loss, degraded, outage_start_fraction, outage_end_fraction)
    path = directory / f"trace_n{n_clients}_r{rounds}_{scenario}_seed{seed}_{trace_tag:08x}.npz"
    if not path.exists():
        return create_trace(path, n_clients, n_domains, rounds, scenario, seed, link_loss, outage_start_fraction, outage_end_fraction, degraded)
    with np.load(path, allow_pickle=False) as archive:
        active = archive["active"].astype(bool)
        link_up = archive["link_up"].astype(bool)
        stored_checksum = str(archive["checksum"][0])
    outage_domain = int(seed % n_domains) if scenario == "one_domain" else -1
    outage_start = int(np.floor(rounds * outage_start_fraction)) if scenario == "one_domain" else -1
    outage_end = int(np.ceil(rounds * outage_end_fraction)) if scenario == "one_domain" else -1
    degraded_pair = tuple(sorted(((outage_domain + 1) % n_domains, (outage_domain + 2) % n_domains))) if scenario == "one_domain" and degraded > link_loss else (-1, -1)
    metadata = {"n_clients": n_clients, "n_domains": n_domains, "rounds": rounds, "scenario": scenario, "seed": seed, "outage_domain": outage_domain, "outage_start": outage_start, "outage_end": outage_end, "link_loss": link_loss, "degraded_link_loss": degraded, "degraded_domain_pair": list(degraded_pair)}
    checksum = _trace_checksum(active, link_up, metadata)
    if checksum != stored_checksum:
        raise RuntimeError(f"Trace checksum mismatch: {path}")
    return NetworkTrace(active, link_up, scenario, seed, outage_domain, outage_start, outage_end, link_loss, degraded, degraded_pair, checksum, str(path.resolve()))
