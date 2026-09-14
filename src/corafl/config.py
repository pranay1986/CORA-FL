from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .utils import sha256_json


REQUIRED_METHODS = {
    "fedavg",
    "ring_gt",
    "random_gossip_gt",
    "exponential_gt",
    "static_rr_gt",
    "matcha_gt",
    "adaptive_similarity_gt",
    "cora_fl_v1",
    "cora_fl",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    config["_config_path"] = str(config_path)
    config["_config_hash"] = sha256_json({k: v for k, v in config.items() if not k.startswith("_")})
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    exp = config["experiment"]
    methods = set(exp["methods"])
    if len(methods) != len(exp["methods"]):
        raise ValueError("Method names must be unique")
    unknown = methods - REQUIRED_METHODS
    if unknown:
        raise ValueError(f"Unknown method(s): {sorted(unknown)}")
    if "cora_fl" not in methods:
        raise ValueError("The experiment must include cora_fl")
    n_clients = int(exp["n_clients"])
    if n_clients < 4 or n_clients % 2:
        raise ValueError("n_clients must be an even integer >= 4")
    if n_clients & (n_clients - 1):
        raise ValueError("n_clients must be a power of two for the exponential schedule")
    n_domains = int(config["topology"]["n_domains"])
    if n_clients % n_domains:
        raise ValueError("n_clients must be divisible by n_domains")
    if not 0.0 < float(config["partition"]["dirichlet_alpha"]):
        raise ValueError("dirichlet_alpha must be positive")
    if not exp["main_seeds"]:
        raise ValueError("At least one main seed is required")
    if len(set(int(seed) for seed in exp["main_seeds"])) != len(exp["main_seeds"]):
        raise ValueError("Main seeds must be unique")
    partitions = set(exp["partitions"])
    if "iid" not in partitions or not partitions.intersection({"noniid", "noniid_domain"}):
        raise ValueError("IID and at least one supported non-IID partition are required")
    if set(exp["scenarios"]) != {"nominal", "one_domain"}:
        raise ValueError("The experiment matrix requires nominal and one_domain scenarios")
    rounds = int(exp["rounds"])
    eval_every = int(exp["eval_every"])
    if rounds <= 0 or eval_every <= 0 or rounds % eval_every:
        raise ValueError("rounds must be positive and divisible by eval_every")
    topology = config["topology"]
    start = rounds * float(topology["outage_start_fraction"])
    end = rounds * float(topology["outage_end_fraction"])
    if not (0.0 < start < end < rounds):
        raise ValueError("Outage fractions must define an interior nonempty interval")
    if not float(start).is_integer() or not float(end).is_integer() or int(start) % eval_every or int(end) % eval_every:
        raise ValueError("Outage boundaries must align with evaluation rounds")
    nominal_loss = float(topology["nominal_link_loss"])
    outage_loss = float(topology["outage_link_loss"])
    degraded_loss = float(topology.get("degraded_link_loss", outage_loss))
    if not (0.0 <= nominal_loss <= 1.0 and 0.0 <= outage_loss <= degraded_loss <= 1.0):
        raise ValueError("Link-loss probabilities must satisfy 0 <= nominal <= outage <= degraded <= 1")
    learning_rates = [float(value) for value in config["pilot"]["learning_rates"]]
    if not learning_rates or any(value <= 0 for value in learning_rates) or learning_rates != sorted(set(learning_rates)):
        raise ValueError("Pilot learning rates must be positive, unique, and increasing")
    analysis = config.get("analysis", {})
    if "primary_partition" in analysis and analysis["primary_partition"] not in exp["partitions"]:
        raise ValueError("primary_partition must be included in the experiment")
    if "primary_comparator" in analysis:
        comparator = analysis["primary_comparator"]
        if comparator not in methods or comparator in {"fedavg", "cora_fl"}:
            raise ValueError("primary_comparator must be a decentralized peer other than cora_fl")
    if "noninferiority_margin" in analysis and float(analysis["noninferiority_margin"]) < 0.0:
        raise ValueError("noninferiority_margin must be non-negative")
