from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .data import load_bundled_dataset
from .partitions import make_partitions
from .simulation import run_simulation
from .topology import verify_one_domain_certificate
from .traces import load_or_create_trace
from .utils import environment_record, sha256_file, sha256_tree, stable_seed, write_json


BASELINE_SCOPE = {
    "fedavg": "One-local-step centralized FedAvg reference with sample-weighted active-client gradients.",
    "ring_gt": "Gradient tracking over alternating matchings whose two-round union is the natural client-id ring.",
    "random_gossip_gt": "Gradient tracking over a fresh seeded random matching each round.",
    "exponential_gt": "Gradient tracking over a rotating XOR/exponential matching schedule.",
    "static_rr_gt": "Gradient tracking over four fixed seeded matchings; a sparse random-regular-style overlay control.",
    "matcha_gt": "Gradient tracking over one independently sampled factor of a complete-graph matching decomposition per round, using the symmetric one-matching MATCHA budget.",
    "adaptive_similarity_gt": "Gradient tracking over pairwise label-histogram-similarity matching; a diagnostic baseline, not a claimed reimplementation of a named paper.",
    "cora_fl_v1": "Development-only alias retained for rejected-variant audit logs; excluded from the confirmatory matrix.",
    "cora_fl": "Gradient tracking over a four-round certified scaffold interleaved with coverage-debt/domain/reliability-aware adaptive matching.",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def implementation_hash() -> str:
    root = repo_root()
    files = [path.relative_to(root) for base in (root / "src", root / "scripts", root / "tests") for path in base.rglob("*.py")]
    return sha256_tree(root, files)


def _paths() -> dict[str, Path]:
    root = repo_root()
    return {"root": root, "pilot": root / "results" / "pilot", "raw": root / "results" / "raw", "derived": root / "results" / "derived", "traces": root / "results" / "traces", "manifests": root / "data" / "manifests"}


def _common_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {"batch_size": int(config["model"]["batch_size"]), "l2": float(config["model"]["l2"]), "coverage_weight": float(config["topology"]["coverage_weight"]), "domain_weight": float(config["topology"]["domain_weight"]), "reliability_weight": float(config["topology"]["reliability_weight"]), "coverage_ema_rate": float(config["topology"]["coverage_ema_rate"]), "reliability_ema_rate": float(config["topology"].get("reliability_ema_rate", 0.1)), "mixing_window": int(config["topology"]["mixing_window"])}


def _partition(config: dict[str, Any], dataset: Any, mode: str, seed: int) -> Any:
    return make_partitions(dataset, mode, int(config["experiment"]["n_clients"]), float(config["partition"]["dirichlet_alpha"]), int(config["partition"]["min_train_samples"]), seed, int(config["topology"]["n_domains"]))


def _trace(config: dict[str, Any], rounds: int, scenario: str, seed: int) -> Any:
    link_loss = float(config["topology"]["nominal_link_loss"] if scenario == "nominal" else config["topology"]["outage_link_loss"])
    degraded_link_loss = float(config["topology"].get("degraded_link_loss", link_loss)) if scenario != "nominal" else link_loss
    return load_or_create_trace(_paths()["traces"], int(config["experiment"]["n_clients"]), int(config["topology"]["n_domains"]), rounds, scenario, seed, link_loss, float(config["topology"]["outage_start_fraction"]), float(config["topology"]["outage_end_fraction"]), degraded_link_loss)


def _write_dataset_manifest(config: dict[str, Any], dataset: Any) -> None:
    manifest = {"name": dataset.name, "source": dataset.source, "task": dataset.task, "n_features": dataset.n_features, "n_outputs": dataset.n_outputs, "train_samples": len(dataset.y_train), "validation_samples": len(dataset.y_val), "test_samples": len(dataset.y_test), "split_seed": dataset.split_seed, "fingerprint": dataset.fingerprint, "scope_warning": "Bundled classification benchmark; not a smart-city traffic result."}
    write_json(_paths()["manifests"] / f"{dataset.name}.json", manifest)


def _save_environment(config: dict[str, Any]) -> None:
    record = environment_record(_paths()["root"])
    record["config_path"] = config["_config_path"]
    record["config_hash"] = config["_config_hash"]
    record["implementation_hash"] = implementation_hash()
    write_json(_paths()["derived"] / "environment.json", record)


def _save_certificate(config: dict[str, Any]) -> None:
    n_clients = int(config["experiment"]["n_clients"])
    n_domains = int(config["topology"]["n_domains"])
    domains = np.arange(n_clients, dtype=np.int64) % n_domains
    certificate = verify_one_domain_certificate(domains)
    certificate["domains"] = domains.tolist()
    certificate["scope"] = "Structural certificate only: no claim for arbitrary node sets, Byzantine faults, or independent link losses outside the declared one-domain family."
    if not certificate["valid"]:
        raise RuntimeError(f"CORA-FL structural certificate failed: {certificate['failures']}")
    write_json(_paths()["derived"] / "certificate.json", certificate)


def run_pilot(config_path: str | Path) -> Path:
    config = load_config(config_path)
    paths = _paths()
    paths["pilot"].mkdir(parents=True, exist_ok=True)
    _save_environment(config)
    _save_certificate(config)
    exp, pilot, common = config["experiment"], config["pilot"], _common_kwargs(config)
    all_curves: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    total = len(exp["datasets"]) * len(exp["partitions"]) * len(exp["methods"]) * len(pilot["learning_rates"])
    counter = 0
    for dataset_name in exp["datasets"]:
        dataset = load_bundled_dataset(dataset_name, int(exp["split_seed"]))
        _write_dataset_manifest(config, dataset)
        for partition_mode in exp["partitions"]:
            partitions = _partition(config, dataset, partition_mode, stable_seed("pilot-partition", int(pilot["seed"]), dataset_name, partition_mode))
            trace = _trace(config, int(pilot["rounds"]), "nominal", stable_seed("pilot-trace", int(pilot["seed"]), dataset_name, partition_mode))
            for method in exp["methods"]:
                for learning_rate in pilot["learning_rates"]:
                    counter += 1
                    run_id = f"pilot__{dataset_name}__{partition_mode}__{method}__lr{learning_rate:g}"
                    try:
                        output = run_simulation(run_id=run_id, dataset=dataset, partitions=partitions, trace=trace, method=method, seed=int(pilot["seed"]), rounds=int(pilot["rounds"]), eval_every=int(pilot["eval_every"]), eval_split="val", learning_rate=float(learning_rate), **common)
                        curves = output.curves.copy()
                        curves["status"] = "succeeded"
                        all_curves.append(curves)
                        records.append({**output.summary, "status": "succeeded", "error": "", "baseline_scope": BASELINE_SCOPE[method], "config_hash": config["_config_hash"], "implementation_hash": implementation_hash()})
                    except Exception as exc:
                        records.append({"run_id": run_id, "dataset": dataset_name, "partition": partition_mode, "method": method, "learning_rate": float(learning_rate), "status": "failed", "final_loss": math.inf, "error": f"{type(exc).__name__}: {exc}", "config_hash": config["_config_hash"], "implementation_hash": implementation_hash()})
                    print(f"pilot {counter}/{total}: {run_id}", flush=True)
    trials = pd.DataFrame(records)
    trials.to_csv(paths["pilot"] / "pilot_trials.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(paths["pilot"] / "pilot_curves.csv", index=False)
    succeeded = trials[trials["status"] == "succeeded"].copy()
    selected_rows = [group.sort_values(["final_loss", "learning_rate"], ascending=[True, True]).iloc[0] for _, group in succeeded.groupby(["dataset", "partition", "method"], sort=True)]
    selection = pd.DataFrame(selected_rows)[["dataset", "partition", "method", "learning_rate", "final_loss", "run_id"]].rename(columns={"final_loss": "validation_loss", "run_id": "pilot_run_id"})
    expected = len(exp["datasets"]) * len(exp["partitions"]) * len(exp["methods"])
    if len(selection) != expected:
        raise RuntimeError(f"Pilot selected {len(selection)} settings; expected {expected}")
    selection.to_csv(paths["pilot"] / "lr_selection.csv", index=False)
    lower_boundary = float(min(pilot["learning_rates"]))
    upper_boundary = float(max(pilot["learning_rates"]))
    lower_boundary_count = int(np.isclose(selection["learning_rate"], lower_boundary).sum())
    boundary_count = int(np.isclose(selection["learning_rate"], upper_boundary).sum())
    write_json(paths["pilot"] / "pilot_policy.json", {"selection_metric": pilot["selection_metric"], "selection_rule": "Minimum final validation loss; smaller learning rate breaks exact ties.", "test_set_accessed": False, "pilot_seed": int(pilot["seed"]), "candidate_learning_rates": [float(x) for x in pilot["learning_rates"]], "selected_count": len(selection), "lower_boundary": lower_boundary, "lower_boundary_selection_count": lower_boundary_count, "upper_boundary": upper_boundary, "upper_boundary_selection_count": boundary_count, "grid_note": pilot.get("grid_note", ""), "config_hash": config["_config_hash"], "implementation_hash": implementation_hash()})
    fail_on_boundary = bool(pilot.get("fail_on_boundary", pilot.get("fail_on_upper_boundary", True)))
    if fail_on_boundary and (lower_boundary_count or boundary_count):
        raise RuntimeError(
            "Pilot grid remains boundary-limited: "
            f"{lower_boundary_count} selections equal {lower_boundary} and "
            f"{boundary_count} equal {upper_boundary}"
        )
    return paths["pilot"] / "lr_selection.csv"


def _learning_rate_lookup(path: Path) -> dict[tuple[str, str, str], float]:
    selection = pd.read_csv(path)
    return {(row.dataset, row.partition, row.method): float(row.learning_rate) for row in selection.itertuples(index=False)}


def _existing_valid(metadata_path: Path, curves_path: Path, config_hash: str, code_hash: str) -> bool:
    if not metadata_path.is_file() or not curves_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("config_hash") == config_hash
            and metadata.get("implementation_hash") == code_hash
            and metadata.get("status") == "succeeded"
            and metadata.get("curves_sha256") == sha256_file(curves_path)
        )
    except (json.JSONDecodeError, OSError):
        return False


def run_main(config_path: str | Path, force: bool = False) -> Path:
    config = load_config(config_path)
    paths = _paths()
    for key in ("raw", "derived", "traces", "manifests"):
        paths[key].mkdir(parents=True, exist_ok=True)
    _save_environment(config)
    _save_certificate(config)
    selection_path = paths["pilot"] / "lr_selection.csv"
    if not selection_path.is_file():
        run_pilot(config_path)
    policy_path = paths["pilot"] / "pilot_policy.json"
    if not policy_path.is_file():
        raise RuntimeError("Pilot policy is missing; rerun the validation-only pilot")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    code_hash = implementation_hash()
    if policy.get("config_hash") != config["_config_hash"] or policy.get("implementation_hash") != code_hash:
        raise RuntimeError("Pilot selections do not match the active config and implementation; rerun the pilot")
    learning_rates = _learning_rate_lookup(selection_path)
    exp, common = config["experiment"], _common_kwargs(config)
    matrix = [(dataset, partition, scenario, method, int(seed)) for dataset in exp["datasets"] for partition in exp["partitions"] for scenario in exp["scenarios"] for method in exp["methods"] for seed in exp["main_seeds"]]
    completed = skipped = 0
    for run_number, (dataset_name, partition_mode, scenario, method, seed) in enumerate(matrix, start=1):
        run_id = f"{exp['name']}__{config['_config_hash'][:8]}__{dataset_name}__{partition_mode}__{scenario}__{method}__seed{seed}"
        curves_path = paths["raw"] / f"{run_id}.csv"
        metadata_path = paths["raw"] / f"{run_id}.json"
        if not force and _existing_valid(metadata_path, curves_path, config["_config_hash"], code_hash):
            skipped += 1
            continue
        dataset = load_bundled_dataset(dataset_name, int(exp["split_seed"]))
        _write_dataset_manifest(config, dataset)
        partitions = _partition(config, dataset, partition_mode, stable_seed("main-partition", seed, dataset_name, partition_mode))
        trace = _trace(config, int(exp["rounds"]), scenario, stable_seed("main-trace", seed, dataset_name, partition_mode))
        try:
            output = run_simulation(run_id=run_id, dataset=dataset, partitions=partitions, trace=trace, method=method, seed=seed, rounds=int(exp["rounds"]), eval_every=int(exp["eval_every"]), eval_split="test", learning_rate=learning_rates[(dataset_name, partition_mode, method)], **common)
            output.curves.to_csv(curves_path, index=False)
            metadata = {**output.summary, "status": "succeeded", "config_hash": config["_config_hash"], "implementation_hash": code_hash, "config_path": config["_config_path"], "baseline_scope": BASELINE_SCOPE[method], "curves_sha256": sha256_file(curves_path), "scope_warning": "Bundled classification benchmark; not a real smart-city traffic result."}
            write_json(metadata_path, metadata)
            completed += 1
        except Exception as exc:
            write_json(metadata_path, {"run_id": run_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            raise
        print(f"main {run_number}/{len(matrix)}: {run_id}", flush=True)
    metadata_rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths["raw"].glob(f"{exp['name']}__{config['_config_hash'][:8]}__*.json"))]
    index = pd.DataFrame(metadata_rows)
    index_path = paths["derived"] / "run_index.csv"
    index.to_csv(index_path, index=False)
    write_json(paths["derived"] / "completion.json", {"expected_runs": len(matrix), "indexed_runs": len(index), "completed_this_invocation": completed, "reused_cached_runs": skipped, "failed_runs": int((index.get("status", pd.Series(dtype=str)) != "succeeded").sum()), "config_hash": config["_config_hash"]})
    return index_path


def run_all(config_path: str | Path, force: bool = False) -> Path:
    run_pilot(config_path)
    return run_main(config_path, force=force)
