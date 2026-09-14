#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corafl.config import load_config
from corafl.data import load_bundled_dataset
from corafl.runner import implementation_hash
from corafl.topology import exponential_matching, natural_ring_order, ring_matching, safety_matching
from corafl.traces import _trace_checksum
from corafl.utils import sha256_file, write_json


FIGURES = {
    "fig1_accuracy_vs_round",
    "fig2_accuracy_vs_bytes",
    "fig3_consensus_vs_round",
    "fig4_dataset_heatmap",
    "fig5_partition_generalizability",
    "fig6_outage_utility",
    "fig7_pareto_accuracy_bytes",
    "fig8_worst_domain_accuracy",
}
TABLES = {"table1_performance", "table2_system", "table3_inference"}


def _assert_frame_close(actual: pd.DataFrame, expected: pd.DataFrame, sort: list[str]) -> None:
    actual = actual.sort_values(sort).reset_index(drop=True)
    expected = expected.sort_values(sort).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=False,
        atol=1e-9,
        rtol=1e-9,
    )


def _paired_metrics(curves: pd.DataFrame, index: pd.DataFrame, config: dict) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    key_columns = ["dataset", "partition", "method", "seed"]
    outage_meta = index[index.scenario == "one_domain"].set_index(key_columns)
    recovery_fraction = float(config["analysis"]["recovery_fraction_of_nominal_final"])
    for key, group in curves.groupby(key_columns, sort=True):
        nominal = group[group.scenario == "nominal"].sort_values("round")
        outage = group[group.scenario == "one_domain"].sort_values("round")
        assert len(nominal) == len(outage) and len(nominal) > 1
        meta = outage_meta.loc[key]
        start, end = int(meta.outage_start), int(meta.outage_end)
        joined = nominal[["round", "accuracy"]].merge(
            outage[["round", "accuracy"]], on="round", suffixes=("_nominal", "_outage")
        )
        post = joined[joined["round"] >= start]
        positive_gap = np.maximum(
            post.accuracy_nominal.to_numpy() - post.accuracy_outage.to_numpy(), 0.0
        )
        during = outage[(outage["round"] >= start) & (outage["round"] <= end)]
        span = float(during["round"].iloc[-1] - during["round"].iloc[0])
        assert span > 0
        nominal_final = float(nominal.iloc[-1].accuracy)
        target = recovery_fraction * nominal_final
        recovered_rows = outage[(outage["round"] >= end) & (outage.accuracy >= target)]
        recovered = not recovered_rows.empty
        recovery_rounds = (
            max(0, int(recovered_rows.iloc[0]["round"]) - end)
            if recovered
            else int(config["experiment"]["rounds"]) - end + int(config["experiment"]["eval_every"])
        )
        records.append(
            {
                "dataset": key[0],
                "partition": key[1],
                "method": key[2],
                "seed": int(key[3]),
                "nominal_final_accuracy": nominal_final,
                "outage_final_accuracy": float(outage.iloc[-1].accuracy),
                "outage_accuracy_auc": float(np.trapezoid(during.accuracy, during["round"]) / span),
                "outage_worst_domain_auc": float(
                    np.trapezoid(during.worst_domain_metric, during["round"]) / span
                ),
                "positive_accuracy_loss_auc": float(np.trapezoid(positive_gap, post["round"])),
                "peak_accuracy_drop": float(np.max(positive_gap)),
                "recovery_rounds": recovery_rounds,
                "recovered": recovered,
                "total_mib": float(outage.iloc[-1].total_bytes / (1024.0**2)),
                "control_kib": float(outage.iloc[-1].control_bytes / 1024.0),
                "messages": int(outage.iloc[-1].messages),
                "successful_edges": int(outage.iloc[-1].successful_edges),
                "final_consensus_error": float(outage.iloc[-1].consensus_error),
            }
        )
    return pd.DataFrame(records)


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values))
    adjusted = np.ones(len(p_values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * float(p_values[int(position)]))
        running = max(running, candidate)
        adjusted[int(position)] = running
    return adjusted.tolist()


def _inference(metrics: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    analysis = config["analysis"]
    partition = str(analysis["primary_partition"])
    comparator = str(analysis["primary_comparator"])
    margin = float(analysis["noninferiority_margin"])
    peers = [m for m in config["experiment"]["methods"] if m not in {"fedavg", "cora_fl"}]
    specs = [
        ("outage_final_accuracy", "higher"),
        ("outage_accuracy_auc", "higher"),
        ("outage_worst_domain_auc", "higher"),
        ("positive_accuracy_loss_auc", "lower"),
    ]
    rows: list[dict[str, object]] = []
    for metric, direction in specs:
        macro = (
            metrics[metrics.partition == partition]
            .groupby(["seed", "method"], as_index=False)[metric]
            .mean()
            .pivot(index="seed", columns="method", values=metric)
        )
        pending: list[dict[str, object]] = []
        p_values: list[float] = []
        for peer in peers:
            raw = macro.cora_fl.to_numpy() - macro[peer].to_numpy()
            oriented = raw if direction == "higher" else -raw
            mean = float(np.mean(raw))
            half = float(t.ppf(0.975, len(raw) - 1) * np.std(raw, ddof=1) / math.sqrt(len(raw)))
            p_value = float(ttest_1samp(oriented, 0.0, alternative="greater").pvalue)
            p_values.append(p_value)
            pending.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "comparator": peer,
                    "cora_minus_comparator": mean,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                    "one_sided_superiority_p": p_value,
                    "n_seed_macros": len(raw),
                }
            )
        for row, adjusted in zip(pending, _holm(p_values)):
            row["holm_adjusted_p"] = adjusted
            row["superiority_after_holm"] = bool(
                adjusted < 0.05
                and ((float(row["cora_minus_comparator"]) > 0) == (direction == "higher"))
            )
            rows.append(row)
    table = pd.DataFrame(rows)
    primary = (
        metrics[metrics.partition == partition]
        .groupby(["seed", "method"], as_index=False).outage_final_accuracy.mean()
        .pivot(index="seed", columns="method", values="outage_final_accuracy")
    )
    delta = primary.cora_fl.to_numpy() - primary[comparator].to_numpy()
    standard_error = float(np.std(delta, ddof=1) / math.sqrt(len(delta)))
    mean = float(np.mean(delta))
    half = float(t.ppf(0.975, len(delta) - 1) * standard_error)
    lower_one_sided = float(mean - t.ppf(0.95, len(delta) - 1) * standard_error)
    return table, {
        "primary_partition": partition,
        "primary_comparator": comparator,
        "noninferiority_margin": margin,
        "n_seed_macros": len(delta),
        "mean_accuracy_delta": mean,
        "one_sided_95_lower": lower_one_sided,
        "two_sided_95_interval": [mean - half, mean + half],
        "accuracy_noninferiority_passed": bool(lower_one_sided > -margin),
        "accuracy_superiority_passed": bool(mean - half > 0.0),
    }


def _pair_partitions(nodes: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
    if not nodes:
        return [tuple()]
    first = nodes[0]
    result: list[tuple[tuple[int, int], ...]] = []
    for position in range(1, len(nodes)):
        remaining = nodes[1:position] + nodes[position + 1 :]
        for rest in _pair_partitions(remaining):
            result.append(((first, nodes[position]), *rest))
    return result


def _connected(nodes: set[int], edges: set[tuple[int, int]]) -> bool:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        if left in nodes and right in nodes:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen = {next(iter(nodes))}
    stack = list(seen)
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node] - seen:
            seen.add(neighbor)
            stack.append(neighbor)
    return seen == nodes


def _valid_schedule(domains: np.ndarray, schedule: Callable[[int], list[tuple[int, int]]], period: int) -> bool:
    for failed in range(4):
        survivors = {i for i in range(8) if int(domains[i]) != failed}
        for start in range(period):
            edges = {
                edge
                for offset in range(4)
                for edge in schedule((start + offset) % period)
                if edge[0] in survivors and edge[1] in survivors
            }
            if not _connected(survivors, edges):
                return False
    return True


def _certificate() -> pd.DataFrame:
    mappings = _pair_partitions(tuple(range(8)))
    rows: list[dict[str, object]] = []
    for method in ("ring_gt", "exponential_gt", "cora_fl"):
        valid = 0
        for groups in mappings:
            domains = np.empty(8, dtype=np.int64)
            for domain, pair in enumerate(groups):
                domains[list(pair)] = domain
            if method == "ring_gt":
                passed = _valid_schedule(
                    domains, lambda r: ring_matching(natural_ring_order(8), r % 2), 2
                )
            elif method == "exponential_gt":
                passed = _valid_schedule(domains, lambda r: exponential_matching(8, r), 3)
            else:
                passed = _valid_schedule(domains, lambda r, d=domains: safety_matching(r, d), 4)
            valid += int(passed)
        rows.append(
            {
                "method": method,
                "schedule_type": "deterministic",
                "valid_domain_mappings": float(valid),
                "audited_domain_mappings": len(mappings),
                "mapping_agnostic_certificate": bool(valid == len(mappings)),
            }
        )
    for method in ("random_gossip_gt", "matcha_gt", "adaptive_similarity_gt"):
        rows.append(
            {
                "method": method,
                "schedule_type": "stochastic/data-adaptive",
                "valid_domain_mappings": np.nan,
                "audited_domain_mappings": len(mappings),
                "mapping_agnostic_certificate": False,
            }
        )
    rows.append(
        {
            "method": "fedavg",
            "schedule_type": "centralized",
            "valid_domain_mappings": np.nan,
            "audited_domain_mappings": len(mappings),
            "mapping_agnostic_certificate": False,
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit the CORA-FL confirmatory bundle")
    parser.add_argument("--config", default="configs/confirmatory.yaml")
    args = parser.parse_args()
    config_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    config = load_config(config_path)
    exp = config["experiment"]
    raw = ROOT / "results" / "raw"
    pilot = ROOT / "results" / "pilot"
    derived = ROOT / "results" / "derived"
    traces = ROOT / "results" / "traces"
    index = pd.read_csv(derived / "run_index.csv")
    current_implementation_hash = implementation_hash()
    checks: list[dict[str, object]] = []

    def check(name: str, function: Callable[[], str]) -> None:
        try:
            detail = function()
            checks.append({"name": name, "passed": True, "detail": detail})
            print(f"[PASS] {name}: {detail}")
        except Exception as exc:
            checks.append({"name": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    lock: dict[str, object] = {}

    def locked_inputs() -> str:
        nonlocal lock
        lock = json.loads((ROOT / "CONFIRMATORY_LOCK.json").read_text(encoding="utf-8"))
        assert lock["stage"] == "pre_results"
        assert lock["config_hash"] == config["_config_hash"]
        assert lock["implementation_hash"] == current_implementation_hash
        assert lock["config_file_sha256"] == sha256_file(config_path)
        protocol = ROOT / "docs" / "CONFIRMATORY_PROTOCOL.md"
        assert lock["protocol_sha256"] == sha256_file(protocol)
        assert int(lock["expected_main_runs"]) == 3780
        assert int(lock["expected_pilot_trials"]) == 630
        return f"config {str(lock['config_hash'])[:12]} and implementation {str(lock['implementation_hash'])[:12]}"

    check("pre-result protocol and implementation lock", locked_inputs)

    expected_keys = {
        (dataset, partition, scenario, method, int(seed))
        for dataset, partition, scenario, method, seed in itertools.product(
            exp["datasets"], exp["partitions"], exp["scenarios"], exp["methods"], exp["main_seeds"]
        )
    }

    def matrix() -> str:
        actual = {
            (row.dataset, row.partition, row.scenario, row.method, int(row.seed))
            for row in index.itertuples(index=False)
        }
        assert len(index) == len(expected_keys) == 3780
        assert actual == expected_keys
        assert not index.run_id.duplicated().any()
        assert (index.status == "succeeded").all()
        assert (index.config_hash == config["_config_hash"]).all()
        assert (index.implementation_hash == current_implementation_hash).all()
        return "3,780 unique succeeded runs with locked hashes"

    check("complete factorial matrix", matrix)

    expected_rounds = list(range(0, int(exp["rounds"]) + 1, int(exp["eval_every"])))
    frames: list[pd.DataFrame] = []

    def raw_logs() -> str:
        trace_cache: dict[str, str] = {}
        for row in index.itertuples(index=False):
            csv_path = raw / f"{row.run_id}.csv"
            json_path = raw / f"{row.run_id}.json"
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            assert metadata["curves_sha256"] == sha256_file(csv_path)
            assert metadata["implementation_hash"] == current_implementation_hash
            frame = pd.read_csv(csv_path)
            frames.append(frame)
            assert frame["round"].astype(int).tolist() == expected_rounds
            numeric = ["loss", "accuracy", "gradient_norm", "worst_domain_metric", "consensus_error", "total_bytes", "messages"]
            assert np.isfinite(frame[numeric].to_numpy()).all()
            assert np.allclose(frame.payload_bytes + frame.control_bytes, frame.total_bytes, atol=0, rtol=0)
            assert set(frame.eval_split) == {"test"}
            final = frame.iloc[-1]
            assert np.isclose(float(metadata["final_accuracy"]), float(final.accuracy), atol=1e-12)
            assert np.isclose(float(metadata["final_loss"]), float(final.loss), atol=1e-12)
            recorded = Path(metadata["trace_path"])
            trace_path = recorded if recorded.is_file() else traces / recorded.name
            key = str(trace_path.resolve())
            if key not in trace_cache:
                with np.load(trace_path, allow_pickle=False) as archive:
                    active = archive["active"].astype(bool)
                    link_up = archive["link_up"].astype(bool)
                    stored = str(archive["checksum"][0])
                    trace_meta = ast.literal_eval(str(archive["metadata"][0]))
                assert active.shape == (int(exp["rounds"]), int(exp["n_clients"]))
                assert link_up.shape == (int(exp["rounds"]), int(exp["n_clients"]), int(exp["n_clients"]))
                assert np.array_equal(link_up, np.swapaxes(link_up, 1, 2))
                assert stored == _trace_checksum(active, link_up, trace_meta)
                scenario = str(trace_meta["scenario"])
                expected_loss = float(config["topology"]["nominal_link_loss"] if scenario == "nominal" else config["topology"]["outage_link_loss"])
                expected_degraded = expected_loss if scenario == "nominal" else float(config["topology"]["degraded_link_loss"])
                assert float(trace_meta["link_loss"]) == expected_loss
                assert float(trace_meta["degraded_link_loss"]) == expected_degraded
                if scenario == "nominal":
                    assert active.all() and tuple(trace_meta["degraded_domain_pair"]) == (-1, -1)
                else:
                    start, end = int(trace_meta["outage_start"]), int(trace_meta["outage_end"])
                    assert (start, end) == (35, 65)
                    assert int(active[start:end].sum()) == (end - start) * 6
                trace_cache[key] = stored
            assert metadata["trace_checksum"] == trace_cache[key]
        return f"{len(index)} curve/metadata pairs and {len(trace_cache)} frozen traces"

    check("raw values and cryptographic checksums", raw_logs)

    def fairness() -> str:
        assert index.groupby(["dataset", "partition", "scenario", "seed"]).trace_checksum.nunique().max() == 1
        assert index.groupby(["dataset", "partition", "seed"]).partition_fingerprint.nunique().max() == 1
        assert index.groupby(["dataset"]).dataset_fingerprint.nunique().max() == 1
        for dataset_name in exp["datasets"]:
            expected = load_bundled_dataset(dataset_name, int(exp["split_seed"])).fingerprint
            assert set(index[index.dataset == dataset_name].dataset_fingerprint) == {expected}
        assert set(index.groupby(["dataset", "partition", "method", "seed"]).scenario.nunique()) == {2}
        selection = pd.read_csv(pilot / "lr_selection.csv")
        lookup = selection.set_index(["dataset", "partition", "method"]).learning_rate
        for row in index.itertuples(index=False):
            assert float(row.learning_rate) == float(lookup.loc[(row.dataset, row.partition, row.method)])
        return "identical paired partitions, traces, datasets, and locked learning rates"

    check("paired-comparison fairness", fairness)

    def pilot_policy() -> str:
        trials = pd.read_csv(pilot / "pilot_trials.csv")
        curves = pd.read_csv(pilot / "pilot_curves.csv")
        selected = pd.read_csv(pilot / "lr_selection.csv")
        policy = json.loads((pilot / "pilot_policy.json").read_text(encoding="utf-8"))
        expected_trials = len(exp["datasets"]) * len(exp["partitions"]) * len(exp["methods"]) * len(config["pilot"]["learning_rates"])
        expected_selected = len(exp["datasets"]) * len(exp["partitions"]) * len(exp["methods"])
        assert expected_trials == 630 and len(trials) == expected_trials
        assert expected_selected == 63 and len(selected) == expected_selected
        assert (trials.status == "succeeded").all()
        assert (trials.config_hash == config["_config_hash"]).all()
        assert (trials.implementation_hash == current_implementation_hash).all()
        assert set(curves.eval_split) == {"val"}
        assert not bool(policy["test_set_accessed"])
        assert policy["config_hash"] == config["_config_hash"]
        assert policy["implementation_hash"] == current_implementation_hash
        assert int(policy["lower_boundary_selection_count"]) == 0
        assert int(policy["upper_boundary_selection_count"]) == 0
        recomputed_rows = []
        for _, group in trials.groupby(["dataset", "partition", "method"], sort=True):
            recomputed_rows.append(group.sort_values(["final_loss", "learning_rate"]).iloc[0])
        recomputed = pd.DataFrame(recomputed_rows)[["dataset", "partition", "method", "learning_rate", "final_loss", "run_id"]].rename(columns={"final_loss": "validation_loss", "run_id": "pilot_run_id"})
        _assert_frame_close(selected, recomputed, ["dataset", "partition", "method"])
        return "630 validation-only trials, 63 interior-grid selections, zero test access"

    check("validation-only tuning policy", pilot_policy)

    certificate = _certificate()

    def certificate_check() -> str:
        stored = pd.read_csv(derived / "topology_certificate.csv")
        _assert_frame_close(stored, certificate, ["method"])
        lookup = certificate.set_index("method")
        assert int(lookup.loc["cora_fl", "valid_domain_mappings"]) == 105
        assert int(lookup.loc["exponential_gt", "valid_domain_mappings"]) == 105
        assert int(lookup.loc["ring_gt", "valid_domain_mappings"]) == 2
        return "all 105 pair-domain mappings audited; CORA-FL 105/105, Exponential-GT 105/105, Ring-GT 2/105"

    check("exhaustive topology certificate", certificate_check)

    metrics = _paired_metrics(pd.concat(frames, ignore_index=True), index, config) if frames else pd.DataFrame()
    inference, decision = _inference(metrics, config) if not metrics.empty else (pd.DataFrame(), {})

    def derived_statistics() -> str:
        stored_metrics = pd.read_csv(derived / "paired_robustness.csv")
        stored_inference = pd.read_csv(derived / "confirmatory_inference.csv")
        _assert_frame_close(stored_metrics, metrics, ["dataset", "partition", "method", "seed"])
        _assert_frame_close(stored_inference, inference, ["metric", "comparator"])
        assert set(inference.n_seed_macros.astype(int)) == {30}
        certificate_lookup = certificate.set_index("method")["mapping_agnostic_certificate"]
        comparator = str(decision["primary_comparator"])
        decision["cora_mapping_agnostic_certificate"] = bool(certificate_lookup.loc["cora_fl"])
        decision["primary_comparator_mapping_agnostic_certificate"] = bool(certificate_lookup.loc[comparator])
        decision["qualified_claim_passed"] = bool(
            decision["accuracy_noninferiority_passed"]
            and decision["cora_mapping_agnostic_certificate"]
            and not decision["primary_comparator_mapping_agnostic_certificate"]
        )
        stored_decision = json.loads((derived / "claim_decision.json").read_text(encoding="utf-8"))
        for key, expected in decision.items():
            actual = stored_decision[key]
            if isinstance(expected, float):
                assert np.isclose(float(actual), expected, atol=1e-12, rtol=1e-12)
            elif isinstance(expected, list):
                assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)
            else:
                assert actual == expected
        return "1,890 paired records and all predeclared seed-macro tests recomputed"

    check("derived robustness and inference", derived_statistics)

    def tables() -> str:
        methods = list(exp["methods"])
        selected = metrics[metrics.partition == config["analysis"]["primary_partition"]]
        macro = selected.groupby(["seed", "method"], as_index=False)[["outage_final_accuracy", "outage_accuracy_auc", "outage_worst_domain_auc", "positive_accuracy_loss_auc"]].mean()
        expected_performance = macro.groupby("method").agg(
            final_accuracy_mean=("outage_final_accuracy", "mean"),
            final_accuracy_std=("outage_final_accuracy", "std"),
            outage_accuracy_auc_mean=("outage_accuracy_auc", "mean"),
            outage_accuracy_auc_std=("outage_accuracy_auc", "std"),
            worst_domain_auc_mean=("outage_worst_domain_auc", "mean"),
            degradation_auc_mean=("positive_accuracy_loss_auc", "mean"),
            n_seeds=("seed", "count"),
        ).reindex(methods).reset_index()
        expected_system = selected.groupby("method").agg(
            total_mib=("total_mib", "mean"),
            control_kib=("control_kib", "mean"),
            messages=("messages", "mean"),
            successful_edges=("successful_edges", "mean"),
            consensus=("final_consensus_error", "mean"),
        ).reindex(methods).reset_index().merge(certificate, on="method", how="left")
        _assert_frame_close(pd.read_csv(ROOT / "tables" / "table1_performance.csv"), expected_performance, ["method"])
        _assert_frame_close(pd.read_csv(ROOT / "tables" / "table2_system.csv"), expected_system, ["method"])
        _assert_frame_close(pd.read_csv(ROOT / "tables" / "table3_inference.csv"), inference, ["metric", "comparator"])
        return "three tables independently recomputed from raw runs"

    check("table recomputation", tables)

    def artifacts() -> str:
        pngs = {path.stem for path in (ROOT / "figures").glob("fig*.png")}
        pdfs = {path.stem for path in (ROOT / "figures").glob("fig*.pdf")}
        plot_data = {path.stem for path in (derived / "plot_data").glob("fig*.csv")}
        csvs = {path.stem for path in (ROOT / "tables").glob("table*.csv")}
        texs = {path.stem for path in (ROOT / "tables").glob("table*.tex")}
        assert pngs == pdfs == plot_data == FIGURES
        assert csvs == texs == TABLES
        for stem in FIGURES:
            png = ROOT / "figures" / f"{stem}.png"
            pdf = ROOT / "figures" / f"{stem}.pdf"
            assert png.stat().st_size > 10_000 and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            assert pdf.stat().st_size > 1_000 and pdf.read_bytes()[:4] == b"%PDF"
            assert (derived / "plot_data" / f"{stem}.csv").stat().st_size > 0
        return "exactly 8 PNG, 8 PDF, 8 plot-data CSV, 3 CSV tables, and 3 TeX tables"

    check("artifact cardinality and signatures", artifacts)

    passed = all(bool(item["passed"]) for item in checks)
    report = {
        "passed": passed,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config["_config_hash"],
        "implementation_hash": current_implementation_hash,
        "checks": checks,
    }
    verification_path = derived / "verification.json"
    write_json(verification_path, report)
    if not passed:
        raise SystemExit(1)

    candidates = sorted(
        [
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and path.name != "ARTIFACT_MANIFEST.sha256"
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".zip"}
        ],
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    manifest = "\n".join(
        f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}" for path in candidates
    ) + "\n"
    (ROOT / "ARTIFACT_MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    print(f"Verified {len(candidates)} files; manifest written")


if __name__ == "__main__":
    main()
