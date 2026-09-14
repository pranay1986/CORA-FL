#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corafl.config import load_config
from corafl.topology import verify_one_domain_certificate
from corafl.traces import _trace_checksum
from corafl.utils import sha256_file, write_json

FIGURES = {"fig1_accuracy_vs_round", "fig2_accuracy_vs_bytes", "fig3_consensus_vs_round", "fig4_dataset_partition_heatmap", "fig5_iid_vs_noniid", "fig6_robustness_auc_cvar", "fig7_recovery_rounds", "fig8_worst_domain_accuracy"}
TABLES = {"table1_accuracy", "table2_robustness", "table3_cost"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the supplied result bundle")
    parser.add_argument("--config", default="configs/development.yaml")
    args = parser.parse_args()
    config_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    config = load_config(config_path)
    exp = config["experiment"]
    derived, raw, traces = ROOT / "results/derived", ROOT / "results/raw", ROOT / "results/traces"
    index = pd.read_csv(derived / "run_index.csv")
    checks: list[dict[str, object]] = []

    def check(name: str, fn) -> None:
        try:
            detail = fn()
            checks.append({"name": name, "passed": True, "detail": detail})
            print(f"[PASS] {name}: {detail}")
        except Exception as exc:
            checks.append({"name": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    expected_keys = {(d, p, s, m, int(seed)) for d in exp["datasets"] for p in exp["partitions"] for s in exp["scenarios"] for m in exp["methods"] for seed in exp["main_seeds"]}

    def matrix() -> str:
        actual = {(r.dataset, r.partition, r.scenario, r.method, int(r.seed)) for r in index.itertuples(index=False)}
        assert len(index) == len(expected_keys) and actual == expected_keys
        assert not index.run_id.duplicated().any() and (index.status == "succeeded").all()
        assert (index.config_hash == config["_config_hash"]).all()
        return f"{len(index)} unique succeeded runs"

    check("complete locked factorial matrix", matrix)
    expected_rounds = list(range(0, int(exp["rounds"]) + 1, int(exp["eval_every"])))

    def raw_logs() -> str:
        trace_cache: dict[Path, str] = {}
        for row in index.itertuples(index=False):
            csv_path, json_path = raw / f"{row.run_id}.csv", raw / f"{row.run_id}.json"
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            assert metadata["curves_sha256"] == sha256_file(csv_path)
            frame = pd.read_csv(csv_path)
            assert frame["round"].astype(int).tolist() == expected_rounds
            assert np.isfinite(frame[["loss", "accuracy", "gradient_norm", "total_bytes", "messages"]].to_numpy()).all()
            assert np.allclose(frame.payload_bytes + frame.control_bytes, frame.total_bytes, atol=0, rtol=0)
            assert set(frame.eval_split) == {"test"}
            final = frame.iloc[-1]
            assert np.isclose(float(metadata["final_accuracy"]), float(final.accuracy), atol=1e-12)
            assert np.isclose(float(metadata["final_loss"]), float(final.loss), atol=1e-12)
            recorded = Path(metadata["trace_path"])
            path = recorded if recorded.is_file() else traces / recorded.name
            if path not in trace_cache:
                match = re.search(r"_seed(\d+)\.npz$", path.name)
                assert match is not None
                trace_seed = int(match.group(1))
                with np.load(path, allow_pickle=False) as archive:
                    active, link_up, stored = archive["active"].astype(bool), archive["link_up"].astype(bool), str(archive["checksum"][0])
                scenario = str(row.scenario)
                meta = {"n_clients": int(exp["n_clients"]), "n_domains": int(config["topology"]["n_domains"]), "rounds": int(exp["rounds"]), "scenario": scenario, "seed": trace_seed, "outage_domain": trace_seed % int(config["topology"]["n_domains"]) if scenario == "one_domain" else -1, "outage_start": int(np.floor(int(exp["rounds"]) * config["topology"]["outage_start_fraction"])) if scenario == "one_domain" else -1, "outage_end": int(np.ceil(int(exp["rounds"]) * config["topology"]["outage_end_fraction"])) if scenario == "one_domain" else -1, "link_loss": float(config["topology"]["outage_link_loss"] if scenario == "one_domain" else config["topology"]["nominal_link_loss"])}
                assert stored == _trace_checksum(active, link_up, meta)
                trace_cache[path] = stored
            assert metadata["trace_checksum"] == trace_cache[path]
        return f"{len(index)} CSV/JSON pairs and {len(trace_cache)} traces"

    check("raw logs, final values, and checksums", raw_logs)

    def fairness() -> str:
        assert index.groupby(["dataset", "partition", "scenario", "seed"]).trace_checksum.nunique().max() == 1
        assert index.groupby(["dataset", "partition", "seed"]).partition_fingerprint.nunique().max() == 1
        assert set(index.groupby(["dataset", "partition", "method", "seed"]).scenario.nunique()) == {2}
        return "shared partitions and traces verified across methods"

    check("paired-comparison fairness", fairness)

    def pilot() -> str:
        trials = pd.read_csv(ROOT / "results/pilot/pilot_trials.csv")
        selected = pd.read_csv(ROOT / "results/pilot/lr_selection.csv")
        policy = json.loads((ROOT / "results/pilot/pilot_policy.json").read_text())
        expected = len(exp["datasets"]) * len(exp["partitions"]) * len(exp["methods"]) * len(config["pilot"]["learning_rates"])
        assert len(trials) == expected and (trials.status == "succeeded").all()
        assert len(selected) == 42 and not policy["test_set_accessed"] and policy["upper_boundary_selection_count"] == 0
        return f"{len(trials)} validation-only trials; {len(selected)} locked selections"

    check("validation-only hyperparameter policy", pilot)

    def certificate() -> str:
        stored = json.loads((derived / "certificate.json").read_text())
        recomputed = verify_one_domain_certificate(np.asarray(stored["domains"]), int(stored["period"]), int(stored["window"]))
        assert stored["valid"] and recomputed["valid"] and not recomputed["failures"]
        return "all declared one-domain removals and cyclic windows"

    check("structural schedule certificate", certificate)

    def tables() -> str:
        t1 = pd.read_csv(ROOT / "tables/table1_accuracy.csv")
        expected = index[index.scenario == "one_domain"].groupby(["method", "dataset", "partition", "scenario"]).final_accuracy.agg(["mean", "std", "count"]).reset_index()
        pd.testing.assert_frame_equal(t1.sort_values(["method", "dataset", "partition"]).reset_index(drop=True), expected.sort_values(["method", "dataset", "partition"]).reset_index(drop=True), check_dtype=False, check_exact=False, atol=1e-9, rtol=1e-9)
        assert all((ROOT / "tables" / f"{stem}.csv").stat().st_size > 0 for stem in TABLES)
        return "three nonempty tables; accuracy table recomputed from runs"

    check("derived table recomputation", tables)

    def artifacts() -> str:
        pngs = {p.stem for p in (ROOT / "figures").glob("fig*.png")}
        pdfs = {p.stem for p in (ROOT / "figures").glob("fig*.pdf")}
        csvs = {p.stem for p in (ROOT / "tables").glob("table*.csv")}
        texs = {p.stem for p in (ROOT / "tables").glob("table*.tex")}
        assert pngs == FIGURES and pdfs == FIGURES and csvs == TABLES and texs == TABLES
        for stem in FIGURES:
            assert (ROOT / "figures" / f"{stem}.png").stat().st_size > 10_000
            assert (derived / "plot_data" / f"{stem}.csv").stat().st_size > 0
        return "8 PNG + 8 PDF figures, 8 plot-data CSVs, and 3 CSV + 3 TeX tables"

    check("artifact cardinality and file signatures", artifacts)
    passed = all(bool(item["passed"]) for item in checks)
    report = {"passed": passed, "verified_at_utc": datetime.now(timezone.utc).isoformat(), "config_hash": config["_config_hash"], "checks": checks}
    verification_path = derived / "verification.json"
    candidates = sorted([p for p in ROOT.rglob("*") if p.is_file() and p.name != "ARTIFACT_MANIFEST.sha256" and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".zip"}], key=lambda p: p.relative_to(ROOT).as_posix())
    if verification_path not in candidates:
        candidates.append(verification_path)
        candidates.sort(key=lambda p: p.relative_to(ROOT).as_posix())
    report["manifest_entries"] = len(candidates)
    write_json(verification_path, report)
    if passed:
        (ROOT / "ARTIFACT_MANIFEST.sha256").write_text("\n".join(f"{sha256_file(p)}  {p.relative_to(ROOT).as_posix()}" for p in candidates) + "\n", encoding="utf-8")
        print(f"Verified {len(candidates)} files; manifest written")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
