from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corafl.data import load_bundled_dataset, load_traffic_tensor
from corafl.confirmatory_analysis import _topology_certificate
from corafl.partitions import make_partitions
from corafl.simulation import run_simulation
from corafl.topology import complete_graph_factorization, mixing_matrix, verify_one_domain_certificate
from corafl.traces import load_or_create_trace


class TopologyTests(unittest.TestCase):
    def test_metropolis_matrix_is_doubly_stochastic(self) -> None:
        active = np.ones(8, dtype=bool)
        matrix = mixing_matrix(8, [(0, 1), (2, 3), (4, 5), (6, 7)], active)
        np.testing.assert_allclose(matrix.sum(axis=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(matrix >= 0.0))

    def test_declared_one_domain_certificate(self) -> None:
        certificate = verify_one_domain_certificate(np.arange(8) % 4)
        self.assertTrue(certificate["valid"], certificate["failures"])

    def test_complete_graph_factorization_covers_each_edge_once(self) -> None:
        factors = complete_graph_factorization(8)
        self.assertEqual(len(factors), 7)
        self.assertTrue(all(len(factor) == 4 for factor in factors))
        edges = [edge for factor in factors for edge in factor]
        self.assertEqual(len(edges), 28)
        self.assertEqual(len(set(edges)), 28)

    def test_exhaustive_pair_domain_certificate_counts(self) -> None:
        certificate = _topology_certificate(8).set_index("method")
        self.assertEqual(int(certificate.loc["cora_fl", "valid_domain_mappings"]), 105)
        self.assertEqual(int(certificate.loc["exponential_gt", "valid_domain_mappings"]), 105)
        self.assertEqual(int(certificate.loc["ring_gt", "valid_domain_mappings"]), 2)


class PartitionTests(unittest.TestCase):
    def test_iid_and_noniid_cover_every_split_once(self) -> None:
        dataset = load_bundled_dataset("wine")
        for mode in ("iid", "noniid", "noniid_domain"):
            partition = make_partitions(dataset, mode, 8, 0.3, 5, 1234, 4)
            for parts, target in ((partition.train, dataset.y_train), (partition.val, dataset.y_val), (partition.test, dataset.y_test)):
                joined = np.concatenate(parts)
                self.assertEqual(len(joined), len(target))
                self.assertEqual(len(np.unique(joined)), len(target))
            self.assertGreaterEqual(min(len(indices) for indices in partition.train), 5)

    def test_domain_noniid_preserves_shards_and_groups_similar_clients(self) -> None:
        dataset = load_bundled_dataset("digits")
        ordinary = make_partitions(dataset, "noniid", 8, 0.3, 5, 4321, 4)
        correlated = make_partitions(dataset, "noniid_domain", 8, 0.3, 5, 4321, 4)
        ordinary_shards = {tuple(x.tolist()) for x in ordinary.train}
        correlated_shards = {tuple(x.tolist()) for x in correlated.train}
        self.assertEqual(ordinary_shards, correlated_shards)
        within = [
            np.abs(correlated.class_histograms[d] - correlated.class_histograms[d + 4]).sum()
            for d in range(4)
        ]
        ordinary_within = [
            np.abs(ordinary.class_histograms[d] - ordinary.class_histograms[d + 4]).sum()
            for d in range(4)
        ]
        self.assertLess(float(np.mean(within)), float(np.mean(ordinary_within)))


class DataIntegrityTests(unittest.TestCase):
    def test_dataset_fingerprint_covers_exact_split_arrays(self) -> None:
        first = load_bundled_dataset("wine", 123)
        second = load_bundled_dataset("wine", 123)
        different_split = load_bundled_dataset("wine", 124)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, different_split.fingerprint)

    def test_missing_traffic_file_has_no_synthetic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "No substitute or synthetic data"):
                load_traffic_tensor(Path(tmp) / "missing.npz", "missing")

    def test_trace_cache_reloads_identical_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = load_or_create_trace(directory, 8, 4, 20, "one_domain", 77, 0.05, 0.25, 0.75)
            second = load_or_create_trace(directory, 8, 4, 20, "one_domain", 77, 0.05, 0.25, 0.75)
            self.assertEqual(first.checksum, second.checksum)
            np.testing.assert_array_equal(first.active, second.active)
            self.assertEqual(int(first.active[5:15].sum()), 60)

    def test_correlated_trace_records_degraded_survivor_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = load_or_create_trace(
                Path(tmp), 8, 4, 100, "one_domain", 88, 0.05, 0.35, 0.65, 0.60
            )
            failed = trace.outage_domain
            self.assertEqual(trace.degraded_link_loss, 0.60)
            self.assertNotIn(failed, trace.degraded_domain_pair)
            self.assertEqual(int(trace.active[35:65].sum()), 180)


class SmokeTest(unittest.TestCase):
    def test_short_cora_run_is_finite_and_logged(self) -> None:
        dataset = load_bundled_dataset("wine")
        partition = make_partitions(dataset, "noniid", 8, 0.3, 5, 1234, 4)
        with tempfile.TemporaryDirectory() as tmp:
            trace = load_or_create_trace(Path(tmp), 8, 4, 8, "one_domain", 1234, 0.05, 0.25, 0.625)
            output = run_simulation(run_id="smoke", dataset=dataset, partitions=partition, trace=trace, method="cora_fl", seed=1234, rounds=8, eval_every=2, eval_split="val", learning_rate=0.01, batch_size=8, l2=1e-4, coverage_weight=2.0, domain_weight=0.75, reliability_weight=0.25, coverage_ema_rate=0.2, reliability_ema_rate=0.1, mixing_window=4)
        self.assertEqual(output.curves["round"].tolist(), [0, 2, 4, 6, 8])
        self.assertTrue(np.isfinite(output.curves["loss"]).all())
        self.assertGreater(int(output.summary["total_bytes"]), 0)


if __name__ == "__main__":
    unittest.main()
