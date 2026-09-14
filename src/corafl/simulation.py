from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import DatasetBundle
from .models import LinearRegression, SoftmaxRegression, make_model
from .partitions import PartitionBundle
from .topology import component_count, cross_domain_fraction, filter_edges, initialize_state, mixing_matrix, proposed_edges, update_adaptive_state
from .traces import NetworkTrace
from .utils import stable_seed

Model = SoftmaxRegression | LinearRegression


@dataclass(frozen=True)
class SimulationOutput:
    curves: pd.DataFrame
    summary: dict[str, Any]


def _split_arrays(dataset: DatasetBundle, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        return dataset.x_train, dataset.y_train
    if split == "val":
        return dataset.x_val, dataset.y_val
    if split == "test":
        return dataset.x_test, dataset.y_test
    raise ValueError(f"Unknown split: {split}")


def _split_parts(partitions: PartitionBundle, split: str) -> list[np.ndarray]:
    if split == "train":
        return partitions.train
    if split == "val":
        return partitions.val
    if split == "test":
        return partitions.test
    raise ValueError(f"Unknown split: {split}")


def _batch_indices(indices: np.ndarray, batch_size: int, seed: int, client_id: int, round_id: int) -> np.ndarray:
    if len(indices) <= batch_size:
        return indices
    rng = np.random.default_rng(stable_seed("batch", seed, client_id, round_id))
    return np.sort(rng.choice(indices, size=batch_size, replace=False))


def _client_gradient(model: Model, theta: np.ndarray, dataset: DatasetBundle, indices: np.ndarray) -> np.ndarray:
    loss, gradient = model.loss_grad(theta, dataset.x_train[indices], dataset.y_train[indices])
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Non-finite client loss or gradient")
    return gradient


def _domain_metric(model: Model, theta: np.ndarray, dataset: DatasetBundle, partitions: PartitionBundle, split: str) -> float:
    x, y = _split_arrays(dataset, split)
    parts = _split_parts(partitions, split)
    values: list[float] = []
    for domain in sorted(set(int(item) for item in partitions.domains)):
        selected_clients = np.flatnonzero(partitions.domains == domain)
        nonempty = [parts[int(client)] for client in selected_clients if len(parts[int(client)])]
        if not nonempty:
            continue
        indices = np.concatenate(nonempty)
        metrics = model.metrics(theta, x[indices], y[indices])
        values.append(metrics["accuracy"] if dataset.task == "classification" else metrics["mae"])
    if not values:
        return float("nan")
    return float(min(values) if dataset.task == "classification" else max(values))


def _evaluate(model: Model, theta: np.ndarray, dataset: DatasetBundle, partitions: PartitionBundle, split: str) -> dict[str, float]:
    x, y = _split_arrays(dataset, split)
    metrics = model.metrics(theta, x, y)
    metrics["worst_domain_metric"] = _domain_metric(model, theta, dataset, partitions, split)
    return metrics


def _control_accounting(n_active: int, n_outputs: int) -> tuple[int, int]:
    record_bytes = np.zeros(n_outputs, dtype=np.float32).nbytes + np.zeros(1, dtype=np.int16).nbytes + np.zeros(2, dtype=np.int64).nbytes
    return int(n_active * record_bytes), int(2 * n_active)


def run_simulation(*, run_id: str, dataset: DatasetBundle, partitions: PartitionBundle, trace: NetworkTrace, method: str, seed: int, rounds: int, eval_every: int, eval_split: str, learning_rate: float, batch_size: int, l2: float, coverage_weight: float, domain_weight: float, reliability_weight: float, coverage_ema_rate: float, reliability_ema_rate: float, mixing_window: int) -> SimulationOutput:
    if trace.active.shape[0] != rounds:
        raise ValueError("Trace length and requested rounds disagree")
    model = make_model(dataset, l2)
    n_clients = len(partitions.train)
    parameter_count = model.parameter_count
    parameter_bytes = model.initial_parameters().nbytes
    start_time = time.perf_counter()
    payload_bytes = control_bytes = message_count = attempted_edge_count = successful_edge_count = 0
    recent_edges: list[list[tuple[int, int]]] = []
    state = initialize_state(partitions.class_histograms, seed)

    if method == "fedavg":
        global_theta = model.initial_parameters()
        client_theta = np.repeat(global_theta[None, :], n_clients, axis=0)
        tracker = np.zeros_like(client_theta)
        grad_previous = np.zeros_like(client_theta)
    else:
        client_theta = np.repeat(model.initial_parameters()[None, :], n_clients, axis=0)
        grad_previous = np.zeros_like(client_theta)
        for client_id, indices in enumerate(partitions.train):
            batch = _batch_indices(indices, batch_size, seed, client_id, 0)
            grad_previous[client_id] = _client_gradient(model, client_theta[client_id], dataset, batch)
        tracker = grad_previous.copy()
        global_theta = model.initial_parameters()

    rows: list[dict[str, Any]] = []

    def append_evaluation(round_value: int, active: np.ndarray) -> None:
        if method == "fedavg":
            report_theta = global_theta
            consensus_error = 0.0
        else:
            active_ids = np.flatnonzero(active)
            report_theta = client_theta[active_ids].mean(axis=0) if len(active_ids) else client_theta.mean(axis=0)
            consensus_error = float(np.mean(np.sum((client_theta[active_ids] - report_theta) ** 2, axis=1))) if len(active_ids) else float("nan")
        metrics = _evaluate(model, report_theta, dataset, partitions, eval_split)
        if recent_edges:
            union = sorted({edge for edge_list in recent_edges[-mixing_window:] for edge in edge_list})
            components, largest_fraction = component_count(n_clients, union, active)
        else:
            components, largest_fraction = component_count(n_clients, [], active)
        rows.append({"run_id": run_id, "round": round_value, "dataset": dataset.name, "partition": partitions.mode, "scenario": trace.scenario, "method": method, "seed": seed, "eval_split": eval_split, "learning_rate": learning_rate, "loss": metrics["loss"], "accuracy": metrics["accuracy"], "mae": metrics["mae"], "rmse": metrics["rmse"], "gradient_norm": metrics["gradient_norm"], "worst_domain_metric": metrics["worst_domain_metric"], "consensus_error": consensus_error, "payload_bytes": payload_bytes, "control_bytes": control_bytes, "total_bytes": payload_bytes + control_bytes, "messages": message_count, "attempted_edges": attempted_edge_count, "successful_edges": successful_edge_count, "active_clients": int(active.sum()), "window_components": components, "largest_component_fraction": largest_fraction, "wall_time_seconds": time.perf_counter() - start_time})

    append_evaluation(0, trace.active[0])
    for round_id in range(rounds):
        active = trace.active[round_id]
        active_ids = np.flatnonzero(active)
        if method == "fedavg":
            weighted_gradient = np.zeros(parameter_count, dtype=np.float64)
            total_weight = 0
            for client_id in active_ids:
                client = int(client_id)
                batch = _batch_indices(partitions.train[client], batch_size, seed, client, round_id + 1)
                gradient = _client_gradient(model, global_theta, dataset, batch)
                weight = len(batch)
                weighted_gradient += weight * gradient
                total_weight += weight
            if total_weight:
                global_theta = global_theta - learning_rate * weighted_gradient / total_weight
            client_theta[active_ids] = global_theta
            payload_bytes += int(len(active_ids) * 2 * parameter_bytes)
            message_count += int(len(active_ids) * 2)
            recent_edges.append([])
        else:
            scheduled, adaptive = proposed_edges(method, round_id, active, partitions.class_histograms, partitions.domains, state, seed, coverage_weight, domain_weight, reliability_weight)
            attempted, successful = filter_edges(scheduled, active, trace.link_up[round_id])
            weights = mixing_matrix(n_clients, successful, active)
            theta_new = weights @ client_theta - learning_rate * tracker
            theta_new[~active] = client_theta[~active]
            grad_new = grad_previous.copy()
            for client_id in active_ids:
                client = int(client_id)
                batch = _batch_indices(partitions.train[client], batch_size, seed, client, round_id + 1)
                grad_new[client] = _client_gradient(model, theta_new[client], dataset, batch)
            tracker_new = weights @ tracker + grad_new - grad_previous
            tracker_new[~active] = tracker[~active]
            client_theta, tracker, grad_previous = theta_new, tracker_new, grad_new
            if adaptive:
                update_adaptive_state(state, attempted, successful, partitions.class_histograms, coverage_ema_rate, reliability_ema_rate)
                added_bytes, added_messages = _control_accounting(int(active.sum()), dataset.n_outputs)
                control_bytes += added_bytes
                message_count += added_messages
            payload_bytes += int(len(attempted) * 4 * parameter_bytes)
            message_count += int(len(attempted) * 4)
            attempted_edge_count += len(attempted)
            successful_edge_count += len(successful)
            recent_edges.append(successful)
        if (round_id + 1) % eval_every == 0 or round_id + 1 == rounds:
            append_evaluation(round_id + 1, active)

    curves = pd.DataFrame(rows)
    final = curves.iloc[-1]
    summary = {"run_id": run_id, "dataset": dataset.name, "dataset_source": dataset.source, "dataset_fingerprint": dataset.fingerprint, "partition": partitions.mode, "partition_fingerprint": partitions.fingerprint, "scenario": trace.scenario, "trace_path": trace.path, "trace_checksum": trace.checksum, "outage_domain": trace.outage_domain, "outage_start": trace.outage_start, "outage_end": trace.outage_end, "base_link_loss": trace.link_loss, "degraded_link_loss": trace.degraded_link_loss, "degraded_domain_pair": list(trace.degraded_domain_pair), "method": method, "seed": seed, "rounds": rounds, "eval_every": eval_every, "eval_split": eval_split, "learning_rate": learning_rate, "batch_size": batch_size, "l2": l2, "n_clients": n_clients, "n_domains": int(len(set(int(x) for x in partitions.domains))), "parameter_count": parameter_count, "parameter_dtype": "float64", "final_loss": float(final["loss"]), "final_accuracy": float(final["accuracy"]), "final_mae": float(final["mae"]), "final_rmse": float(final["rmse"]), "final_worst_domain_metric": float(final["worst_domain_metric"]), "final_consensus_error": float(final["consensus_error"]), "payload_bytes": int(final["payload_bytes"]), "control_bytes": int(final["control_bytes"]), "total_bytes": int(final["total_bytes"]), "messages": int(final["messages"]), "attempted_edges": int(final["attempted_edges"]), "successful_edges": int(final["successful_edges"]), "final_window_components": int(final["window_components"]), "final_largest_component_fraction": float(final["largest_component_fraction"]), "mean_cross_domain_fraction": float(np.mean([cross_domain_fraction(edges, partitions.domains) for edges in recent_edges if edges])) if any(recent_edges) else 0.0, "wall_time_seconds": float(time.perf_counter() - start_time)}
    return SimulationOutput(curves=curves, summary=summary)
