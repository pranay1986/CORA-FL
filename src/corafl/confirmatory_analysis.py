from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp

from .config import load_config
from .runner import repo_root
from .topology import (
    Edge,
    _connected,
    exponential_matching,
    natural_ring_order,
    ring_matching,
    safety_matching,
)
from .utils import write_json


METHOD_LABEL = {
    "fedavg": "FedAvg (central)",
    "ring_gt": "Ring-GT",
    "random_gossip_gt": "Random-Gossip-GT",
    "exponential_gt": "Exponential-GT",
    "matcha_gt": "MATCHA-GT",
    "adaptive_similarity_gt": "Similarity-Adaptive-GT",
    "cora_fl_v1": "CORA-FL v1 (development)",
    "cora_fl": "CORA-FL",
}

COLORS = {
    "fedavg": "#222222",
    "ring_gt": "#D55E00",
    "random_gossip_gt": "#0072B2",
    "exponential_gt": "#56B4E9",
    "matcha_gt": "#009E73",
    "adaptive_similarity_gt": "#CC79A7",
    "cora_fl_v1": "#F0C36E",
    "cora_fl": "#E69F00",
}

PARTITION_LABEL = {
    "iid": "IID",
    "noniid": "Dirichlet non-IID",
    "noniid_domain": "Domain-correlated non-IID",
}


def _paths() -> dict[str, Path]:
    root = repo_root()
    return {
        "root": root,
        "raw": root / "results" / "raw",
        "derived": root / "results" / "derived",
        "plot_data": root / "results" / "derived" / "plot_data",
        "figures": root / "figures",
        "tables": root / "tables",
    }


def _methods(config: dict[str, Any]) -> list[str]:
    return [str(method) for method in config["experiment"]["methods"]]


def _load(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _paths()
    index = pd.read_csv(paths["derived"] / "run_index.csv")
    index = index[index["config_hash"] == config["_config_hash"]].copy()
    exp = config["experiment"]
    expected = len(exp["datasets"]) * len(exp["partitions"]) * len(exp["scenarios"]) * len(exp["methods"]) * len(exp["main_seeds"])
    if len(index) != expected or not (index.status == "succeeded").all():
        raise RuntimeError(f"Expected {expected} succeeded confirmatory runs; found {len(index)}")
    frames: list[pd.DataFrame] = []
    for run_id in index.run_id:
        frame = pd.read_csv(paths["raw"] / f"{run_id}.csv")
        required = ["accuracy", "loss", "worst_domain_metric", "consensus_error", "total_bytes"]
        if not np.isfinite(frame[required].to_numpy()).all():
            raise RuntimeError(f"Non-finite result in {run_id}")
        frames.append(frame)
    return index, pd.concat(frames, ignore_index=True)


def _paired_metrics(curves: pd.DataFrame, index: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["dataset", "partition", "method", "seed"]
    outage_meta = index[index.scenario == "one_domain"].set_index(keys)
    recovery_fraction = float(config["analysis"]["recovery_fraction_of_nominal_final"])
    for key, pair in curves.groupby(keys, sort=True):
        nominal = pair[pair.scenario == "nominal"].sort_values("round")
        outage = pair[pair.scenario == "one_domain"].sort_values("round")
        if nominal.empty or outage.empty:
            raise RuntimeError(f"Missing nominal/outage pair: {key}")
        meta = outage_meta.loc[key]
        start, end = int(meta.outage_start), int(meta.outage_end)
        joined = nominal[["round", "accuracy"]].merge(
            outage[["round", "accuracy"]], on="round", suffixes=("_nominal", "_outage")
        )
        post = joined[joined["round"] >= start]
        positive_gap = np.maximum(post.accuracy_nominal.to_numpy() - post.accuracy_outage.to_numpy(), 0.0)
        during = outage[(outage["round"] >= start) & (outage["round"] <= end)]
        span = float(during["round"].iloc[-1] - during["round"].iloc[0])
        if span <= 0:
            raise RuntimeError("Outage interval has fewer than two evaluation points")
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
                "outage_worst_domain_auc": float(np.trapezoid(during.worst_domain_metric, during["round"]) / span),
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


def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    half = float(t.ppf((1.0 + confidence) / 2.0, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values))
    adjusted = np.ones(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, position in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[int(position)]))
        running = max(running, value)
        adjusted[int(position)] = running
    return adjusted.tolist()


def _macro_by_seed(metrics: pd.DataFrame, partition: str, value: str) -> pd.DataFrame:
    selected = metrics[metrics.partition == partition]
    return selected.groupby(["seed", "method"], as_index=False)[value].mean()


def _inference(metrics: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    analysis = config["analysis"]
    partition = str(analysis["primary_partition"])
    comparator = str(analysis["primary_comparator"])
    margin = float(analysis["noninferiority_margin"])
    methods = [method for method in _methods(config) if method not in {"fedavg", "cora_fl"}]
    metric_specs = [
        ("outage_final_accuracy", "higher"),
        ("outage_accuracy_auc", "higher"),
        ("outage_worst_domain_auc", "higher"),
        ("positive_accuracy_loss_auc", "lower"),
    ]
    rows: list[dict[str, Any]] = []
    for metric, direction in metric_specs:
        macro = _macro_by_seed(metrics, partition, metric).pivot(index="seed", columns="method", values=metric)
        metric_rows: list[dict[str, Any]] = []
        p_values: list[float] = []
        for peer in methods:
            raw = macro.cora_fl.to_numpy() - macro[peer].to_numpy()
            oriented = raw if direction == "higher" else -raw
            mean, low, high = _mean_ci(raw)
            test = ttest_1samp(oriented, popmean=0.0, alternative="greater")
            p_values.append(float(test.pvalue))
            metric_rows.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "comparator": peer,
                    "cora_minus_comparator": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "one_sided_superiority_p": float(test.pvalue),
                    "n_seed_macros": len(raw),
                }
            )
        adjusted = _holm(p_values)
        for row, p_adjusted in zip(metric_rows, adjusted):
            row["holm_adjusted_p"] = p_adjusted
            row["superiority_after_holm"] = bool(p_adjusted < 0.05 and ((row["cora_minus_comparator"] > 0) == (direction == "higher")))
            rows.append(row)
    table = pd.DataFrame(rows)
    primary = _macro_by_seed(metrics, partition, "outage_final_accuracy").pivot(index="seed", columns="method", values="outage_final_accuracy")
    delta = primary.cora_fl.to_numpy() - primary[comparator].to_numpy()
    se = float(np.std(delta, ddof=1) / math.sqrt(len(delta)))
    one_sided_lower = float(np.mean(delta) - t.ppf(0.95, len(delta) - 1) * se)
    two_mean, two_low, two_high = _mean_ci(delta)
    decision = {
        "primary_partition": partition,
        "primary_comparator": comparator,
        "noninferiority_margin": margin,
        "n_seed_macros": len(delta),
        "mean_accuracy_delta": two_mean,
        "one_sided_95_lower": one_sided_lower,
        "two_sided_95_interval": [two_low, two_high],
        "accuracy_noninferiority_passed": bool(one_sided_lower > -margin),
        "accuracy_superiority_passed": bool(two_low > 0.0),
    }
    # The structural portion of the qualified claim is attached only after the
    # exhaustive certificate has been recomputed in generate_confirmatory_artifacts.
    decision["qualified_claim_passed"] = False
    return table, decision


def _pair_partitions(nodes: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
    if not nodes:
        return [tuple()]
    first = nodes[0]
    result: list[tuple[tuple[int, int], ...]] = []
    for position in range(1, len(nodes)):
        second = nodes[position]
        remaining = nodes[1:position] + nodes[position + 1 :]
        for rest in _pair_partitions(remaining):
            result.append(((first, second), *rest))
    return result


def _schedule_valid(domains: np.ndarray, schedule: Callable[[int], list[Edge]], period: int, window: int = 4) -> bool:
    for failed_domain in sorted(set(int(item) for item in domains)):
        survivors = {i for i in range(len(domains)) if int(domains[i]) != failed_domain}
        for start in range(period):
            union: set[Edge] = set()
            for offset in range(window):
                union.update(schedule((start + offset) % period))
            usable = {edge for edge in union if edge[0] in survivors and edge[1] in survivors}
            if not _connected(survivors, usable):
                return False
    return True


def _topology_certificate(n_clients: int) -> pd.DataFrame:
    if n_clients != 8:
        raise ValueError("The exhaustive domain-mapping audit is locked to eight clients")
    mappings = _pair_partitions(tuple(range(n_clients)))
    rows: list[dict[str, Any]] = []
    for method in ("ring_gt", "exponential_gt", "cora_fl"):
        valid = 0
        for groups in mappings:
            domains = np.empty(n_clients, dtype=np.int64)
            for domain, pair in enumerate(groups):
                for node in pair:
                    domains[node] = domain
            if method == "ring_gt":
                check = _schedule_valid(domains, lambda r: ring_matching(natural_ring_order(n_clients), r % 2), 2)
            elif method == "exponential_gt":
                check = _schedule_valid(domains, lambda r: exponential_matching(n_clients, r), int(np.log2(n_clients)))
            else:
                check = _schedule_valid(domains, lambda r, d=domains: safety_matching(r, d), 4)
            valid += int(check)
        rows.append(
            {
                "method": method,
                "schedule_type": "deterministic",
                "valid_domain_mappings": valid,
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


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    paths = _paths()
    paths["figures"].mkdir(parents=True, exist_ok=True)
    fig.text(0.995, 0.004, "Locked confirmatory benchmark; measured and unsmoothed", ha="right", va="bottom", fontsize=5.8, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(paths["figures"] / f"{stem}.png", bbox_inches="tight")
    fig.savefig(paths["figures"] / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _curve_summary(frame: pd.DataFrame, metric: str, x_metric: str = "round") -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for round_id, group in frame.groupby("round", sort=True):
        mean, low, high = _mean_ci(group[metric].to_numpy())
        x_value = float(round_id) if x_metric == "round" else float(group[x_metric].mean() / (1024.0**2))
        rows.append({"round": float(round_id), "x_value": x_value, "mean": mean, "ci_low": low, "ci_high": high, "n": len(group)})
    return pd.DataFrame(rows)


def _line_plot(curves: pd.DataFrame, config: dict[str, Any], metric: str, ylabel: str, title: str, stem: str, x_metric: str = "round", include_central: bool = True, symlog: bool = False) -> None:
    methods = _methods(config)
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    exports: list[pd.DataFrame] = []
    for method in methods:
        if not include_central and method == "fedavg":
            continue
        summary = _curve_summary(curves[curves.method == method], metric, x_metric)
        ax.plot(summary.x_value, summary["mean"], color=COLORS[method], label=METHOD_LABEL[method], linewidth=1.7)
        if not symlog:
            ax.fill_between(summary.x_value, summary.ci_low, summary.ci_high, color=COLORS[method], alpha=0.12, linewidth=0)
        summary["method"] = method
        summary["metric"] = metric
        exports.append(summary)
    if x_metric == "round":
        rounds = int(config["experiment"]["rounds"])
        start = rounds * float(config["topology"]["outage_start_fraction"])
        end = rounds * float(config["topology"]["outage_end_fraction"])
        ax.axvspan(start, end, color="#999999", alpha=0.12, label="regional outage")
        ax.set_xlabel("Communication round")
    else:
        ax.set_xlabel("Cumulative transmitted MiB")
    if symlog:
        ax.set_yscale("symlog", linthresh=1e-3, linscale=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2, frameon=False)
    _save(fig, stem)
    pd.concat(exports, ignore_index=True).to_csv(_paths()["plot_data"] / f"{stem}.csv", index=False)


def _make_figures(curves: pd.DataFrame, index: pd.DataFrame, metrics: pd.DataFrame, config: dict[str, Any]) -> None:
    _style()
    paths = _paths()
    paths["plot_data"].mkdir(parents=True, exist_ok=True)
    methods = _methods(config)
    analysis = config["analysis"]
    primary = curves[
        (curves.dataset == analysis["primary_dataset"])
        & (curves.partition == analysis["primary_partition"])
        & (curves.scenario == "one_domain")
    ]
    primary_title = f"{str(analysis['primary_dataset']).replace('_', ' ').title()}, domain-correlated non-IID outage"
    _line_plot(primary, config, "accuracy", "Test accuracy", primary_title, "fig1_accuracy_vs_round")
    _line_plot(primary, config, "accuracy", "Test accuracy", "Accuracy versus transmitted bytes", "fig2_accuracy_vs_bytes", x_metric="total_bytes")
    _line_plot(primary, config, "consensus_error", "Consensus error (symlog)", "Peer-model disagreement", "fig3_consensus_vs_round", include_central=False, symlog=True)

    target = metrics[metrics.partition == analysis["primary_partition"]]
    heat = target.pivot_table(index="method", columns="dataset", values="outage_final_accuracy", aggfunc="mean").reindex(methods)[config["experiment"]["datasets"]]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    image = ax.imshow(heat.to_numpy(), vmin=max(0.70, float(heat.min().min()) - 0.02), vmax=1.0, cmap="YlGnBu", aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = float(heat.iloc[i, j])
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color="white" if value > 0.94 else "black")
    ax.set_xticks(range(heat.shape[1]), [str(x).replace("_", " ").title() for x in heat.columns])
    ax.set_yticks(range(heat.shape[0]), [METHOD_LABEL[x] for x in heat.index])
    ax.set_title("Final accuracy across datasets")
    fig.colorbar(image, ax=ax, label="Mean test accuracy")
    _save(fig, "fig4_dataset_heatmap")
    heat.to_csv(paths["plot_data"] / "fig4_dataset_heatmap.csv")

    partition_macro = metrics.groupby(["seed", "method", "partition"], as_index=False).outage_final_accuracy.mean()
    partition_summary = partition_macro.groupby(["method", "partition"]).outage_final_accuracy.agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(methods))
    partitions = [str(item) for item in config["experiment"]["partitions"]]
    width = 0.78 / len(partitions)
    palette = ["#4C78A8", "#F58518", "#54A24B"]
    for pos, partition in enumerate(partitions):
        selected = partition_summary.set_index(["method", "partition"]).loc[[(m, partition) for m in methods]]
        ci = t.ppf(0.975, selected["count"] - 1) * selected["std"] / np.sqrt(selected["count"])
        offset = (pos - (len(partitions) - 1) / 2.0) * width
        ax.bar(x + offset, selected["mean"], width, yerr=ci, capsize=2, label=PARTITION_LABEL[partition], color=palette[pos], alpha=0.88)
    ax.set_xticks(x, [METHOD_LABEL[m] for m in methods], rotation=24, ha="right")
    ax.set_ylim(max(0.70, float(partition_summary["mean"].min()) - 0.04), 1.0)
    ax.set_ylabel("Macro final accuracy")
    ax.set_title("IID and non-IID generalizability")
    ax.legend(frameon=False, ncol=len(partitions))
    _save(fig, "fig5_partition_generalizability")
    partition_summary.to_csv(paths["plot_data"] / "fig5_partition_generalizability.csv", index=False)

    seed_primary = target.groupby(["seed", "method"], as_index=False)[["outage_accuracy_auc", "positive_accuracy_loss_auc"]].mean()
    robust_summary = seed_primary.groupby("method")[["outage_accuracy_auc", "positive_accuracy_loss_auc"]].agg(["mean", "std", "count"]).reindex(methods)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    for ax, metric, label, title in [
        (axes[0], "outage_accuracy_auc", "Outage accuracy AUC", "Absolute service utility (higher)"),
        (axes[1], "positive_accuracy_loss_auc", "Positive loss AUC", "Degradation from nominal (lower)"),
    ]:
        means = robust_summary[(metric, "mean")].to_numpy()
        counts = robust_summary[(metric, "count")].to_numpy()
        errors = t.ppf(0.975, counts - 1) * robust_summary[(metric, "std")].to_numpy() / np.sqrt(counts)
        ax.bar(np.arange(len(methods)), means, yerr=errors, capsize=2, color=[COLORS[m] for m in methods], alpha=0.85)
        ax.set_xticks(np.arange(len(methods)), [METHOD_LABEL[m] for m in methods], rotation=55, ha="right")
        ax.set_ylabel(label)
        ax.set_title(title)
    _save(fig, "fig6_outage_utility")
    robust_summary.to_csv(paths["plot_data"] / "fig6_outage_utility.csv")

    pareto_seed = target.groupby(["seed", "method"], as_index=False)[["outage_final_accuracy", "total_mib"]].mean()
    pareto = pareto_seed.groupby("method").agg(accuracy=("outage_final_accuracy", "mean"), accuracy_std=("outage_final_accuracy", "std"), total_mib=("total_mib", "mean"), total_mib_std=("total_mib", "std"), n=("seed", "count")).reindex(methods)
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for method, row in pareto.iterrows():
        ax.errorbar(row.total_mib, row.accuracy, xerr=1.96 * row.total_mib_std / math.sqrt(row.n), yerr=1.96 * row.accuracy_std / math.sqrt(row.n), fmt="o", color=COLORS[method], capsize=2, label=METHOD_LABEL[method])
    ax.set_xlabel("Total transmitted MiB")
    ax.set_ylabel("Final outage accuracy")
    ax.set_title("Communication–accuracy trade-off")
    ax.legend(frameon=False, ncol=2)
    _save(fig, "fig7_pareto_accuracy_bytes")
    pareto.to_csv(paths["plot_data"] / "fig7_pareto_accuracy_bytes.csv")

    _line_plot(primary, config, "worst_domain_metric", "Worst-domain test accuracy", "Failure-domain tail utility", "fig8_worst_domain_accuracy")


def _latex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]]) -> None:
    alignment = "l" + "c" * (len(headers) - 1)
    end = r" \\"
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small", f"\\begin{{tabular}}{{{alignment}}}", "\\toprule", " & ".join(headers) + end, "\\midrule"]
    lines.extend(" & ".join(row) + end for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_tables(metrics: pd.DataFrame, index: pd.DataFrame, inference: pd.DataFrame, certificate: pd.DataFrame, config: dict[str, Any]) -> None:
    paths = _paths()
    paths["tables"].mkdir(parents=True, exist_ok=True)
    methods = _methods(config)
    partition = str(config["analysis"]["primary_partition"])
    selected = metrics[metrics.partition == partition]
    macro = selected.groupby(["seed", "method"], as_index=False)[["outage_final_accuracy", "outage_accuracy_auc", "outage_worst_domain_auc", "positive_accuracy_loss_auc"]].mean()
    performance = macro.groupby("method").agg(
        final_accuracy_mean=("outage_final_accuracy", "mean"),
        final_accuracy_std=("outage_final_accuracy", "std"),
        outage_accuracy_auc_mean=("outage_accuracy_auc", "mean"),
        outage_accuracy_auc_std=("outage_accuracy_auc", "std"),
        worst_domain_auc_mean=("outage_worst_domain_auc", "mean"),
        degradation_auc_mean=("positive_accuracy_loss_auc", "mean"),
        n_seeds=("seed", "count"),
    ).reindex(methods).reset_index()
    performance.to_csv(paths["tables"] / "table1_performance.csv", index=False, float_format="%.10g")
    rows = []
    for row in performance.itertuples(index=False):
        rows.append([METHOD_LABEL[row.method], f"{row.final_accuracy_mean:.4f} $\\pm$ {row.final_accuracy_std:.4f}", f"{row.outage_accuracy_auc_mean:.4f} $\\pm$ {row.outage_accuracy_auc_std:.4f}", f"{row.worst_domain_auc_mean:.4f}", f"{row.degradation_auc_mean:.4f}"])
    _latex_table(paths["tables"] / "table1_performance.tex", "Locked domain-correlated non-IID results, macro-averaged across datasets within each seed.", "tab:confirmatory_performance", ["Method", "Final accuracy", "Outage utility", "Worst-domain utility", "Degradation AUC"], rows)

    system = selected.groupby("method").agg(total_mib=("total_mib", "mean"), control_kib=("control_kib", "mean"), messages=("messages", "mean"), successful_edges=("successful_edges", "mean"), consensus=("final_consensus_error", "mean")).reindex(methods).reset_index()
    system = system.merge(certificate, on="method", how="left")
    system.to_csv(paths["tables"] / "table2_system.csv", index=False, float_format="%.10g")
    system_rows = []
    for row in system.itertuples(index=False):
        mapping = "--" if pd.isna(row.valid_domain_mappings) else f"{int(row.valid_domain_mappings)}/{int(row.audited_domain_mappings)}"
        system_rows.append([METHOD_LABEL[row.method], f"{row.total_mib:.3f}", f"{row.control_kib:.2f}", f"{row.messages:.1f}", f"{row.successful_edges:.1f}", f"{row.consensus:.2e}", mapping])
    _latex_table(paths["tables"] / "table2_system.tex", "Communication, consensus, and exhaustive four-round connectivity audit over all 105 pair-domain mappings.", "tab:confirmatory_system", ["Method", "MiB", "Control KiB", "Messages", "Success edges", "Consensus", "Certified mappings"], system_rows)

    inference.to_csv(paths["tables"] / "table3_inference.csv", index=False, float_format="%.10g")
    final_rows = inference[inference.metric == "outage_final_accuracy"]
    inference_rows = [[METHOD_LABEL[row.comparator], f"{row.cora_minus_comparator:+.5f}", f"[{row.ci95_low:+.5f}, {row.ci95_high:+.5f}]", f"{row.holm_adjusted_p:.4g}", "yes" if row.superiority_after_holm else "no"] for row in final_rows.itertuples(index=False)]
    _latex_table(paths["tables"] / "table3_inference.tex", "Paired CORA-FL final-accuracy comparisons using dataset-macro values within each locked seed. Holm adjustment covers decentralized peers.", "tab:confirmatory_inference", ["Comparator", "$\\Delta$ accuracy", "95\\% CI", "Holm $p$", "Superior"], inference_rows)


def _write_report(metrics: pd.DataFrame, inference: pd.DataFrame, decision: dict[str, Any], config: dict[str, Any]) -> None:
    partition = str(config["analysis"]["primary_partition"])
    macro = metrics[metrics.partition == partition].groupby(["seed", "method"], as_index=False)[["outage_final_accuracy", "outage_accuracy_auc", "positive_accuracy_loss_auc"]].mean()
    summary = macro.groupby("method")[["outage_final_accuracy", "outage_accuracy_auc", "positive_accuracy_loss_auc"]].mean().reindex(_methods(config))
    lines = [
        "# CORA-FL locked confirmatory report",
        "",
        "> These are measured classification-benchmark results. They are not traffic-field results and must not be described as such.",
        "",
        f"- Locked seeds: {len(config['experiment']['main_seeds'])}.",
        f"- Primary unit: one three-dataset macro-average per seed ($n={decision['n_seed_macros']}$).",
        f"- Primary comparator: {METHOD_LABEL[decision['primary_comparator']]}",
        f"- Non-inferiority margin: {decision['noninferiority_margin']:.4f} absolute accuracy.",
        "",
        "## Primary operating regime",
        "",
        "| Method | Final accuracy | Outage utility AUC | Positive degradation AUC |",
        "|---|---:|---:|---:|",
    ]
    for method, row in summary.iterrows():
        lines.append(f"| {METHOD_LABEL[method]} | {row.outage_final_accuracy:.6f} | {row.outage_accuracy_auc:.6f} | {row.positive_accuracy_loss_auc:.6f} |")
    lines.extend(
        [
            "",
            "## Confirmatory decision",
            "",
            f"CORA-FL minus {METHOD_LABEL[decision['primary_comparator']]} final accuracy: {decision['mean_accuracy_delta']:+.6f}; two-sided 95% CI [{decision['two_sided_95_interval'][0]:+.6f}, {decision['two_sided_95_interval'][1]:+.6f}].",
            "",
            f"One-sided 95% lower bound: {decision['one_sided_95_lower']:+.6f}. Non-inferiority {'passed' if decision['accuracy_noninferiority_passed'] else 'failed'}; strict accuracy superiority {'passed' if decision['accuracy_superiority_passed'] else 'was not established'}.",
            "",
            f"CORA-FL's mapping-agnostic certificate {'passed' if decision['cora_mapping_agnostic_certificate'] else 'failed'}. The primary comparator {'has' if decision['primary_comparator_mapping_agnostic_certificate'] else 'does not have'} the same deterministic certificate.",
            "",
            f"The predeclared qualified robustness claim {'passed' if decision['qualified_claim_passed'] else 'failed'}. The structural certificate and predictive tests are separate; this decision does not imply universal accuracy dominance.",
            "",
        ]
    )
    (_paths()["derived"] / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _clear_generated_artifacts() -> None:
    """Remove only files owned by this artifact generator."""
    paths = _paths()
    patterns = {
        paths["figures"]: ("fig*.png", "fig*.pdf"),
        paths["tables"]: ("table*.csv", "table*.tex"),
        paths["plot_data"]: ("fig*.csv",),
    }
    for directory, globs in patterns.items():
        if not directory.exists():
            continue
        for pattern in globs:
            for path in directory.glob(pattern):
                path.unlink()


def generate_confirmatory_artifacts(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    index, curves = _load(config)
    paths = _paths()
    for key in ("derived", "plot_data", "figures", "tables"):
        paths[key].mkdir(parents=True, exist_ok=True)
    _clear_generated_artifacts()
    metrics = _paired_metrics(curves, index, config)
    inference, decision = _inference(metrics, config)
    certificate = _topology_certificate(int(config["experiment"]["n_clients"]))
    certificate_lookup = certificate.set_index("method")["mapping_agnostic_certificate"]
    comparator = str(decision["primary_comparator"])
    decision["cora_mapping_agnostic_certificate"] = bool(certificate_lookup.loc["cora_fl"])
    decision["primary_comparator_mapping_agnostic_certificate"] = bool(certificate_lookup.loc[comparator])
    decision["qualified_claim_passed"] = bool(
        decision["accuracy_noninferiority_passed"]
        and decision["cora_mapping_agnostic_certificate"]
        and not decision["primary_comparator_mapping_agnostic_certificate"]
    )
    metrics.to_csv(paths["derived"] / "paired_robustness.csv", index=False, float_format="%.10g")
    inference.to_csv(paths["derived"] / "confirmatory_inference.csv", index=False, float_format="%.10g")
    certificate.to_csv(paths["derived"] / "topology_certificate.csv", index=False)
    curves.to_csv(paths["derived"] / "all_curves.csv", index=False)
    write_json(paths["derived"] / "claim_decision.json", decision)
    _make_tables(metrics, index, inference, certificate, config)
    _make_figures(curves, index, metrics, config)
    _write_report(metrics, inference, decision, config)
    artifacts = {
        "plots_png": sorted(path.name for path in paths["figures"].glob("fig*.png")),
        "plots_pdf": sorted(path.name for path in paths["figures"].glob("fig*.pdf")),
        "tables_csv": sorted(path.name for path in paths["tables"].glob("table*.csv")),
        "tables_tex": sorted(path.name for path in paths["tables"].glob("table*.tex")),
    }
    write_json(paths["derived"] / "artifact_counts.json", artifacts)
    return {**artifacts, "decision": decision}
