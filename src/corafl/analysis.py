from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from .config import load_config
from .runner import repo_root
from .utils import sha256_file, write_json

METHOD_ORDER = ["fedavg", "ring_gt", "random_gossip_gt", "exponential_gt", "static_rr_gt", "adaptive_similarity_gt", "cora_fl"]
METHOD_LABEL = {"fedavg": "FedAvg (central)", "ring_gt": "Ring-GT", "random_gossip_gt": "Random-Gossip-GT", "exponential_gt": "Exponential-GT", "static_rr_gt": "Static-RR-GT", "adaptive_similarity_gt": "Similarity-Adaptive-GT", "cora_fl": "CORA-FL"}
COLORS = {"fedavg": "#222222", "ring_gt": "#D55E00", "random_gossip_gt": "#0072B2", "exponential_gt": "#56B4E9", "static_rr_gt": "#009E73", "adaptive_similarity_gt": "#CC79A7", "cora_fl": "#E69F00"}
PARTITION_LABEL = {"iid": "IID", "noniid": "Non-IID"}


def _paths() -> dict[str, Path]:
    root = repo_root()
    return {"root": root, "raw": root / "results" / "raw", "derived": root / "results" / "derived", "figures": root / "figures", "tables": root / "tables", "plot_data": root / "results" / "derived" / "plot_data"}


def _load(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _paths()
    run_index = pd.read_csv(paths["derived"] / "run_index.csv")
    selected = run_index[run_index["config_hash"] == config["_config_hash"]].copy()
    expected = len(config["experiment"]["datasets"]) * len(config["experiment"]["partitions"]) * len(config["experiment"]["scenarios"]) * len(config["experiment"]["methods"]) * len(config["experiment"]["main_seeds"])
    if len(selected) != expected or not (selected["status"] == "succeeded").all():
        raise RuntimeError(f"Found {len(selected)} succeeded indexed runs; expected {expected}")
    frames = []
    for run_id in selected["run_id"]:
        frame = pd.read_csv(paths["raw"] / f"{run_id}.csv")
        if not np.isfinite(frame[["loss", "accuracy", "gradient_norm", "total_bytes"]].to_numpy()).all():
            raise RuntimeError(f"Non-finite required value in {run_id}")
        frames.append(frame)
    return selected, pd.concat(frames, ignore_index=True)


def _mean_ci(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    records = []
    for round_id, group in frame.groupby("round", sort=True):
        values = group[value].astype(float).to_numpy()
        mean = float(np.mean(values))
        half = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        records.append({"round": float(round_id), "mean": mean, "ci_low": mean - half, "ci_high": mean + half, "n": len(values)})
    return pd.DataFrame(records)


def _setup_style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.22, "axes.spines.top": False, "axes.spines.right": False})


def _save_figure(fig: plt.Figure, stem: str) -> None:
    paths = _paths()
    paths["figures"].mkdir(parents=True, exist_ok=True)
    fig.text(0.995, 0.004, "Development benchmark; measured and unsmoothed", ha="right", va="bottom", fontsize=5.8, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(paths["figures"] / f"{stem}.png", bbox_inches="tight")
    fig.savefig(paths["figures"] / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _primary_curves(curves: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    analysis = config["analysis"]
    return curves[(curves["dataset"] == analysis["primary_dataset"]) & (curves["partition"] == analysis["primary_partition"]) & (curves["scenario"] == "one_domain")].copy()


def _line_figure(data: pd.DataFrame, metric: str, ylabel: str, title: str, stem: str, config: dict[str, Any], x_metric: str = "round", symlog_y: bool = False, include_fedavg: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 3.7))
    plot_records = []
    for method in METHOD_ORDER:
        if method == "fedavg" and not include_fedavg:
            continue
        subset = data[data["method"] == method]
        aggregate = _mean_ci(subset, metric)
        x_values = aggregate["round"].to_numpy() if x_metric == "round" else subset.groupby("round", sort=True)[x_metric].mean().to_numpy() / (1024.0**2)
        ax.plot(x_values, aggregate["mean"], label=METHOD_LABEL[method], color=COLORS[method], linewidth=1.8)
        if not symlog_y:
            ax.fill_between(x_values, aggregate["ci_low"], aggregate["ci_high"], color=COLORS[method], alpha=0.12, linewidth=0)
        export = aggregate.copy()
        export["method"] = method
        export["x_value"] = x_values
        export["metric"] = metric
        plot_records.append(export)
    if x_metric == "round":
        rounds = int(config["experiment"]["rounds"])
        ax.axvspan(rounds * float(config["topology"]["outage_start_fraction"]), rounds * float(config["topology"]["outage_end_fraction"]), color="#999999", alpha=0.12, label="domain outage")
        ax.set_xlabel("Communication round")
    else:
        ax.set_xlabel("Cumulative transmitted MiB")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if symlog_y:
        ax.set_yscale("symlog", linthresh=1e-3, linscale=0.7)
    ax.legend(ncol=2, frameon=False)
    _save_figure(fig, stem)
    pd.concat(plot_records, ignore_index=True).to_csv(_paths()["plot_data"] / f"{stem}.csv", index=False)


def _paired_robustness(curves: pd.DataFrame, run_index: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    records = []
    keys = ["dataset", "partition", "method", "seed"]
    outage_meta = run_index[run_index["scenario"] == "one_domain"].set_index(keys)
    recovery_fraction = float(config["analysis"]["recovery_fraction_of_nominal_final"])
    for key, pair in curves.groupby(keys, sort=True):
        nominal = pair[pair["scenario"] == "nominal"].sort_values("round")
        outage = pair[pair["scenario"] == "one_domain"].sort_values("round")
        if nominal.empty or outage.empty:
            raise RuntimeError(f"Missing paired scenario for {key}")
        merged = nominal[["round", "accuracy"]].merge(outage[["round", "accuracy"]], on="round", suffixes=("_nominal", "_outage"))
        meta = outage_meta.loc[key]
        outage_start, outage_end = int(meta["outage_start"]), int(meta["outage_end"])
        post = merged[merged["round"] >= outage_start]
        gap = np.maximum(post["accuracy_nominal"].to_numpy() - post["accuracy_outage"].to_numpy(), 0.0)
        target = recovery_fraction * float(nominal.iloc[-1]["accuracy"])
        recovered_rows = outage[(outage["round"] >= outage_end) & (outage["accuracy"] >= target)]
        recovered = not recovered_rows.empty
        recovery_rounds = max(0, int(recovered_rows.iloc[0]["round"]) - outage_end) if recovered else int(config["experiment"]["rounds"]) - outage_end + int(config["experiment"]["eval_every"])
        records.append({"dataset": key[0], "partition": key[1], "method": key[2], "seed": key[3], "outage_start": outage_start, "outage_end": outage_end, "positive_accuracy_loss_auc": float(np.trapezoid(gap, post["round"].to_numpy())), "peak_accuracy_drop": float(np.max(gap)), "nominal_final_accuracy": float(nominal.iloc[-1]["accuracy"]), "outage_final_accuracy": float(outage.iloc[-1]["accuracy"]), "recovery_target_accuracy": target, "recovery_rounds": recovery_rounds, "recovered": recovered})
    return pd.DataFrame(records)


def _empirical_cvar(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    return float(np.mean(ordered[: max(1, int(math.ceil((1.0 - alpha) * len(ordered))))]))


def _fmt(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _latex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]]) -> None:
    align = "l" + "c" * (len(headers) - 1)
    row_end = r" \\"
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small", f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(headers) + row_end, "\\midrule"]
    lines.extend(" & ".join(row) + row_end for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_tables(run_index: pd.DataFrame, robustness: pd.DataFrame, config: dict[str, Any]) -> None:
    paths = _paths()
    paths["tables"].mkdir(parents=True, exist_ok=True)
    outage_stats = run_index[run_index["scenario"] == "one_domain"].groupby(["method", "dataset", "partition", "scenario"])["final_accuracy"].agg(["mean", "std", "count"]).reset_index()
    outage_stats.to_csv(paths["tables"] / "table1_accuracy.csv", index=False, float_format="%.10g")
    headers = ["Method"] + [f"{dataset.replace('_', ' ').title()} {PARTITION_LABEL[partition]}" for dataset in config["experiment"]["datasets"] for partition in config["experiment"]["partitions"]]
    rows = []
    for method in METHOD_ORDER:
        row = [METHOD_LABEL[method]]
        for dataset in config["experiment"]["datasets"]:
            for partition in config["experiment"]["partitions"]:
                match = outage_stats[(outage_stats.method == method) & (outage_stats.dataset == dataset) & (outage_stats.partition == partition)].iloc[0]
                row.append(_fmt(float(match["mean"]), float(match["std"])))
        rows.append(row)
    _latex_table(paths["tables"] / "table1_accuracy.tex", "Final test accuracy under the one-domain outage. Values are mean $\\pm$ standard deviation over five locked seeds.", "tab:development_accuracy", headers, rows)

    noniid = robustness[robustness["partition"] == "noniid"]
    alpha = float(config["analysis"]["cvar_alpha"])
    robust_rows = []
    for method in METHOD_ORDER:
        group = noniid[noniid.method == method]
        robust_rows.append({"method": method, "mean_positive_accuracy_loss_auc": group.positive_accuracy_loss_auc.mean(), "std_positive_accuracy_loss_auc": group.positive_accuracy_loss_auc.std(ddof=1), f"empirical_cvar_{alpha:g}": _empirical_cvar(group.positive_accuracy_loss_auc.to_numpy(), alpha), "mean_peak_accuracy_drop": group.peak_accuracy_drop.mean(), "mean_recovery_rounds": group.recovery_rounds.mean(), "recovery_rate": group.recovered.mean(), "n_pairs": len(group)})
    robust_table = pd.DataFrame(robust_rows)
    robust_table.to_csv(paths["tables"] / "table2_robustness.csv", index=False, float_format="%.10g")
    cvar_column = f"empirical_cvar_{alpha:g}"
    latex_rows = [
        [
            METHOD_LABEL[str(row["method"])],
            f"{float(row['mean_positive_accuracy_loss_auc']):.4f}",
            f"{float(row[cvar_column]):.4f}",
            f"{float(row['mean_peak_accuracy_drop']):.4f}",
            f"{float(row['mean_recovery_rounds']):.2f}",
            f"{100.0 * float(row['recovery_rate']):.1f}\\%",
        ]
        for _, row in robust_table.iterrows()
    ]
    _latex_table(paths["tables"] / "table2_robustness.tex", "Paired non-IID outage metrics across three development datasets and five seeds. Lower is better except recovery rate.", "tab:development_robustness", ["Method", "Loss AUC", f"CVaR$_{{{alpha:g}}}$", "Peak drop", "Recovery", "Recovered"], latex_rows)

    primary = str(config["analysis"]["primary_dataset"])
    cost = run_index[(run_index.dataset == primary) & (run_index.partition == "noniid") & (run_index.scenario == "one_domain")].copy()
    cost["total_mib"] = cost.total_bytes / (1024.0**2)
    cost["control_kib"] = cost.control_bytes / 1024.0
    cost_table = cost.groupby("method").agg(total_mib_mean=("total_mib", "mean"), total_mib_std=("total_mib", "std"), control_kib_mean=("control_kib", "mean"), messages_mean=("messages", "mean"), successful_edges_mean=("successful_edges", "mean"), consensus_error_mean=("final_consensus_error", "mean"), wall_time_seconds_mean=("wall_time_seconds", "mean"), n_runs=("run_id", "count")).reindex(METHOD_ORDER).reset_index()
    cost_table.to_csv(paths["tables"] / "table3_cost.csv", index=False, float_format="%.10g")
    cost_rows = [[METHOD_LABEL[row.method], _fmt(row.total_mib_mean, row.total_mib_std, 3), f"{row.control_kib_mean:.2f}", f"{row.messages_mean:.1f}", f"{row.successful_edges_mean:.1f}", f"{row.consensus_error_mean:.3e}"] for row in cost_table.itertuples(index=False)]
    _latex_table(paths["tables"] / "table3_cost.tex", f"Communication and consensus accounting for {primary.replace('_', ' ').title()} non-IID one-domain runs over five seeds.", "tab:development_cost", ["Method", "Total MiB", "Control KiB", "Messages", "Edges", "Consensus"], cost_rows)


def _make_figures(curves: pd.DataFrame, run_index: pd.DataFrame, robustness: pd.DataFrame, config: dict[str, Any]) -> None:
    _setup_style()
    paths = _paths()
    paths["plot_data"].mkdir(parents=True, exist_ok=True)
    primary = _primary_curves(curves, config)
    _line_figure(primary, "accuracy", "Test accuracy", "Digits, non-IID, one-domain outage", "fig1_accuracy_vs_round", config)
    _line_figure(primary, "accuracy", "Test accuracy", "Accuracy per transmitted byte", "fig2_accuracy_vs_bytes", config, x_metric="total_bytes")
    _line_figure(primary, "consensus_error", "Consensus error (symlog)", "Peer-model disagreement during the outage", "fig3_consensus_vs_round", config, symlog_y=True, include_fedavg=False)
    final = run_index[run_index.scenario == "one_domain"]
    columns = [(d, p) for d in config["experiment"]["datasets"] for p in config["experiment"]["partitions"]]
    heat = np.zeros((len(METHOD_ORDER), len(columns)))
    for i, method in enumerate(METHOD_ORDER):
        for j, (dataset, partition) in enumerate(columns):
            heat[i, j] = final[(final.method == method) & (final.dataset == dataset) & (final.partition == partition)].final_accuracy.mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    image = ax.imshow(heat, vmin=max(0.75, float(np.nanmin(heat)) - 0.01), vmax=1.0, cmap="YlGnBu", aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white" if heat[i, j] > 0.94 else "black")
    ax.set_xticks(range(len(columns)), [f"{d.replace('_', ' ').title()}\n{PARTITION_LABEL[p]}" for d, p in columns], rotation=25, ha="right")
    ax.set_yticks(range(len(METHOD_ORDER)), [METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.set_title("Final accuracy under the one-domain outage")
    fig.colorbar(image, ax=ax, label="Mean test accuracy")
    _save_figure(fig, "fig4_dataset_partition_heatmap")
    pd.DataFrame(heat, index=METHOD_ORDER, columns=[f"{d}__{p}" for d, p in columns]).to_csv(paths["plot_data"] / "fig4_dataset_partition_heatmap.csv")
    seed_macro = final.groupby(["method", "partition", "seed"], as_index=False).final_accuracy.mean().rename(columns={"final_accuracy": "macro_accuracy"})
    part_summary = seed_macro.groupby(["method", "partition"]).macro_accuracy.agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    x, width = np.arange(len(METHOD_ORDER)), 0.36
    for offset, partition in [(-width / 2, "iid"), (width / 2, "noniid")]:
        subset = part_summary.set_index(["method", "partition"]).loc[[(m, partition) for m in METHOD_ORDER]]
        ci = t.ppf(0.975, subset["count"] - 1) * subset["std"] / np.sqrt(subset["count"])
        ax.bar(x + offset, subset["mean"], width, yerr=ci, capsize=2, label=PARTITION_LABEL[partition], color="#4C78A8" if partition == "iid" else "#F58518", alpha=0.88)
    ax.set_xticks(x, [METHOD_LABEL[m] for m in METHOD_ORDER], rotation=25, ha="right")
    ax.set_ylim(max(0.75, part_summary["mean"].min() - 0.05), 1.0)
    ax.set_ylabel("Final test accuracy")
    ax.set_title("IID and non-IID generalizability under one-domain outage")
    ax.legend(frameon=False)
    _save_figure(fig, "fig5_iid_vs_noniid")
    part_summary.to_csv(paths["plot_data"] / "fig5_iid_vs_noniid.csv", index=False)
    alpha = float(config["analysis"]["cvar_alpha"])
    noniid = robustness[robustness.partition == "noniid"]
    robust_plot = pd.DataFrame([{"method": m, "mean": noniid[noniid.method == m].positive_accuracy_loss_auc.mean(), "cvar": _empirical_cvar(noniid[noniid.method == m].positive_accuracy_loss_auc.to_numpy(), alpha)} for m in METHOD_ORDER])
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    ax.bar(x - 0.18, robust_plot["mean"], 0.36, label="Mean", color="#4C78A8")
    ax.bar(x + 0.18, robust_plot["cvar"], 0.36, label=f"Empirical CVaR$_{{{alpha:g}}}$", color="#E45756")
    ax.set_xticks(x, [METHOD_LABEL[m] for m in METHOD_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Positive accuracy-loss AUC")
    ax.set_title("Paired outage loss; lower is better")
    ax.legend(frameon=False)
    _save_figure(fig, "fig6_robustness_auc_cvar")
    robust_plot.to_csv(paths["plot_data"] / "fig6_robustness_auc_cvar.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    box = ax.boxplot([noniid[noniid.method == m].recovery_rounds.to_numpy() for m in METHOD_ORDER], tick_labels=[METHOD_LABEL[m] for m in METHOD_ORDER], patch_artist=True, showmeans=True)
    for patch, method in zip(box["boxes"], METHOD_ORDER):
        patch.set_facecolor(COLORS[method]); patch.set_alpha(0.65)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Recovery rounds")
    ax.set_title("Recovery to 95% of paired nominal final accuracy")
    _save_figure(fig, "fig7_recovery_rounds")
    noniid[["dataset", "method", "seed", "recovery_rounds", "recovered"]].to_csv(paths["plot_data"] / "fig7_recovery_rounds.csv", index=False)
    _line_figure(primary, "worst_domain_metric", "Worst-domain test accuracy", "Geographic-domain tail performance", "fig8_worst_domain_accuracy", config)


def _write_results_report(run_index: pd.DataFrame, robustness: pd.DataFrame, config: dict[str, Any]) -> None:
    subset = run_index[(run_index.partition == "noniid") & (run_index.scenario == "one_domain")]
    macro = subset.groupby("method").final_accuracy.mean().reindex(METHOD_ORDER)
    robust = robustness[robustness.partition == "noniid"].groupby("method").positive_accuracy_loss_auc.mean().reindex(METHOD_ORDER)
    accuracy_pairs = subset.pivot(index=["dataset", "seed"], columns="method", values="final_accuracy")
    d_acc = (accuracy_pairs.cora_fl - accuracy_pairs.random_gossip_gt).to_numpy()
    h_acc = float(t.ppf(0.975, len(d_acc) - 1) * np.std(d_acc, ddof=1) / math.sqrt(len(d_acc)))
    robust_pairs = robustness[robustness.partition == "noniid"].pivot(index=["dataset", "seed"], columns="method", values="positive_accuracy_loss_auc")
    d_auc = (robust_pairs.cora_fl - robust_pairs.random_gossip_gt).to_numpy()
    h_auc = float(t.ppf(0.975, len(d_auc) - 1) * np.std(d_auc, ddof=1) / math.sqrt(len(d_auc)))
    lines = ["# CORA-FL development benchmark: exact generated report", "", "> Scope: algorithmic development benchmark on three scikit-learn bundled classification datasets. These values are not smart-city traffic results.", "", f"- Completed runs: {len(run_index)}; failed runs: 0.", f"- Main seeds: {config['experiment']['main_seeds']}.", f"- Partitions: IID and Dirichlet non-IID (alpha = {config['partition']['dirichlet_alpha']}).", "- Outage: one declared domain absent from update rounds 35 to 64 inclusive, with 5% independent attempted-link loss.", "- Curves are unsmoothed; bands, where shown, are 95% t confidence intervals.", "", "## Macro-average final accuracy: non-IID, one-domain outage", "", "| Method | Accuracy | Mean positive accuracy-loss AUC |", "|---|---:|---:|"]
    for method in METHOD_ORDER:
        lines.append(f"| {METHOD_LABEL[method]} | {macro[method]:.6f} | {robust[method]:.6f} |")
    lines.extend(["", "## Guarded interpretation", "", f"CORA-FL minus Random-Gossip-GT accuracy: {np.mean(d_acc):+.6f}, 95% t interval [{np.mean(d_acc)-h_acc:+.6f}, {np.mean(d_acc)+h_acc:+.6f}].", "", f"CORA-FL minus Random-Gossip-GT loss AUC: {np.mean(d_auc):+.6f}, 95% t interval [{np.mean(d_auc)-h_auc:+.6f}, {np.mean(d_auc)+h_auc:+.6f}]. Lower AUC is better.", "", "These development results identify tradeoffs and do not establish universal superiority.", ""])
    (_paths()["derived"] / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def generate_artifacts(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    run_index, curves = _load(config)
    robustness = _paired_robustness(curves, run_index, config)
    paths = _paths()
    paths["derived"].mkdir(parents=True, exist_ok=True)
    robustness.to_csv(paths["derived"] / "paired_robustness.csv", index=False, float_format="%.10g")
    curves.to_csv(paths["derived"] / "all_curves.csv", index=False)
    _make_tables(run_index, robustness, config)
    _make_figures(curves, run_index, robustness, config)
    _write_results_report(run_index, robustness, config)
    artifacts = {"plots": sorted(path.name for path in paths["figures"].glob("fig*.png")), "plot_pdfs": sorted(path.name for path in paths["figures"].glob("fig*.pdf")), "tables_csv": sorted(path.name for path in paths["tables"].glob("table*.csv")), "tables_tex": sorted(path.name for path in paths["tables"].glob("table*.tex"))}
    write_json(paths["derived"] / "artifact_counts.json", artifacts)
    return artifacts


def artifact_checksums() -> dict[str, str]:
    paths = _paths()
    candidates = [*sorted(paths["figures"].glob("fig*.*")), *sorted(paths["tables"].glob("table*.*")), paths["derived"] / "paired_robustness.csv", paths["derived"] / "RESULTS.md", paths["derived"] / "certificate.json", paths["derived"] / "completion.json"]
    return {str(path.relative_to(paths["root"])): sha256_file(path) for path in candidates if path.is_file()}
