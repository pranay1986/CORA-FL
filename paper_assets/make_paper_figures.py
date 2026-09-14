"""Render compact paper figures from the locked, exported plot CSV files.

The script never reads raw test predictions and never recomputes reported values.
It changes presentation only. Every plotted coordinate comes from
results/derived/plot_data in the confirmatory artifact.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "derived" / "plot_data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "fedavg",
    "ring_gt",
    "random_gossip_gt",
    "exponential_gt",
    "matcha_gt",
    "adaptive_similarity_gt",
    "cora_fl",
]
LABEL = {
    "fedavg": "FedAvg",
    "ring_gt": "Ring-GT",
    "random_gossip_gt": "Gossip-GT",
    "exponential_gt": "Exp.-GT",
    "matcha_gt": "MATCHA-GT",
    "adaptive_similarity_gt": "Similarity-GT",
    "cora_fl": "CORA-FL",
}
COLOR = {
    "fedavg": "#222222",
    "ring_gt": "#D55E00",
    "random_gossip_gt": "#0072B2",
    "exponential_gt": "#56B4E9",
    "matcha_gt": "#009E73",
    "adaptive_similarity_gt": "#CC79A7",
    "cora_fl": "#E69F00",
}
MARKER = {
    "fedavg": "o",
    "ring_gt": "s",
    "random_gossip_gt": "^",
    "exponential_gt": "v",
    "matcha_gt": "D",
    "adaptive_similarity_gt": "P",
    "cora_fl": "*",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.2,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def add_outage(ax: plt.Axes) -> None:
    ax.axvspan(35, 65, color="#888888", alpha=0.12, linewidth=0)


def line_plot(
    csv_name: str,
    stem: str,
    ylabel: str,
    *,
    x_label: str,
    x_col: str = "x_value",
    include_central: bool = True,
    symlog: bool = False,
    legend: bool = True,
) -> None:
    frame = pd.read_csv(DATA / csv_name)
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for method in METHODS:
        if method == "fedavg" and not include_central:
            continue
        part = frame[frame.method == method].sort_values("round")
        ax.plot(
            part[x_col],
            part["mean"],
            color=COLOR[method],
            marker=MARKER[method],
            markevery=4,
            markersize=2.5,
            linewidth=1.15,
            label=LABEL[method],
        )
        if not symlog:
            ax.fill_between(part[x_col], part.ci_low, part.ci_high, color=COLOR[method], alpha=0.09, linewidth=0)
    if x_col == "x_value" and x_label == "Communication round":
        add_outage(ax)
    if symlog:
        ax.set_yscale("symlog", linthresh=1e-3, linscale=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    if legend:
        ncols = 4 if include_central else 3
        ax.legend(
            ncol=ncols,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            handlelength=1.4,
            columnspacing=0.65,
            borderaxespad=0.0,
        )
    fig.tight_layout(pad=0.25)
    save(fig, stem)


def plot_heatmap() -> None:
    heat = pd.read_csv(DATA / "fig4_dataset_heatmap.csv", index_col=0).reindex(METHODS)
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    image = ax.imshow(heat.to_numpy(), vmin=0.91, vmax=0.99, cmap="YlGnBu", aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = float(heat.iloc[i, j])
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.0, color="white" if value > 0.95 else "black")
    ax.set_xticks(range(3), ["Digits", "Breast cancer", "Wine"])
    ax.set_yticks(range(len(METHODS)), [LABEL[m] for m in METHODS])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.ax.tick_params(labelsize=5.8)
    fig.tight_layout(pad=0.2)
    save(fig, "plot4_dataset_heatmap")


def plot_partitions() -> None:
    frame = pd.read_csv(DATA / "fig5_partition_generalizability.csv")
    partitions = ["iid", "noniid", "noniid_domain"]
    part_label = {"iid": "IID", "noniid": r"Dirichlet $\alpha=0.3$", "noniid_domain": "Domain-correlated"}
    palette = ["#4C78A8", "#F58518", "#54A24B"]
    x = np.arange(len(METHODS))
    width = 0.78 / len(partitions)
    fig, ax = plt.subplots(figsize=(3.8, 2.55))
    for pos, partition in enumerate(partitions):
        selected = frame[frame.partition == partition].set_index("method").reindex(METHODS)
        error = t.ppf(0.975, selected["count"] - 1) * selected["std"] / np.sqrt(selected["count"])
        offset = (pos - 1) * width
        ax.bar(x + offset, selected["mean"], width, yerr=error, capsize=1.2, linewidth=0.4, color=palette[pos], label=part_label[partition])
    ax.set_xticks(x, [LABEL[m] for m in METHODS], rotation=30, ha="right")
    ax.set_ylim(0.915, 0.991)
    ax.set_ylabel("Macro final accuracy")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.6,
        handlelength=1.2,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.25)
    save(fig, "plot5_partitions")


def plot_robustness() -> None:
    frame = pd.read_csv(DATA / "fig6_outage_utility.csv", header=[0, 1], index_col=0).reindex(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35))
    specs = [
        ("outage_accuracy_auc", r"$U_A$", "Outage utility, higher is better", (0.91, 0.985)),
        ("positive_accuracy_loss_auc", r"$D_+$", "Positive degradation, lower is better", (0.0, 1.02)),
    ]
    x = np.arange(len(METHODS))
    for ax, (metric, ylabel, title, ylim) in zip(axes, specs):
        means = frame[(metric, "mean")].to_numpy(dtype=float)
        counts = frame[(metric, "count")].to_numpy(dtype=float)
        errors = t.ppf(0.975, counts - 1) * frame[(metric, "std")].to_numpy(dtype=float) / np.sqrt(counts)
        ax.bar(x, means, yerr=errors, capsize=1.5, color=[COLOR[m] for m in METHODS], linewidth=0.3)
        ax.set_xticks(x, [LABEL[m] for m in METHODS], rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3, w_pad=1.1)
    save(fig, "plot6_robustness")


def plot_pareto() -> None:
    frame = pd.read_csv(DATA / "fig7_pareto_accuracy_bytes.csv", index_col=0).reindex(METHODS)
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    for method, row in frame.iterrows():
        xerr = 1.96 * float(row.total_mib_std) / np.sqrt(float(row.n))
        yerr = 1.96 * float(row.accuracy_std) / np.sqrt(float(row.n))
        ax.errorbar(float(row.total_mib), float(row.accuracy), xerr=xerr, yerr=yerr, fmt=MARKER[method], color=COLOR[method], capsize=1.5, markersize=4.0)
        offsets = {
            "fedavg": (4, 8),
            "ring_gt": (-8, 8),
            "random_gossip_gt": (4, -12),
            "exponential_gt": (-11, -12),
            "matcha_gt": (-4, 9),
            "adaptive_similarity_gt": (-34, 7),
            "cora_fl": (-30, 8),
        }
        ax.annotate(
            LABEL[method],
            (float(row.total_mib), float(row.accuracy)),
            xytext=offsets[method],
            textcoords="offset points",
            fontsize=5.4,
        )
    ax.set_xlabel("Total transmitted MiB")
    ax.set_ylabel("Final outage accuracy")
    ax.set_xlim(2.58, 2.9)
    ax.set_ylim(0.951, 0.978)
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.25)
    save(fig, "plot7_pareto")


def main() -> None:
    style()
    line_plot("fig1_accuracy_vs_round.csv", "plot1_accuracy_round", "Test accuracy", x_label="Communication round")
    line_plot("fig2_accuracy_vs_bytes.csv", "plot2_accuracy_bytes", "Test accuracy", x_label="Cumulative transmitted MiB")
    line_plot("fig3_consensus_vs_round.csv", "plot3_consensus", "Consensus error", x_label="Communication round", include_central=False, symlog=True)
    plot_heatmap()
    plot_partitions()
    plot_robustness()
    plot_pareto()
    line_plot("fig8_worst_domain_accuracy.csv", "plot8_worst_domain", "Worst-domain accuracy", x_label="Communication round")


if __name__ == "__main__":
    main()
