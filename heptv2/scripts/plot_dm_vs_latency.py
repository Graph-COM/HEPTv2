"""Plot detector matching score versus inference latency for baseline models.

The values are taken from the main-model comparison table in the paper draft.

Usage:
    python heptv2/scripts/plot_dm_vs_latency.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter


MODELS = [
    {
        "name": "ACORN-GNN",
        "dm": 97.5,
        "fake_rate": 0.9,
        "latency_ms": 783.7,
        "memory_gib": 16.6,
        "marker": "o",
    },
    {
        "name": "OC-GNN",
        "dm": 96.4,
        "fake_rate": 0.9,
        "latency_ms": 571.5,
        "memory_gib": 5.4,
        "marker": "s",
    },
    {
        "name": "Two-stage MF",
        "dm": 94.1,
        "fake_rate": 0.7,
        "latency_ms": 99,
        "memory_gib": None,
        "marker": "^",
    },
    {
        "name": "HEPT+DBSCAN",
        "dm": 89.6,
        "fake_rate": 3.3,
        "latency_ms": 105.5,
        "memory_gib": 7.6,
        "marker": "D",
    },
    {
        "name": "HEPTv2",
        "dm": 98.6,
        "fake_rate": 0.8,
        "latency_ms": 15.1,
        "memory_gib": 0.4,
        "marker": "*",
    },
]

LABEL_OFFSETS = {
    "ACORN-GNN": (10, 12, "right"),
    "OC-GNN": (0, -15, "center"),
    "Two-stage MF": (0, -15, "center"),
    "HEPT+DBSCAN": (0, 15, "center"),
    "HEPTv2": (9, 7, "left"),
}


def plot_dm_vs_latency() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 16,
            "axes.labelsize": 16.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(3.55, 2.9))

    for model in MODELS:
        is_heptv2 = model["name"] == "HEPTv2"

        tune_s = 175 if is_heptv2 else 72
        if model["name"] == "Two-stage MF":
            tune_s = 95
        if model["name"] == "ACORN-GNN":
            tune_s = 95

        ax.scatter(
            model["latency_ms"],
            model["dm"],
            s=tune_s,
            marker=model["marker"],
            color="#D55E00" if is_heptv2 else "#0072B2",
            edgecolor="black",
            linewidth=0.9,
            zorder=4 if is_heptv2 else 3,
        )

        dx, dy, ha = LABEL_OFFSETS[model["name"]]
        ax.annotate(
            model["name"],
            xy=(model["latency_ms"], model["dm"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=12.8,
        )

    ax.set_xscale("log")
    ax.set_xlim(10, 1000)
    ax.set_ylim(88.8, 99.2)
    ax.set_yticks([90.0, 92.5, 95.0, 97.5])
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel(r"$\epsilon^{\mathrm{DM}}_{p_T>0.9}$ (%)")

    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(axis="both", which="major", labelsize=12.5, width=0.85, length=4.5)
    ax.tick_params(axis="x", which="minor", width=0.55, length=2.5)
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        color="#b8b8b8",
        linewidth=0.75,
        linestyle=(0, (2.4, 2.4)),
        alpha=0.85,
    )
    ax.grid(
        True,
        which="minor",
        axis="x",
        color="#d8d8d8",
        linewidth=0.45,
        linestyle=(0, (2.4, 2.4)),
        alpha=0.55,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    fig.tight_layout(pad=0.15)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated figure files.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="dm_vs_latency",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png", "svg"],
        choices=["pdf", "png", "svg"],
        help="Figure formats to write.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_dm_vs_latency()
    for fmt in args.formats:
        out_path = args.out_dir / f"{args.stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = 300
        fig.savefig(out_path, **save_kwargs)
        print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
