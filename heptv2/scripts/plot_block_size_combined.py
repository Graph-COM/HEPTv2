"""Plot block-size DM and encoder-cost ablations side by side.

Usage:
    python heptv2/scripts/plot_block_size_combined.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


BLOCK_SIZE = np.array([128, 256, 512, 1024])
LATENCY_MS = np.array([6.578, 6.595, 7.323, 8.568])
FLOPS_REL = np.array([1.00, 1.34, 2.03, 3.38])
LATENCY_REL_1024 = LATENCY_MS / LATENCY_MS[-1]
FLOPS_REL_1024 = FLOPS_REL / FLOPS_REL[-1]

ORANGE = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
GRAY = "#6A737D"
LATENCY_COLOR = "#C44E52"

DM_SERIES = [
    {
        "label": "HEPTv2 (Full)",
        "dm": [98.3, 98.5, 98.5, 98.6],
        "color": ORANGE,
        "marker": "*",
        "markersize": 12.0,
        "linestyle": "-",
    },
    {
        "label": "HEPTv2 (Z-H)",
        "dm": [95.2, 95.8, 96.5, 97.4],
        "color": "#4D4D4D",
        "marker": "*",
        "markersize": 12.0,
        "linestyle": "-",
    },
    {
        "label": r"HEPTv2 ($\phi$)",
        "dm": [92.7, 95.5, 96.8, 97.6],
        "color": GREEN,
        "marker": "*",
        "markersize": 12.0,
        "linestyle": "-",
    },
    {
        "label": r"Two-stage MF ($\phi$)",
        "block_size": [1024],
        "dm": [94.1],
        "color": BLUE,
        "marker": "^",
        "markersize": 10.0,
        "linestyle": "None",
        "markerfacecolor": "white",
        "markeredgecolor": BLUE,
        "markeredgewidth": 1.45,
    },
]


def _setup_style() -> None:
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
            "legend.fontsize": 10.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _format_block_axis(ax: plt.Axes, *, reverse: bool = False) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlim((1200, 110) if reverse else (110, 1200))
    ax.set_xticks(BLOCK_SIZE)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="both", which="major", labelsize=12.5, width=0.85, length=4.5)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        color="#b8b8b8",
        linewidth=0.75,
        linestyle=(0, (2.4, 2.4)),
        alpha=0.85,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)


def _plot_dm(ax: plt.Axes) -> None:
    for series in DM_SERIES:
        block_size = series.get("block_size", BLOCK_SIZE)
        ax.plot(
            block_size,
            series["dm"],
            label=series["label"],
            color=series["color"],
            marker=series["marker"],
            linestyle=series["linestyle"],
            linewidth=1.45,
            markersize=series["markersize"],
            markerfacecolor=series.get("markerfacecolor", series["color"]),
            markeredgecolor=series.get("markeredgecolor", "black"),
            markeredgewidth=series.get("markeredgewidth", 0.65),
            zorder=4 if series["label"] == "HEPTv2 (Full)" else 3,
        )

    _format_block_axis(ax, reverse=True)
    ax.set_ylim(92.2, 99.0)
    ax.set_xlabel("Encoder Block Size")
    ax.set_ylabel(r"$\epsilon^{\mathrm{DM}}_{p_T>0.9}$ (%)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.25,
        columnspacing=0.75,
        handletextpad=0.3,
        borderaxespad=0.0,
    )


def _plot_cost(ax: plt.Axes) -> None:
    ax.plot(
        BLOCK_SIZE,
        FLOPS_REL_1024,
        color=ORANGE,
        marker="*",
        linestyle=(0, (6, 2.2)),
        linewidth=1.65,
        markersize=12.0,
        markeredgecolor="black",
        markeredgewidth=0.65,
        label="FLOPs",
        zorder=4,
    )
    ax.plot(
        BLOCK_SIZE,
        LATENCY_REL_1024,
        color=ORANGE,
        marker="*",
        markevery=[0, 1, 2],
        linestyle=(0, (1.2, 1.8)),
        linewidth=1.65,
        markersize=12.0,
        markerfacecolor="white",
        markeredgecolor=ORANGE,
        markeredgewidth=1.35,
        label="Latency",
        zorder=4,
    )
    ax.fill_between(
        BLOCK_SIZE,
        FLOPS_REL_1024,
        LATENCY_REL_1024,
        color=GRAY,
        alpha=0.14,
        zorder=2,
    )

    _format_block_axis(ax, reverse=True)
    ax.set_ylim(0.2, 1.08)
    ax.set_xlabel("Encoder Block Size")
    ax.set_ylabel("Relative Cost (vs. 1024)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.35,
        columnspacing=1.1,
        borderaxespad=0.0,
    )


def plot_block_size_combined() -> plt.Figure:
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.35))
    _plot_dm(axes[0])
    _plot_cost(axes[1])
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.18, top=0.80, wspace=0.36)
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
        default="dm_and_cost_vs_block_size",
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
    fig = plot_block_size_combined()
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
