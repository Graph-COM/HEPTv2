"""Generate quick-look options for encoder cost versus block size.

Usage:
    python heptv2/scripts/plot_block_size_cost_options.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


BLOCK_SIZE = np.array([128, 256, 512, 1024])
MEASURED_BLOCK_SIZE = np.array([129, 255, 513, 1023])
LATENCY_MS = np.array([6.578, 6.595, 7.323, 8.568])
FLOPS_REL = np.array([1.00, 1.34, 2.03, 3.38])
HEPTV2_DM = np.array([98.3, 98.5, 98.5, 98.6])

LATENCY_REL_128 = LATENCY_MS / LATENCY_MS[0]
FLOPS_REL_128 = FLOPS_REL / FLOPS_REL[0]
LATENCY_REL_1024 = LATENCY_MS / LATENCY_MS[-1]
FLOPS_REL_1024 = FLOPS_REL / FLOPS_REL[-1]
LATENCY_REDUCTION_1024 = 100.0 * (1.0 - LATENCY_MS / LATENCY_MS[-1])
FLOPS_REDUCTION_1024 = 100.0 * (1.0 - FLOPS_REL / FLOPS_REL[-1])

ORANGE = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
GRAY = "#6A737D"
PURPLE = "#CC79A7"
LATENCY_COLOR = "#C44E52"
FIGSIZE = (3.65, 3.35)
PANEL_ADJUST = {
    "left": 0.23,
    "right": 0.98,
    "bottom": 0.19,
    "top": 0.78,
}


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
            "legend.fontsize": 10.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _format_block_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlim(110, 1200)
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


def option_1_latency_with_flops_labels() -> plt.Figure:
    """Single-axis latency line with FLOPs annotations."""
    fig, ax = plt.subplots(figsize=(4.05, 3.55))
    ax.plot(
        BLOCK_SIZE,
        LATENCY_MS,
        color=BLUE,
        marker="o",
        linewidth=1.65,
        markersize=7.35,
        markeredgecolor="black",
        markeredgewidth=0.65,
        zorder=3,
    )
    for x, y, flops in zip(BLOCK_SIZE, LATENCY_MS, FLOPS_REL):
        ax.annotate(
            f"{flops:.2f}x",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.8,
        )
    _format_block_axis(ax)
    ax.set_ylim(6.35, 8.9)
    ax.set_xlabel("Encoder Block Size")
    ax.set_ylabel("Encoder Latency p50 (ms)")
    fig.tight_layout(pad=0.15)
    return fig


def option_2_dual_axis() -> plt.Figure:
    """Dual-axis latency and relative FLOPs."""
    fig, ax_latency = plt.subplots(figsize=(3.55, 3.05))
    latency_line = ax_latency.plot(
        BLOCK_SIZE,
        LATENCY_MS,
        color=BLUE,
        marker="o",
        linewidth=1.65,
        markersize=7.35,
        markeredgecolor="black",
        markeredgewidth=0.65,
        label="Latency",
        zorder=3,
    )[0]
    _format_block_axis(ax_latency)
    ax_latency.set_ylim(6.35, 8.9)
    ax_latency.set_xlabel("Encoder Block Size")
    ax_latency.set_ylabel("Latency p50 (ms)", color=BLUE)
    ax_latency.tick_params(axis="y", colors=BLUE)

    ax_flops = ax_latency.twinx()
    flops_line = ax_flops.plot(
        BLOCK_SIZE,
        FLOPS_REL,
        color=PURPLE,
        marker="D",
        linewidth=1.5,
        markersize=7.0,
        markeredgecolor="black",
        markeredgewidth=0.65,
        label="FLOPs",
        zorder=4,
    )[0]
    ax_flops.set_ylim(0.75, 3.65)
    ax_flops.set_ylabel("Encoder FLOPs (rel.)", color=PURPLE)
    ax_flops.tick_params(axis="y", colors=PURPLE, labelsize=12.5, width=0.85, length=4.5)
    ax_flops.spines["top"].set_visible(False)
    ax_flops.spines["right"].set_linewidth(0.8)
    ax_flops.legend(
        [latency_line, flops_line],
        ["Latency", "FLOPs"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.35,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.15)
    return fig


def option_3_two_panel_cost() -> plt.Figure:
    """Two stacked panels without a dual y-axis."""
    fig, (ax_latency, ax_flops) = plt.subplots(
        2,
        1,
        figsize=(3.55, 3.55),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.85], "hspace": 0.08},
    )
    ax_latency.plot(
        BLOCK_SIZE,
        LATENCY_MS,
        color=BLUE,
        marker="o",
        linewidth=1.55,
        markersize=6.8,
        markeredgecolor="black",
        markeredgewidth=0.65,
    )
    ax_latency.set_ylim(6.35, 8.9)
    ax_latency.set_ylabel("Latency\np50 (ms)")

    ax_flops.bar(
        BLOCK_SIZE,
        FLOPS_REL,
        width=70,
        color=GREEN,
        edgecolor="black",
        linewidth=0.65,
        zorder=3,
    )
    ax_flops.set_ylim(0.0, 3.75)
    ax_flops.set_ylabel("FLOPs\n(rel.)")
    ax_flops.set_xlabel("Encoder Block Size")
    for x, y in zip(BLOCK_SIZE, FLOPS_REL):
        ax_flops.text(x, y + 0.12, f"{y:.2f}x", ha="center", va="bottom", fontsize=9.6)

    for ax in (ax_latency, ax_flops):
        _format_block_axis(ax)
    ax_latency.tick_params(labelbottom=False)
    fig.tight_layout(pad=0.15)
    return fig


def option_4_dm_latency_tradeoff() -> plt.Figure:
    """Quality-cost tradeoff for the HEPTv2 block-size sweep."""
    fig, ax = plt.subplots(figsize=(3.55, 3.05))
    ax.plot(
        LATENCY_MS,
        HEPTV2_DM,
        color=ORANGE,
        marker="*",
        linewidth=1.45,
        markersize=12.0,
        markeredgecolor="black",
        markeredgewidth=0.65,
        zorder=3,
    )
    for latency, dm, block in zip(LATENCY_MS, HEPTV2_DM, BLOCK_SIZE):
        ax.annotate(
            f"{block}",
            xy=(latency, dm),
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10.5,
        )
    ax.set_xlim(6.35, 8.85)
    ax.set_ylim(98.18, 98.72)
    ax.set_xlabel("Encoder Latency p50 (ms)")
    ax.set_ylabel(r"$\epsilon^{\mathrm{DM}}_{p_T>0.9}$ (%)")
    ax.tick_params(axis="both", which="major", labelsize=12.5, width=0.85, length=4.5)
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
    fig.tight_layout(pad=0.15)
    return fig


def option_5_relative_growth_gap() -> plt.Figure:
    """Relative cost versus block size, normalized to the largest block size."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
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
    _format_block_axis(ax)
    ax.set_xlim(1200, 110)
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
    fig.subplots_adjust(**PANEL_ADJUST)
    return fig


def option_6_reduction_bars() -> plt.Figure:
    """Grouped bars: relative cost normalized to block size 1024."""
    fig, ax = plt.subplots(figsize=(3.55, 3.05))
    x = np.arange(len(BLOCK_SIZE))
    labels = [str(block) for block in BLOCK_SIZE]
    width = 0.34
    flops_cost = FLOPS_REL_1024
    latency_cost = LATENCY_REL_1024
    ax.bar(
        x - width / 2,
        flops_cost,
        width,
        color=GREEN,
        edgecolor="black",
        linewidth=0.65,
        label="FLOPs",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        latency_cost,
        width,
        color=BLUE,
        edgecolor="black",
        linewidth=0.65,
        label="Latency",
        zorder=3,
    )
    for idx, (xpos, value) in enumerate(zip(x - width / 2, flops_cost)):
        ha = "right" if idx == len(x) - 1 else "center"
        ax.text(xpos, value + 0.035, f"{value:.2f}x", ha=ha, va="bottom", fontsize=9.0)
    for idx, (xpos, value) in enumerate(zip(x + width / 2, latency_cost)):
        ha = "left" if idx == len(x) - 1 else "center"
        ax.text(xpos, value + 0.035, f"{value:.2f}x", ha=ha, va="bottom", fontsize=9.0)

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.16)
    ax.set_xlabel("Encoder Block Size")
    ax.set_ylabel("Relative Cost (vs. 1024)")
    ax.tick_params(axis="both", which="major", labelsize=12.5, width=0.85, length=4.5)
    ax.set_axisbelow(True)
    ax.grid(
        True,
        axis="y",
        color="#b8b8b8",
        linewidth=0.75,
        linestyle=(0, (2.4, 2.4)),
        alpha=0.85,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.35,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    fig.tight_layout(pad=0.15)
    return fig


def option_7_savings_dumbbell() -> plt.Figure:
    """Dumbbell plot of FLOPs reduction and latency reduction."""
    fig, ax = plt.subplots(figsize=(3.55, 3.05))
    labels = ["512", "256", "128"]
    y = np.arange(len(labels))
    latency_saved = LATENCY_REDUCTION_1024[[2, 1, 0]]
    flops_saved = FLOPS_REDUCTION_1024[[2, 1, 0]]
    for ypos, latency, flops in zip(y, latency_saved, flops_saved):
        ax.hlines(
            ypos,
            latency,
            flops,
            color=GRAY,
            linewidth=1.35,
            linestyle=(0, (2.4, 2.4)),
            zorder=2,
        )
        ax.text(
            flops + 2.2,
            ypos,
            f"+{flops - latency:.0f} pt gap",
            ha="left",
            va="center",
            fontsize=9.6,
        )
    ax.scatter(
        latency_saved,
        y,
        color=BLUE,
        marker="o",
        s=58,
        edgecolor="black",
        linewidth=0.65,
        label="Latency",
        zorder=3,
    )
    ax.scatter(
        flops_saved,
        y,
        color=GREEN,
        marker="D",
        s=58,
        edgecolor="black",
        linewidth=0.65,
        label="FLOPs",
        zorder=3,
    )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 82)
    ax.set_ylim(-0.55, len(labels) - 0.45)
    ax.set_xlabel("Reduction vs. 1024 (%)")
    ax.set_ylabel("Encoder Block Size")
    ax.tick_params(axis="both", which="major", labelsize=12.5, width=0.85, length=4.5)
    ax.set_axisbelow(True)
    ax.grid(
        True,
        axis="x",
        color="#b8b8b8",
        linewidth=0.75,
        linestyle=(0, (2.4, 2.4)),
        alpha=0.85,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.35,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    fig.tight_layout(pad=0.15)
    return fig


def option_8_flops_latency_scaling_scatter() -> plt.Figure:
    """Scatter of FLOPs growth versus latency growth."""
    fig, ax = plt.subplots(figsize=(3.55, 3.05))
    ax.plot(
        FLOPS_REL_128,
        LATENCY_REL_128,
        color=ORANGE,
        marker="o",
        linewidth=1.45,
        markersize=7.35,
        markeredgecolor="black",
        markeredgewidth=0.65,
        zorder=3,
    )
    ax.plot([0.9, 3.55], [0.9, 3.55], color=GRAY, linewidth=0.8, linestyle=(0, (2.4, 2.4)))
    ax.text(2.25, 2.45, "linear scaling", rotation=35, color=GRAY, fontsize=9.7)
    for flops, latency, block in zip(FLOPS_REL_128, LATENCY_REL_128, BLOCK_SIZE):
        ax.annotate(
            f"{block}",
            xy=(flops, latency),
            xytext=(5, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10.2,
        )
    ax.annotate(
        "latency grows\nmuch slower",
        xy=(3.38, 1.30),
        xytext=(2.25, 1.62),
        arrowprops={"arrowstyle": "->", "linewidth": 0.85, "color": "black"},
        ha="center",
        va="center",
        fontsize=10.0,
    )
    ax.set_xlim(0.85, 3.62)
    ax.set_ylim(0.85, 3.62)
    ax.set_xlabel("FLOPs (rel. to 128)")
    ax.set_ylabel("Latency (rel. to 128)")
    ax.tick_params(axis="both", which="major", labelsize=12.5, width=0.85, length=4.5)
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
    fig.tight_layout(pad=0.15)
    return fig


OPTIONS = [
    ("block_cost_option_5_relative_growth_gap.png", option_5_relative_growth_gap),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated option PNG files.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    _setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, builder in OPTIONS:
        fig = builder()
        out_path = args.out_dir / filename
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
