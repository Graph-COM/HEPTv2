"""Plot DM efficiency versus encoder block size for serialization ablations.

The values are taken from the block-size ablation table in the paper draft.

Usage:
    python heptv2/scripts/plot_dm_vs_block_size.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


SERIES = [
    {
        "label": "HEPTv2 (Full)",
        "block_size": [128, 256, 512, 1024],
        "dm": [98.3, 98.5, 98.5, 98.6],
        "color": "#D55E00",
        "marker": "*",
        "markersize": 12.0,
        "linestyle": "-",
    },
    {
        "label": "HEPTv2 (Z-H)",
        "block_size": [128, 256, 512, 1024],
        "dm": [95.2, 95.8, 96.5, 97.4],
        "color": "#4D4D4D",
        "marker": "*",
        "markersize": 12.0,
        "linestyle": "-",
    },
    {
        "label": r"HEPTv2 ($\phi$)",
        "block_size": [128, 256, 512, 1024],
        "dm": [92.7, 95.5, 96.8, 97.6],
        "color": "#009E73",
        "marker": "*",
        "markersize": 12.0,
        "linestyle": "-",
    },
    {
        "label": r"Two-stage MF ($\phi$)",
        "block_size": [1024],
        "dm": [94.1],
        "color": "#0072B2",
        "marker": "^",
        "markersize": 10.0,
        "linestyle": "None",
        "markerfacecolor": "white",
        "markeredgecolor": "#0072B2",
        "markeredgewidth": 1.45,
    },
]

FIGSIZE = (3.65, 3.35)
PANEL_ADJUST = {
    "left": 0.23,
    "right": 0.98,
    "bottom": 0.19,
    "top": 0.78,
}


def plot_dm_vs_block_size() -> plt.Figure:
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

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for series in SERIES:
        ax.plot(
            series["block_size"],
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

    ax.set_xscale("log", base=2)
    ax.set_xlim(1200, 110)
    ax.set_ylim(92.2, 99.0)
    ax.set_xticks([128, 256, 512, 1024])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Encoder Block Size")
    ax.set_ylabel(r"$\epsilon^{\mathrm{DM}}_{p_T>0.9}$ (%)")

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

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.35,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    fig.subplots_adjust(**PANEL_ADJUST)
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
        default="dm_vs_block_size",
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
    fig = plot_dm_vs_block_size()
    for fmt in args.formats:
        out_path = args.out_dir / f"{args.stem}.{fmt}"
        save_kwargs = {}
        if fmt == "png":
            save_kwargs["dpi"] = 300
        fig.savefig(out_path, **save_kwargs)
        print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
