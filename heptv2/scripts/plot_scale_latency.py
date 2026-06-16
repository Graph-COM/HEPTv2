"""Reproduce the HEPTv2 latency scaling figure.

The HEPTv2 curve is read from the latest scalability benchmark summary and
uses ``total_ms_mean`` by default. Baseline points and the Two-stage MF line
are digitized from the original comparison plot used to make
``figures/scale_latency.pdf``.

Usage:
    python heptv2/scripts/plot_scale_latency.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_HEPT_SUMMARY = Path(
    "codex_logs/scalability_50k_500k_10warmup_20measure/summary.csv"
)
DEFAULT_HEPT_METRIC = "total_ms_mean"
MAIN_COMPARISON_HITS = 60_000.0

# Digitized from figures/scale_latency.svg. Values are in (hits, latency_ms).
TWO_STAGE_MASKFORMER_LINE = [
    (43_082.19045601351, 73.35499082325424),
    (500_000.0113277905, 1_165.0898565889559),
]
TWO_STAGE_LABEL_BOX = {
    "xy": (322_437.05408683035, 852.2777005222845),
    "width": 165_125.91249955702,
    "height": 215.546718292735,
}
TEXT_POSITIONS = {
    "HEPTv2": (530_000, 238),
    "Two-stage MF": (323_071.4103640278, 934.6579592929888),
    "HGNN": (141_000.0, 1_035.0),
    "GNN4ITk": (367_000.0, 474.88024100988855),
}

BASELINE_POINTS = [
    {
        "name": "ACORN-GNN",
        "hits": MAIN_COMPARISON_HITS,
        "latency_ms": 783.7,
        "marker": "o",
        "color": "#0072B2",
        "size": 95,
        "label_xy": (83_000.0, 750.0),
        "ha": "left",
    },
    {
        "name": "OC-GNN",
        "hits": MAIN_COMPARISON_HITS,
        "latency_ms": 571.5,
        "marker": "s",
        "color": "#0072B2",
        "size": 72,
        "label_xy": (83_000.0, 535.0),
        "ha": "left",
    },

    {
        "name": "HGNN",
        "hits": 120_000.0,
        "latency_ms": 1_070.0,
        "marker": "P",
        "color": "#0072B2",
        "facecolor": "none",
        "size": 78,
        "label_xy": TEXT_POSITIONS["HGNN"],
        "ha": "left",
    },
    {
        "name": "GNN4ITk",
        "hits": 350_000.0,
        "latency_ms": 510.0,
        "marker": "X",
        "color": "#0072B2",
        "facecolor": "none",
        "size": 78,
        "label_xy": TEXT_POSITIONS["GNN4ITk"],
        "ha": "left",
    },
]


def _read_heptv2_points(summary_path: Path, metric: str) -> list[tuple[float, float]]:
    if not summary_path.exists():
        return [
            (50_000.0, 15.369704055786134),
            (100_000.0, 29.49062576293945),
            (200_000.0, 59.43148326873779),
            (300_000.0, 89.39402847290039),
            (400_000.0, 122.70748405456543),
            (500_000.0, 150.5047523498535),
        ]

    points: list[tuple[float, float]] = []
    with summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "ok") != "ok":
                continue
            points.append((float(row["target_n"]), float(row[metric])))
    return sorted(points)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "font.size": 11.2,
            "axes.labelsize": 16.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_scale_latency(
    summary_path: Path = DEFAULT_HEPT_SUMMARY,
    metric: str = DEFAULT_HEPT_METRIC,
) -> plt.Figure:
    _configure_style()
    fig, ax = plt.subplots(figsize=(3.75, 2.70))
    fig.subplots_adjust(left=0.235, right=0.970, bottom=0.200, top=0.940)

    two_stage_x, two_stage_y = zip(*TWO_STAGE_MASKFORMER_LINE)
    ax.plot(
        two_stage_x,
        two_stage_y,
        color="#0072B2",
        linewidth=1.85,
        linestyle=(0, (4.3, 2.3)),
        zorder=1,
    )

    hept_points = _read_heptv2_points(summary_path, metric)
    hept_x, hept_y = zip(*hept_points)
    ax.plot(
        hept_x,
        hept_y,
        color="#d55e00",
        linewidth=1.95,
        marker="*",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=0.65,
        markerfacecolor="#d55e00",
        zorder=2.5,
    )

    for point in BASELINE_POINTS:
        ax.scatter(
            point["hits"],
            point["latency_ms"],
            s=point["size"],
            marker=point["marker"],
            facecolors=point.get("facecolor", point["color"]),
            edgecolors=point["color"] if point.get("facecolor") == "none" else "black",
            linewidth=1.15 if point.get("facecolor") == "none" else 0.65,
            zorder=3,
        )

    ax.text(
        *TEXT_POSITIONS["HEPTv2"],
        "HEPTv2",
        ha="right",
        va="baseline",
        color="#d55e00",
        fontsize=11.2,
        zorder=4,
    )
    ax.add_patch(
        Rectangle(
            TWO_STAGE_LABEL_BOX["xy"],
            TWO_STAGE_LABEL_BOX["width"],
            TWO_STAGE_LABEL_BOX["height"],
            facecolor="white",
            edgecolor="none",
            alpha=0.78,
            zorder=2,
        )
    )
    ax.text(
        *TEXT_POSITIONS["Two-stage MF"],
        "Two-stage MF",
        ha="left",
        va="baseline",
        color="black",
        fontsize=11.2,
        zorder=3,
    )
    for point in BASELINE_POINTS:
        ax.text(
            point["label_xy"][0],
            point["label_xy"][1],
            point["name"],
            ha=point["ha"],
            va="baseline",
            color="black",
            fontsize=11.2,
            zorder=4,
        )

    ax.set_xlim(-12_000, 550_000)
    ax.set_ylim(-40, 1_360)
    ax.set_xticks([0, 100_000, 200_000, 300_000, 400_000, 500_000])
    ax.set_xticklabels(["0", "100k", "200k", "300k", "400k", "500k"])
    ax.set_yticks([0, 200, 400, 600, 800, 1_000, 1_200])
    ax.set_xlabel("Number of hits")
    ax.set_ylabel("Latency (ms)")

    ax.grid(
        True,
        which="major",
        color="#b8b8b8",
        alpha=0.85,
        linewidth=0.75,
        linestyle=(0, (2.4, 2.4)),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", length=4.5, width=0.85, pad=3.5)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_HEPT_SUMMARY,
        help="HEPTv2 scalability summary CSV.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_HEPT_METRIC,
        help="Latency column to plot for HEPTv2.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--stem", type=str, default="scale_latency")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png", "svg"],
        choices=["pdf", "png", "svg"],
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_scale_latency(args.summary, args.metric)
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
