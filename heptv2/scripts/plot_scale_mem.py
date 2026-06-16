"""Plot HEPTv2 peak allocated memory versus input size.

The HEPTv2 curve is read from the latest scalability benchmark summary and
uses ``peak_allocated_mb`` by default. The CSV stores MiB, while the axis is
reported in decimal GB to match the existing ``figures/scale_mem.pdf``.

Usage:
    python heptv2/scripts/plot_scale_mem.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_HEPT_SUMMARY = Path(
    "codex_logs/scalability_50k_500k_10warmup_20measure/summary.csv"
)
DEFAULT_HEPT_METRIC = "peak_allocated_mb"
MIB_PER_DECIMAL_GB = 1_000_000_000 / (1024 * 1024)
BASELINE_HITS = 64_925.0
BASELINE_POINTS = [
    {
        "name": "OC-GNN",
        "hits": BASELINE_HITS,
        "memory_gb": 5.4,
        "marker": "s",
        "size": 72,
        "label_xy": (60000.0, 5.25),
    },
    {
        "name": "ACORN-GNN",
        "hits": BASELINE_HITS,
        "memory_gb": 16.6,
        "marker": "o",
        "size": 95,
        "label_xy": (60000.0, 16.45),
    },
]


def _read_heptv2_points(
    summary_path: Path,
    metric: str,
) -> list[tuple[float, float]]:
    if not summary_path.exists():
        return [
            (50_000.0, 320.3828125 / MIB_PER_DECIMAL_GB),
            (100_000.0, 624.0458984375 / MIB_PER_DECIMAL_GB),
            (200_000.0, 1235.67919921875 / MIB_PER_DECIMAL_GB),
            (300_000.0, 1832.19677734375 / MIB_PER_DECIMAL_GB),
            (400_000.0, 2447.17626953125 / MIB_PER_DECIMAL_GB),
            (500_000.0, 3031.14990234375 / MIB_PER_DECIMAL_GB),
        ]

    points: list[tuple[float, float]] = []
    with summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "ok") != "ok":
                continue
            points.append((float(row["target_n"]), float(row[metric]) / MIB_PER_DECIMAL_GB))
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


def plot_scale_mem(
    summary_path: Path = DEFAULT_HEPT_SUMMARY,
    metric: str = DEFAULT_HEPT_METRIC,
) -> plt.Figure:
    _configure_style()
    fig, ax = plt.subplots(figsize=(3.75, 2.70))
    fig.subplots_adjust(left=0.235, right=0.970, bottom=0.200, top=0.940)

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
        zorder=3,
    )
    for point in BASELINE_POINTS:
        ax.scatter(
            point["hits"],
            point["memory_gb"],
            s=point["size"],
            marker=point["marker"],
            color="#0072B2",
            edgecolor="black",
            linewidth=0.9,
            zorder=4,
        )
        ax.text(
            point["label_xy"][0] + 27000,
            point["label_xy"][1],
            point["name"],
            ha="left",
            va="center",
            color="black",
            fontsize=11.2,
            zorder=5,
        )

    ax.text(
        550000,
        4.5,
        "HEPTv2",
        ha="right",
        va="baseline",
        color="#d55e00",
        fontsize=11.2,
        zorder=4,
    )

    ax.set_xlim(-12_000, 550_000)
    ax.set_ylim(-0.5, 18.5)
    ax.set_xticks([0, 100_000, 200_000, 300_000, 400_000, 500_000])
    ax.set_xticklabels(["0", "100k", "200k", "300k", "400k", "500k"])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("Number of hits")
    ax.set_ylabel("Peak memory (GB)")

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
        help="Memory column in MiB to plot for HEPTv2.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--stem", type=str, default="scale_mem")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png", "svg"],
        choices=["pdf", "png", "svg"],
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_scale_mem(args.summary, args.metric)
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
