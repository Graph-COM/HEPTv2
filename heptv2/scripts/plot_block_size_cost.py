"""Plot encoder cost versus block size.

Usage:
    python heptv2/scripts/plot_block_size_cost.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_block_size_cost_options import _setup_style, option_5_relative_growth_gap


def plot_block_size_cost() -> plt.Figure:
    _setup_style()
    return option_5_relative_growth_gap()


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
        default="block_cost_vs_block_size",
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
    fig = plot_block_size_cost()
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
