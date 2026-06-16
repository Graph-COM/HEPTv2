"""Visualize how HEPTAttention buckets points in eta/phi.

For a single event, plot each hit at its (eta, phi) coordinate colored by the
bucket index it lands in after sorting, under four regimes shown in a 2x2 grid:
  1. `random_lsh`: the E2LSH hashing + region-aware geo-shift used by hept.py.
  2. `hilbert`: Hilbert-curve serialization (deterministic).
  3. `zorder`: Morton/Z-order serialization (deterministic).
  4. `phi`: sort by phi coordinate (deterministic).

"Buckets" for deterministic serializations are contiguous chunks of `block_size`
points along the serialization order.

Usage:
    python visualize_hept_buckets.py \
        --config configs/tracking/tracking_trans_hept.yaml \
        --split test --event-idx 0 --out bucket_plots

Outputs one PNG: `<out>/buckets_grid.png`.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.model_utils.hash_utils import E2LSH, get_regions, lsh_mapping
from models.model_utils.serialization_utils import compute_serialization_order
from utils.config_overrides import load_yaml_config
from utils.get_data import get_dataset
from utils.process_data_utils import build_region_indices, pad_to_fixed_len_all


def _load_event(config, split, event_idx):
    dataset_dir = Path(config["data_dir"]) / "tracking"
    dataset = get_dataset(config["dataset_name"], dataset_dir)
    idx = dataset.idx_split[split]
    if event_idx >= len(idx):
        raise IndexError(
            f"event_idx={event_idx} >= {len(idx)} events available in '{split}' split"
        )
    return dataset[idx[event_idx]]


def _prepare_event(data, encoder_block_size, num_regions, n_hashes, num_heads, seed):
    torch.manual_seed(seed)
    data, valid_mask = pad_to_fixed_len_all(
        data, fixed_len=data.x.shape[0], block_size=encoder_block_size
    )
    regions = get_regions(num_regions, n_hashes, num_heads)
    region_indices, regions_h = build_region_indices(data.coords, valid_mask, regions)
    return data, valid_mask, region_indices, regions_h


def _geo_shift(regions_h, hash_shift, region_indices, n_hashes):
    region_indices_eta, region_indices_phi = region_indices
    q_hash_shift_eta = region_indices_eta * hash_shift
    q_hash_shift_phi = (
        region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    )
    shifts = q_hash_shift_eta + q_hash_shift_phi
    # (c*h, N) -> (c, h, N)
    return shifts.view(n_hashes, -1, shifts.shape[-1])


def _random_lsh_positions(coords, valid_mask, region_indices, regions_h,
                          n_hashes, num_heads, seed):
    torch.manual_seed(seed)
    e2lsh = E2LSH(n_hashes=n_hashes, n_heads=num_heads, dim=2)

    pos = coords[..., :2].unsqueeze(0).repeat(num_heads, 1, 1)  # (h, N, 2)
    q_hashed, _, hash_shift = lsh_mapping(e2lsh, pos, pos)  # q_hashed: (c, h, N)

    invalid = ~valid_mask
    q_hashed[..., invalid] = float("inf")

    hash_shift = hash_shift.reshape(n_hashes * num_heads, -1)
    shifts = _geo_shift(regions_h, hash_shift, region_indices, n_hashes)
    q_hashed = q_hashed + shifts  # (c, h, N)

    # argsort gives the permutation (the order we'd read points in).
    # We want, for each point, its position in that order -> argsort of argsort.
    perm = q_hashed.argsort(dim=-1)
    positions = perm.argsort(dim=-1)
    return positions  # (c, h, N)


def _deterministic_positions(coords, valid_mask, serialization_type, num_bits):
    order, inverse = compute_serialization_order(
        point_coords=coords,
        serialization_type=serialization_type,
        valid_mask=valid_mask,
        num_bits=num_bits,
        stage="encoder",
    )
    return inverse  # inverse[i] = position of point i in the sorted order


def _plot_bucket_panel(ax, coords_np, valid, bucket_ids_np, title):
    eta = coords_np[valid, 0]
    phi = coords_np[valid, 1]
    b = bucket_ids_np[valid]
    n_buckets = int(b.max()) + 1 if b.size > 0 else 1
    # Shuffle bucket ids, then cycle a small qualitative palette so neighboring
    # buckets land on visibly different colors instead of adjacent hues.
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_buckets)
    palette = plt.get_cmap("tab20").colors
    colors = np.asarray(palette)[perm[b] % len(palette)]
    scatter = ax.scatter(
        eta, phi, c=colors, s=5, alpha=1.0,
        edgecolors="black", linewidths=0.2,
    )
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(f"{title}  (n_buckets={n_buckets}, n_valid_hits={valid.sum()})")
    return scatter


def _plot_grid(coords, valid_mask, panels, out_path):
    coords_np = coords.detach().cpu().numpy()
    valid = valid_mask.detach().cpu().numpy().astype(bool)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for ax, (title, bucket_ids) in zip(axes.flat, panels):
        bucket_ids_np = bucket_ids.detach().cpu().numpy()
        _plot_bucket_panel(ax, coords_np, valid, bucket_ids_np, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/tracking/tracking_trans_hept.yaml")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"])
    parser.add_argument("--event-idx", type=int, default=0)
    parser.add_argument("--hash-idx", type=int, default=0,
                        help="which of the n_hashes LSH hashes to visualize")
    parser.add_argument("--head-idx", type=int, default=0,
                        help="which attention head's LSH permutation to visualize")
    parser.add_argument("--hilbert-bits", type=int, default=None,
                        help="override encoder_hilbert_bits for the hilbert plot")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the E2LSH random projections")
    parser.add_argument("--block-size", type=int, default=None,
                        help="override encoder_block_size for bucketing")
    parser.add_argument("--out", type=str, default="bucket_plots")
    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))
    mk = config["model_kwargs"]

    block_size = int(args.block_size or mk["encoder_block_size"])
    n_hashes = int(mk["n_hashes"])
    num_heads = int(mk["num_heads"])
    num_regions = int(mk["num_regions"])
    hilbert_bits = args.hilbert_bits or int(mk.get("encoder_hilbert_bits", 10))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_event(config, args.split, args.event_idx)
    data, valid_mask, region_indices, regions_h = _prepare_event(
        data, block_size, num_regions, n_hashes, num_heads, seed=args.seed,
    )

    # --- random_lsh ---
    positions = _random_lsh_positions(
        data.coords, valid_mask, region_indices, regions_h,
        n_hashes, num_heads, seed=args.seed,
    )
    bucket_ids_lsh = positions[args.hash_idx, args.head_idx] // block_size

    # --- deterministic serializations ---
    bucket_ids_hilbert = _deterministic_positions(
        data.coords, valid_mask, "hilbert", hilbert_bits
    ) // block_size
    bucket_ids_zorder = _deterministic_positions(
        data.coords, valid_mask, "zorder", hilbert_bits
    ) // block_size
    bucket_ids_phi = _deterministic_positions(
        data.coords, valid_mask, "phi", hilbert_bits
    ) // block_size

    panels = [
        (f"random_lsh (hash={args.hash_idx}, head={args.head_idx}, bs={block_size})",
         bucket_ids_lsh),
        (f"hilbert (num_bits={hilbert_bits}, bs={block_size})", bucket_ids_hilbert),
        (f"zorder (num_bits={hilbert_bits}, bs={block_size})", bucket_ids_zorder),
        (f"phi (bs={block_size})", bucket_ids_phi),
    ]
    _plot_grid(data.coords, valid_mask, panels, out_path=out_dir / "buckets_grid.png")


if __name__ == "__main__":
    main()
