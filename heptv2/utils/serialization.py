"""Serialization orders used at inference.

Encoder `random_lsh` is resolved inside HEPTAttention. Decoder serialization is
optional: `hilbert` preserves the historical sorted decoder order, while `none`
keeps the existing sub-event order and skips a pure permutation.
"""
from typing import Optional, Tuple
import torch


def canonicalize_serialization_type(serialization_type: str, stage: str) -> str:
    key = serialization_type.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "random": "random_lsh",
        "lsh": "random_lsh",
        "randomized": "random_lsh",
        "randomized_lsh": "random_lsh",
    }
    normalized = aliases.get(key, key)
    if stage == "encoder":
        allowed = {"random_lsh"}
    else:
        if normalized in {"identity", "no", "off", "disabled"}:
            normalized = "none"
        allowed = {"hilbert", "none"}
    if normalized not in allowed:
        raise ValueError(
            f"Unsupported {stage} serialization type '{serialization_type}'. "
            f"heptv2 only supports random_lsh (encoder) and hilbert/none (decoder)."
        )
    return normalized


def _normalize_to_hilbert_grid(
    point_coords: torch.Tensor, valid_mask: Optional[torch.Tensor], num_bits: int
) -> torch.Tensor:
    coords_xy = point_coords[..., :2]
    if valid_mask is None:
        valid_mask = torch.ones(coords_xy.shape[:-1], dtype=torch.bool, device=coords_xy.device)
    else:
        valid_mask = valid_mask.to(torch.bool)

    pos_inf = torch.tensor(float("inf"), dtype=coords_xy.dtype, device=coords_xy.device)
    neg_inf = torch.tensor(float("-inf"), dtype=coords_xy.dtype, device=coords_xy.device)
    min_xy_valid = torch.where(valid_mask[..., None], coords_xy, pos_inf).amin(dim=-2)
    max_xy_valid = torch.where(valid_mask[..., None], coords_xy, neg_inf).amax(dim=-2)
    has_valid = valid_mask.any(dim=-1)
    min_xy = torch.where(has_valid[..., None], min_xy_valid, coords_xy.amin(dim=-2))
    max_xy = torch.where(has_valid[..., None], max_xy_valid, coords_xy.amax(dim=-2))

    scale = (max_xy - min_xy).clamp_min(1e-6)
    normalized = ((coords_xy - min_xy[..., None, :]) / scale[..., None, :]).clamp(0.0, 1.0)
    max_int = (1 << num_bits) - 1
    return (normalized * max_int).to(torch.long)


def _hilbert_indices_2d(x: torch.Tensor, y: torch.Tensor, num_bits: int) -> torch.Tensor:
    x = x.clone()
    y = y.clone()
    hilbert = torch.zeros_like(x, dtype=torch.long)
    side = 1 << num_bits
    for bit in range(num_bits - 1, -1, -1):
        stride = 1 << bit
        rx = ((x & stride) > 0).to(torch.long)
        ry = ((y & stride) > 0).to(torch.long)
        hilbert = hilbert + (stride * stride) * ((3 * rx) ^ ry)
        swap_mask = ry == 0
        flip_mask = swap_mask & (rx == 1)
        x_flipped = torch.where(flip_mask, (side - 1) - x, x)
        y_flipped = torch.where(flip_mask, (side - 1) - y, y)
        x = torch.where(swap_mask, y_flipped, x_flipped)
        y = torch.where(swap_mask, x_flipped, y_flipped)
    return hilbert


def compute_serialization_order(
    point_coords: torch.Tensor,
    serialization_type: str,
    valid_mask: Optional[torch.Tensor] = None,
    num_bits: int = 10,
    stage: str = "decoder",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if point_coords.dim() not in (2, 3) or point_coords.shape[-1] < 2:
        raise ValueError(
            f"point_coords must have shape [N, D] or [B, N, D] with D>=2, got {tuple(point_coords.shape)}."
        )
    if valid_mask is not None:
        valid_mask = valid_mask.to(torch.bool)

    serialization_type = canonicalize_serialization_type(serialization_type, stage=stage)

    if serialization_type == "hilbert":
        coords_int = _normalize_to_hilbert_grid(point_coords, valid_mask, num_bits=num_bits)
        sort_keys = _hilbert_indices_2d(coords_int[..., 0], coords_int[..., 1], num_bits=num_bits)
        if valid_mask is not None:
            sort_keys = sort_keys.masked_fill(~valid_mask, torch.iinfo(sort_keys.dtype).max)
    else:
        raise ValueError("random_lsh does not define a deterministic serialization order.")

    order = torch.argsort(sort_keys, dim=-1, stable=True)
    return order, None
