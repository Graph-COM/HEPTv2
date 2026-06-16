from typing import Optional, Tuple

import torch


def canonicalize_serialization_type(serialization_type: str, stage: str) -> str:
    key = serialization_type.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "random": "random_lsh",
        "lsh": "random_lsh",
        "randomized": "random_lsh",
        "randomized_lsh": "random_lsh",
        "phi_based": "phi",
        "z": "zorder",
        "z_order": "zorder",
        "morton": "zorder",
        "morton_order": "zorder",
        "transzorder": "trans_zorder",
        "trans_z_order": "trans_zorder",
        "trans_morton": "trans_zorder",
        "trans_morton_order": "trans_zorder",
        "thilbert": "trans_hilbert",
        "transhilbert": "trans_hilbert",
    }
    normalized = aliases.get(key, key)

    if stage == "encoder":
        allowed = {"random_lsh", "zorder", "trans_zorder", "hilbert", "trans_hilbert", "phi"}
    elif stage == "decoder":
        allowed = {"hilbert", "phi"}
    else:
        raise ValueError(f"Unsupported serialization stage '{stage}'.")

    if normalized not in allowed:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unsupported {stage} serialization type '{serialization_type}'. "
            f"Expected one of: {allowed_str}."
        )
    return normalized


def _normalize_to_hilbert_grid(
    point_coords: torch.Tensor, valid_mask: Optional[torch.Tensor], num_bits: int
) -> torch.Tensor:
    coords_xy = point_coords[:, :2]
    if valid_mask is None:
        valid_mask = torch.ones(coords_xy.shape[0], dtype=torch.bool, device=coords_xy.device)
    else:
        valid_mask = valid_mask.to(torch.bool)

    pos_inf = torch.tensor(float("inf"), dtype=coords_xy.dtype, device=coords_xy.device)
    neg_inf = torch.tensor(float("-inf"), dtype=coords_xy.dtype, device=coords_xy.device)
    min_xy_valid = torch.where(valid_mask[:, None], coords_xy, pos_inf).amin(dim=0)
    max_xy_valid = torch.where(valid_mask[:, None], coords_xy, neg_inf).amax(dim=0)
    min_xy_all = coords_xy.amin(dim=0)
    max_xy_all = coords_xy.amax(dim=0)
    has_valid = valid_mask.any()
    min_xy = torch.where(has_valid, min_xy_valid, min_xy_all)
    max_xy = torch.where(has_valid, max_xy_valid, max_xy_all)

    scale = (max_xy - min_xy).clamp_min(1e-6)
    normalized = ((coords_xy - min_xy) / scale).clamp(0.0, 1.0)
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


def _morton_indices_2d(x: torch.Tensor, y: torch.Tensor, num_bits: int) -> torch.Tensor:
    morton = torch.zeros_like(x, dtype=torch.long)
    for bit in range(num_bits):
        x_bit = (x >> bit) & 1
        y_bit = (y >> bit) & 1
        morton = morton | (x_bit << (2 * bit + 1)) | (y_bit << (2 * bit))
    return morton


def compute_serialization_order(
    point_coords: torch.Tensor,
    serialization_type: str,
    valid_mask: Optional[torch.Tensor] = None,
    num_bits: int = 10,
    stage: str = "decoder",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if point_coords.dim() != 2 or point_coords.shape[1] < 2:
        raise ValueError(
            f"point_coords must have shape [N, D] with D>=2, got {tuple(point_coords.shape)}."
        )

    if valid_mask is not None:
        valid_mask = valid_mask.to(torch.bool)

    serialization_type = canonicalize_serialization_type(serialization_type, stage=stage)

    if serialization_type == "phi":
        sort_keys = point_coords[:, 1]
        if valid_mask is not None:
            sort_keys = sort_keys.masked_fill(~valid_mask, float("inf"))
    elif serialization_type == "hilbert":
        coords_int = _normalize_to_hilbert_grid(point_coords, valid_mask, num_bits=num_bits)
        sort_keys = _hilbert_indices_2d(coords_int[:, 0], coords_int[:, 1], num_bits=num_bits)
        if valid_mask is not None:
            sort_keys = sort_keys.masked_fill(~valid_mask, torch.iinfo(sort_keys.dtype).max)
    elif serialization_type == "trans_hilbert":
        coords_int = _normalize_to_hilbert_grid(point_coords, valid_mask, num_bits=num_bits)
        sort_keys = _hilbert_indices_2d(coords_int[:, 1], coords_int[:, 0], num_bits=num_bits)
        if valid_mask is not None:
            sort_keys = sort_keys.masked_fill(~valid_mask, torch.iinfo(sort_keys.dtype).max)
    elif serialization_type == "zorder":
        coords_int = _normalize_to_hilbert_grid(point_coords, valid_mask, num_bits=num_bits)
        sort_keys = _morton_indices_2d(coords_int[:, 0], coords_int[:, 1], num_bits=num_bits)
        if valid_mask is not None:
            sort_keys = sort_keys.masked_fill(~valid_mask, torch.iinfo(sort_keys.dtype).max)
    elif serialization_type == "trans_zorder":
        coords_int = _normalize_to_hilbert_grid(point_coords, valid_mask, num_bits=num_bits)
        sort_keys = _morton_indices_2d(coords_int[:, 1], coords_int[:, 0], num_bits=num_bits)
        if valid_mask is not None:
            sort_keys = sort_keys.masked_fill(~valid_mask, torch.iinfo(sort_keys.dtype).max)
    elif serialization_type == "random_lsh":
        raise ValueError("random_lsh does not define a deterministic serialization order.")
    else:
        raise RuntimeError(f"Unexpected serialization type '{serialization_type}'.")

    # Keep equal-key elements (especially padded points with identical sentinel keys)
    # in their original order to avoid unnecessary permutations.
    order = torch.argsort(sort_keys, dim=0, stable=True)
    inverse_order = torch.empty_like(order)
    inverse_order.scatter_(
        0,
        order,
        torch.arange(order.numel(), device=order.device, dtype=order.dtype),
    )
    return order, inverse_order
