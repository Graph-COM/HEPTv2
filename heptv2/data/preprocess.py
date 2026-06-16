"""Per-event preprocessing: eta filter, padding to a fixed length, region
indexing for HEPT, and splitting the event into (overlapping) decoder
sub-events along phi.
"""
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from einops import rearrange
from torch_scatter import scatter_mean

from ..utils.block_size_utils import get_tracking_padded_lengths


def _pad_first_dim(tensor, pad_n, value=0):
    if pad_n <= 0:
        return tensor
    if tensor.dim() == 1:
        return F.pad(tensor, (0, pad_n), value=value)
    pad = [0, 0] * (tensor.dim() - 1) + [0, pad_n]
    return F.pad(tensor, pad, value=value)


def _pad_first_dim_inference(tensor, target_len, value=0.0):
    if int(tensor.shape[0]) == int(target_len):
        return tensor
    out_shape = (int(target_len),) + tuple(tensor.shape[1:])
    out = tensor.new_full(out_shape, value)
    out[: int(tensor.shape[0])].copy_(tensor)
    return out


def _quantile_partition_from_sorted(sorted_indices, num_regions):
    total_elements = sorted_indices.shape[-1]
    region_size = torch.ceil(total_elements / num_regions)
    ranks = torch.arange(total_elements, device=sorted_indices.device, dtype=sorted_indices.dtype)
    inverse_indices = torch.empty_like(sorted_indices)
    inverse_indices.scatter_(-1, sorted_indices, ranks.expand_as(sorted_indices))
    base = ranks[None]
    region_indices = base // region_size + 1
    return region_indices[:, inverse_indices]


def _quantile_partition_from_sorted_with_total(sorted_indices, num_regions, total_elements):
    local_elements = sorted_indices.shape[-1]
    ranks = torch.arange(local_elements, device=sorted_indices.device, dtype=sorted_indices.dtype)
    inverse_indices = torch.empty_like(sorted_indices)
    inverse_indices.scatter_(-1, sorted_indices, ranks.expand_as(sorted_indices))
    region_size = torch.ceil(float(total_elements) / num_regions)
    region_indices = ranks[None] // region_size + 1
    return region_indices[:, inverse_indices]


def pad_to_fixed_len_all(data, fixed_len, block_size):
    if fixed_len % block_size != 0:
        fixed_len = ((fixed_len + block_size - 1) // block_size) * block_size

    raw_n = data.x.shape[0]
    if raw_n > fixed_len:
        raise ValueError(f"Input length {raw_n} exceeds fixed_seq_len {fixed_len}")

    pad_n = fixed_len - raw_n
    valid_mask = torch.cat(
        [
            torch.ones(raw_n, dtype=torch.bool, device=data.x.device),
            torch.zeros(pad_n, dtype=torch.bool, device=data.x.device),
        ]
    )
    if pad_n > 0:
        data.x = _pad_first_dim(data.x, pad_n, value=0.0)
        data.coords = _pad_first_dim(data.coords, pad_n, value=0.0)
        data.pos = _pad_first_dim(data.pos, pad_n, value=0.0)
        data.layer = _pad_first_dim(data.layer, pad_n, value=0)
        data.particle_id = _pad_first_dim(data.particle_id, pad_n, value=-1)
        data.pt = _pad_first_dim(data.pt, pad_n, value=0.0)
        data.reconstructable = _pad_first_dim(data.reconstructable, pad_n, value=0)
        for attr, pad_value in (("eta", 0.0), ("phi", 0.0), ("hit_id", -1)):
            if hasattr(data, attr) and getattr(data, attr) is not None:
                setattr(data, attr, _pad_first_dim(getattr(data, attr), pad_n, value=pad_value))
    return data, valid_mask


def build_region_indices(coords, valid_mask, regions, return_sorted=False):
    eta_for_sort = coords[..., 0].masked_fill(~valid_mask, float("inf"))
    phi_for_sort = coords[..., 1].masked_fill(~valid_mask, float("inf"))
    sorted_eta_idx = torch.argsort(eta_for_sort, dim=-1)
    sorted_phi_idx = torch.argsort(phi_for_sort, dim=-1)
    regions_h = rearrange(regions, "c a h -> a (c h)")
    region_indices_eta = _quantile_partition_from_sorted(sorted_eta_idx, regions_h[0][:, None])
    region_indices_phi = _quantile_partition_from_sorted(sorted_phi_idx, regions_h[1][:, None])
    if return_sorted:
        return [region_indices_eta, region_indices_phi], regions_h, sorted_eta_idx, sorted_phi_idx
    return [region_indices_eta, region_indices_phi], regions_h


def build_region_indices_valid_prefix(coords, raw_n, padded_len, regions, quantile_total_len=None):
    coords_valid = coords[:raw_n]
    sorted_eta_idx = torch.argsort(coords_valid[..., 0], dim=-1)
    sorted_phi_idx = torch.argsort(coords_valid[..., 1], dim=-1)
    regions_h = rearrange(regions, "c a h -> a (c h)")
    if quantile_total_len is None:
        region_indices_eta = _quantile_partition_from_sorted(sorted_eta_idx, regions_h[0][:, None])
        region_indices_phi = _quantile_partition_from_sorted(sorted_phi_idx, regions_h[1][:, None])
    else:
        region_indices_eta = _quantile_partition_from_sorted_with_total(
            sorted_eta_idx,
            regions_h[0][:, None],
            quantile_total_len,
        )
        region_indices_phi = _quantile_partition_from_sorted_with_total(
            sorted_phi_idx,
            regions_h[1][:, None],
            quantile_total_len,
        )
    pad_n = int(padded_len) - int(raw_n)
    if pad_n > 0:
        pad_eta = torch.ones(
            (int(region_indices_eta.shape[0]), pad_n),
            device=coords.device,
            dtype=region_indices_eta.dtype,
        )
        pad_phi = torch.ones(
            (int(region_indices_phi.shape[0]), pad_n),
            device=coords.device,
            dtype=region_indices_phi.dtype,
        )
        region_indices_eta = torch.cat([region_indices_eta, pad_eta], dim=1)
        region_indices_phi = torch.cat([region_indices_phi, pad_phi], dim=1)
    return [region_indices_eta, region_indices_phi], regions_h, sorted_phi_idx


def _eta_values(data):
    if hasattr(data, "eta") and data.eta is not None:
        return data.eta
    return data.pos[:, 0]


def _apply_eta_filter(data, eta_abs_max=4.0):
    cond = _eta_values(data).abs() <= eta_abs_max
    data.x = data.x[cond]
    data.pos = data.pos[cond]
    data.coords = data.coords[cond]
    data.layer = data.layer[cond]
    data.particle_id = data.particle_id[cond]
    data.pt = data.pt[cond]
    data.reconstructable = data.reconstructable[cond]
    for attr in ("eta", "phi", "hit_id"):
        if hasattr(data, attr) and getattr(data, attr) is not None:
            setattr(data, attr, getattr(data, attr)[cond])
    data.edge_index = None
    return data


def _eta_filter_mask_for_inference(data, coords, eta_abs_max=4.0):
    if hasattr(data, "eta") and data.eta is not None:
        eta = data.eta
    elif hasattr(data, "pos") and data.pos is not None:
        eta = data.pos[:, 0]
    else:
        eta = coords[:, 0]
    return eta.abs() <= eta_abs_max


def _build_sub_event(valid_mask_sub, idxs, sub_targets=None):
    entry = {
        "all_idx": idxs,
        "orig_idx": idxs[valid_mask_sub],
        "valid_mask": valid_mask_sub,
    }
    if sub_targets is not None:
        entry["targets"] = sub_targets
    return entry


def _build_cluster_masks(
    particle_id,
    pt,
    reconstructable,
    thld=0.9,
    global_particle_ids=None,
    global_hit_counts=None,
):
    """Per-sub-event training targets: one mask per good-track particle,
    plus a single class label `0` ("object") per query. Matches
    src/utils/process_data_utils.py:build_cluster_masks.
    """
    unique_pids, inv_idx = torch.unique(particle_id, return_inverse=True)
    pt_list = scatter_mean(pt, inv_idx, dim=0)
    rec_list = scatter_mean(reconstructable.float(), inv_idx, dim=0)
    good_mask = (pt_list > thld) & (rec_list == 1.0)
    good_idx = torch.nonzero(good_mask, as_tuple=False).squeeze(1)
    masks = (inv_idx[None, :] == good_idx[:, None]).long().unsqueeze(-1)
    labels = torch.zeros(masks.shape[0], dtype=torch.long, device=particle_id.device)
    hit_counts = None
    if global_particle_ids is not None and global_hit_counts is not None and good_idx.numel() > 0:
        good_pids = unique_pids[good_idx]
        lookup = torch.searchsorted(global_particle_ids, good_pids)
        in_range = lookup < global_particle_ids.numel()
        matched = torch.zeros_like(in_range)
        if in_range.any():
            matched[in_range] = global_particle_ids[lookup[in_range]] == good_pids[in_range]
        hit_counts = masks.flatten(1).sum(dim=1)
        if matched.any():
            hit_counts = hit_counts.to(global_hit_counts.dtype)
            hit_counts[matched] = global_hit_counts[lookup[matched]]
    return labels, masks, hit_counts


def _split_event_into_sub_events(
    data,
    fixed_len,
    encoder_block_size,
    decoder_block_size,
    regions,
    num_sub_events=3,
    decoder_overlap=True,
    random_split_shift=False,
    random_split_shift_max_frac=1.0,
    overlap_shift_jitter_frac=0.0,
    build_targets=False,
):
    """Pad the event and split it into decoder sub-events along phi.

    Returns `(full_data, sub_events)` where `full_data` is the padded event
    with `attn_kwargs` attached, and `sub_events` is a list of per-window
    index dicts consumed by the decoder.

    When `build_targets=True`, each sub-event dict also carries `targets`
    of the form `[[{"labels": [Q], "masks": [Q, N_sub, 1]}]]` used by the
    set loss during training.
    """
    y_true = (data.pt >= 0.9) & (data.reconstructable == 1)

    padded_fixed_len, sub_len = get_tracking_padded_lengths(
        fixed_len=fixed_len,
        encoder_block_size=encoder_block_size,
        decoder_block_size=decoder_block_size,
        num_sub_events=num_sub_events,
    )
    data, valid_mask = pad_to_fixed_len_all(data, padded_fixed_len, encoder_block_size)

    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    pad_idx = torch.nonzero(~valid_mask, as_tuple=False).squeeze(1)
    required_invalid_slots = sub_len * num_sub_events - valid_idx.numel()
    if required_invalid_slots > 0 and pad_idx.numel() == 0:
        data, valid_mask = pad_to_fixed_len_all(data, valid_mask.numel() + encoder_block_size, encoder_block_size)
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        pad_idx = torch.nonzero(~valid_mask, as_tuple=False).squeeze(1)

    region_indices, regions_h = build_region_indices(data.coords, valid_mask, regions)
    data.valid_mask = valid_mask
    data.region_indices = region_indices
    data.regions_h = regions_h
    data.attn_kwargs = {
        "coords": data.coords,
        "valid_mask": valid_mask,
        "region_indices": region_indices,
        "regions_h": regions_h,
    }
    data.targets = [[{"y_true": y_true}]]

    phi_valid = data.coords[valid_idx, 1]
    sorted_valid = valid_idx[torch.argsort(phi_valid)]
    n_valid = sorted_valid.numel()
    global_particle_ids = None
    global_hit_counts = None
    if build_targets and n_valid > 0:
        global_particle_ids, global_hit_counts = torch.unique(
            data.particle_id.index_select(0, valid_idx),
            return_counts=True,
        )
    if random_split_shift and n_valid > 1:
        window = max(1, n_valid // max(1, int(num_sub_events)))
        max_shift = max(1, int(round(window * float(random_split_shift_max_frac))))
        max_shift = min(max_shift, int(n_valid))
        split_shift = int(torch.randint(max_shift, (1,), device=sorted_valid.device).item())
        if split_shift:
            sorted_valid = torch.roll(sorted_valid, shifts=split_shift, dims=0)
    base = n_valid // num_sub_events
    rem = n_valid % num_sub_events
    valid_counts = torch.full((num_sub_events,), base, device=sorted_valid.device, dtype=torch.long)
    if rem > 0:
        valid_counts[:rem] += 1
    pos = torch.arange(sub_len, device=sorted_valid.device)[None, :]

    def _build_indices(sorted_hits):
        valid_mask_grid = pos < valid_counts[:, None]
        full_idx = torch.empty((num_sub_events, sub_len), device=sorted_hits.device, dtype=sorted_hits.dtype)
        if sorted_hits.numel() > 0:
            full_idx[valid_mask_grid] = sorted_hits
        invalid_count = int((~valid_mask_grid).sum().item())
        if invalid_count > 0:
            if pad_idx.numel() == 0:
                raise ValueError(
                    "Decoder sub-events require padded positions, but none were created. "
                    "Increase fixed_seq_len or reduce decoder_block_size."
                )
            fill_idx = pad_idx.repeat((invalid_count + pad_idx.numel() - 1) // pad_idx.numel())[:invalid_count]
            full_idx[~valid_mask_grid] = fill_idx
        return full_idx

    base_idx = _build_indices(sorted_valid)
    if decoder_overlap:
        if n_valid > 0:
            shift = max(1, n_valid // (2 * num_sub_events))
            jitter_frac = float(overlap_shift_jitter_frac)
            if jitter_frac > 0:
                jitter_span = max(1, int(round(shift * jitter_frac)))
                jitter = int(torch.randint(2 * jitter_span + 1, (1,), device=sorted_valid.device).item()) - jitter_span
                shift = max(1, min(int(n_valid) - 1 if n_valid > 1 else 1, shift + jitter))
            shifted_valid = torch.roll(sorted_valid, shifts=-shift, dims=0)
            shifted_idx = _build_indices(shifted_valid)
        else:
            shifted_idx = base_idx.clone()
        all_indices = torch.cat([base_idx, shifted_idx], dim=0)
    else:
        all_indices = base_idx

    all_valid_masks = valid_mask.index_select(0, all_indices.reshape(-1)).view_as(all_indices).bool()
    data.decoder_sub_all_idx = all_indices
    data.decoder_sub_valid_mask = all_valid_masks
    data.decoder_sub_orig_idx_padded = torch.where(
        all_valid_masks,
        all_indices,
        torch.full_like(all_indices, -1),
    )

    sub_events = []
    for idxs, valid_mask_sub in zip(all_indices, all_valid_masks):
        sub_targets = None
        if build_targets:
            sub_pid = data.particle_id.index_select(0, idxs)
            sub_pt = data.pt.index_select(0, idxs)
            sub_rec = data.reconstructable.index_select(0, idxs)
            labels, masks, hit_counts = _build_cluster_masks(
                sub_pid,
                sub_pt,
                sub_rec,
                global_particle_ids=global_particle_ids,
                global_hit_counts=global_hit_counts,
            )
            target = {"labels": labels, "masks": masks}
            if hit_counts is not None:
                target["hit_counts"] = hit_counts
            sub_targets = [[target]]
        sub_events.append(_build_sub_event(valid_mask_sub, idxs, sub_targets))
    data.decoder_overlap = decoder_overlap
    return data, sub_events


@torch.no_grad()
def process_inference_data(
    data,
    fixed_len,
    encoder_block_size,
    decoder_block_size,
    regions,
    num_sub_events=3,
    decoder_overlap=True,
    eta_abs_max=4.0,
    return_sub_events=False,
    dynamic_pad=False,
    pad_bucket_lengths=None,
    match_fixed_region_bins=False,
    build_targets=False,
):
    """Minimal inference-only preprocessing.

    This deliberately keeps only fields needed by the encoder, decoder, and
    fixed post-processing path. Evaluation labels such as particle_id/pt are
    copied only when `build_targets=True`, so latency measurements can isolate
    the online inference pipeline after the event has already been moved to
    the GPU.
    """
    coords_in = data.coords
    keep = _eta_filter_mask_for_inference(data, coords_in, eta_abs_max=eta_abs_max)
    x = data.x[keep]
    coords = coords_in[keep]
    layer = data.layer[keep] if hasattr(data, "layer") and data.layer is not None else None
    particle_id = data.particle_id[keep] if build_targets and hasattr(data, "particle_id") else None
    pt = data.pt[keep] if build_targets and hasattr(data, "pt") else None
    reconstructable = (
        data.reconstructable[keep]
        if build_targets and hasattr(data, "reconstructable")
        else None
    )

    raw_n = int(x.shape[0])
    if pad_bucket_lengths:
        target_fixed_len = None
        for bucket_len in sorted(int(v) for v in pad_bucket_lengths):
            if raw_n <= bucket_len:
                target_fixed_len = bucket_len
                break
        if target_fixed_len is None:
            target_fixed_len = max(int(v) for v in pad_bucket_lengths)
    else:
        target_fixed_len = raw_n if bool(dynamic_pad) else fixed_len
    padded_fixed_len, sub_len = get_tracking_padded_lengths(
        fixed_len=target_fixed_len,
        encoder_block_size=encoder_block_size,
        decoder_block_size=decoder_block_size,
        num_sub_events=num_sub_events,
    )
    if raw_n > padded_fixed_len:
        raise ValueError(f"Input length {raw_n} exceeds fixed_seq_len {padded_fixed_len}")

    pad_n = padded_fixed_len - raw_n
    if pad_n > 0:
        x = _pad_first_dim_inference(x, padded_fixed_len, value=0.0)
        coords = _pad_first_dim_inference(coords, padded_fixed_len, value=0.0)
        if layer is not None:
            layer = _pad_first_dim_inference(layer, padded_fixed_len, value=0)
        if particle_id is not None:
            particle_id = _pad_first_dim_inference(particle_id, padded_fixed_len, value=-1)
        if pt is not None:
            pt = _pad_first_dim_inference(pt, padded_fixed_len, value=0.0)
        if reconstructable is not None:
            reconstructable = _pad_first_dim_inference(reconstructable, padded_fixed_len, value=0)
    valid_mask = torch.empty(padded_fixed_len, dtype=torch.bool, device=x.device)
    valid_mask[:raw_n] = True
    valid_mask[raw_n:] = False

    region_quantile_len = None
    if bool(match_fixed_region_bins):
        region_quantile_len, _ = get_tracking_padded_lengths(
            fixed_len=fixed_len,
            encoder_block_size=encoder_block_size,
            decoder_block_size=decoder_block_size,
            num_sub_events=num_sub_events,
        )
    region_indices, regions_h, sorted_phi_idx = build_region_indices_valid_prefix(
        coords,
        raw_n,
        padded_fixed_len,
        regions,
        quantile_total_len=region_quantile_len,
    )

    pad_idx = torch.arange(raw_n, padded_fixed_len, device=x.device, dtype=torch.long)
    sorted_valid = sorted_phi_idx.to(dtype=torch.long)
    n_valid = int(raw_n)
    base = n_valid // int(num_sub_events)
    rem = n_valid % int(num_sub_events)
    valid_counts = torch.full((int(num_sub_events),), base, device=x.device, dtype=torch.long)
    if rem > 0:
        valid_counts[:rem] += 1
    pos = torch.arange(sub_len, device=x.device)[None, :]

    def _build_indices(sorted_hits):
        valid_mask_grid = pos < valid_counts[:, None]
        full_idx = torch.empty((int(num_sub_events), sub_len), device=x.device, dtype=torch.long)
        if sorted_hits.numel() > 0:
            full_idx[valid_mask_grid] = sorted_hits
        invalid_count = int(valid_mask_grid.numel()) - int(sorted_hits.numel())
        if invalid_count > 0:
            if pad_idx.numel() == 0:
                raise ValueError(
                    "Decoder sub-events require padded positions, but none were created. "
                    "Increase fixed_seq_len or reduce decoder_block_size."
                )
            full_idx[~valid_mask_grid] = pad_idx.repeat(
                (invalid_count + pad_idx.numel() - 1) // pad_idx.numel()
            )[:invalid_count]
        return full_idx

    base_idx = _build_indices(sorted_valid)
    if decoder_overlap:
        if n_valid > 0:
            shift = max(1, n_valid // (2 * int(num_sub_events)))
            shifted_idx = _build_indices(torch.roll(sorted_valid, shifts=-shift, dims=0))
        else:
            shifted_idx = base_idx.clone()
        all_indices = torch.cat([base_idx, shifted_idx], dim=0)
    else:
        all_indices = base_idx
    all_valid_masks = all_indices < int(raw_n)

    full_data = SimpleNamespace(
        x=x,
        coords=coords,
        valid_mask=valid_mask,
        region_indices=region_indices,
        regions_h=regions_h,
        attn_kwargs={
            "coords": coords,
            "valid_mask": valid_mask,
            "region_indices": region_indices,
            "regions_h": regions_h,
        },
        decoder_sub_all_idx=all_indices,
        decoder_sub_valid_mask=all_valid_masks,
        decoder_overlap=decoder_overlap,
    )
    if layer is not None:
        full_data.layer = layer
    if build_targets:
        if particle_id is None or pt is None or reconstructable is None:
            raise ValueError("build_targets=True requires particle_id, pt, and reconstructable fields")
        y_true = (pt[:raw_n] >= 0.9) & (reconstructable[:raw_n] == 1)
        full_data.particle_id = particle_id
        full_data.pt = pt
        full_data.reconstructable = reconstructable
        full_data.targets = [[{"y_true": y_true}]]

    if not return_sub_events:
        return full_data, None
    global_particle_ids = None
    global_hit_counts = None
    if build_targets and raw_n > 0:
        global_particle_ids, global_hit_counts = torch.unique(
            particle_id[:raw_n],
            return_counts=True,
        )
    sub_events = [
        _build_sub_event(
            valid_mask_sub,
            idxs,
            [[{
                "labels": labels,
                "masks": masks,
                **({"hit_counts": hit_counts} if hit_counts is not None else {}),
            }]]
            if build_targets
            else None,
        )
        for idxs, valid_mask_sub in zip(all_indices, all_valid_masks)
        for sub_pid, sub_pt, sub_rec in [(
            particle_id.index_select(0, idxs) if build_targets else None,
            pt.index_select(0, idxs) if build_targets else None,
            reconstructable.index_select(0, idxs) if build_targets else None,
        )]
        for labels, masks, hit_counts in [(
            _build_cluster_masks(
                sub_pid,
                sub_pt,
                sub_rec,
                global_particle_ids=global_particle_ids,
                global_hit_counts=global_hit_counts,
            )
            if build_targets
            else (None, None, None)
        )]
    ]
    return full_data, sub_events


@torch.no_grad()
def process_data(
    data,
    fixed_len,
    encoder_block_size,
    decoder_block_size,
    regions,
    num_sub_events=3,
    decoder_overlap=True,
    random_split_shift=False,
    random_split_shift_max_frac=1.0,
    overlap_shift_jitter_frac=0.0,
    eta_abs_max=4.0,
    build_targets=False,
):
    """Full per-event preprocessing: eta filter → pad → region indices → sub-events."""
    data = _apply_eta_filter(data, eta_abs_max=eta_abs_max)
    return _split_event_into_sub_events(
        data,
        fixed_len=fixed_len,
        encoder_block_size=encoder_block_size,
        decoder_block_size=decoder_block_size,
        regions=regions,
        num_sub_events=num_sub_events,
        decoder_overlap=decoder_overlap,
        random_split_shift=random_split_shift,
        random_split_shift_max_frac=random_split_shift_max_frac,
        overlap_shift_jitter_frac=overlap_shift_jitter_frac,
        build_targets=build_targets,
    )
