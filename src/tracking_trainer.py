import argparse
import os
from functools import lru_cache
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from torch.amp import autocast

from utils.config_overrides import (
    apply_config_overrides,
    dump_yaml_config,
    load_yaml_config,
    resolve_tracking_config_path,
)
from utils import set_seed, get_optimizer, log, get_lr_scheduler, get_loss, setup_amp
from utils.block_size_utils import get_tracking_padded_lengths, resolve_block_sizes
from utils.get_data import get_data_loader, get_dataset
from utils.get_model import get_model
from utils.process_data_utils import (
    apply_setting_mask,
    build_cluster_masks,
    pad_to_fixed_len_all,
    build_region_indices,
)
from gnn_tracking_prev.metrics.cluster_metrics import tracking_metrics, tracking_metric_df
from torch_scatter import scatter_mean


DM_PT_THLD = 0.9
DM_PREDICTED_COUNT_THLD = 3
DM_MIN_TRACK_LENGTH = 0
DEFAULT_TRACKING_POSTPROCESSING = {
    "mode": "decoder_kmeans",
    "kmeans_iters": 3,
    "dbscan_eps": 0.5,
    "dbscan_min_samples": 1,
}
TRACKING_POSTPROCESSING_ALIASES = {
    "current": "decoder_kmeans",
    "decoder_kmeans": "decoder_kmeans",
    "encoder_kmeans": "encoder_kmeans",
    "encoder_dbscan": "encoder_dbscan",
    "decoder_assignments": "decoder_assignments",
    "raw_decoder": "decoder_assignments",
}
TRACKING_POSTPROCESSING_MODES = set(TRACKING_POSTPROCESSING_ALIASES.values())

DEFAULT_DM_EFF_ETA_BINS = np.arange(-4.0, 4.4, step=0.4)
DEFAULT_DM_EFF_PT_BINS = np.logspace(np.log10(DM_PT_THLD), np.log10(50.0), 10)
DEFAULT_GNN_TEXT_OUTPUT_DIR = Path("/storage/home/hcoda1/0/sgovil9/r-pli77-0/dec-sectorized/data/part9_out")
DEFAULT_GNN_PLOT_LABEL = "GNN"


def _unwrap_model(model):
    model = model.module if hasattr(model, "module") else model
    model = getattr(model, "_orig_mod", model)
    return model


def _as_list(obj):
    if obj is None:
        return []
    return obj if isinstance(obj, (list, tuple)) else [obj]


def _strip_checkpoint_prefixes(state_dict):
    """Normalize common wrapper prefixes from checkpoint keys."""
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in ("module.", "_orig_mod."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized[normalized_key] = value
    return normalized


def _count_state_dict_matches(load_result, target_state_dict_len, source_state_dict_len):
    missing_count = len(load_result.missing_keys)
    unexpected_count = len(load_result.unexpected_keys)
    matched_count = max(0, source_state_dict_len - unexpected_count)
    is_zero_match = missing_count == target_state_dict_len and unexpected_count == source_state_dict_len
    return matched_count, missing_count, unexpected_count, is_zero_match


def _format_tracking_metrics_for_log(metrics_dict):
    ordered_keys = [
        ("dm", "dm"),
        ("technical_efficiency", "tech_eff"),
        ("fake_rate", "fake_rate"),
        ("dup_rate", "dup_rate"),
    ]
    parts = []
    for key, label in ordered_keys:
        if key not in metrics_dict:
            continue
        try:
            value = float(metrics_dict[key])
        except (TypeError, ValueError):
            continue
        parts.append(f"{label}: {value:.4f}")
    return ", ".join(parts)


def _validate_hist_bins(bins, name):
    bins = np.asarray(bins, dtype=np.float64)
    if bins.ndim != 1 or bins.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 entries.")
    if np.any(np.diff(bins) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return bins


def _get_dm_eff_bins(config):
    pt_bins_cfg = config.get("dm_eff_pt_bins")
    eta_bins_cfg = config.get("dm_eff_eta_bins")
    pt_bins = DEFAULT_DM_EFF_PT_BINS if pt_bins_cfg is None else pt_bins_cfg
    eta_bins = DEFAULT_DM_EFF_ETA_BINS if eta_bins_cfg is None else eta_bins_cfg
    return _validate_hist_bins(pt_bins, "dm_eff_pt_bins"), _validate_hist_bins(eta_bins, "dm_eff_eta_bins")


def _resolve_gnn_reference_config(config):
    enabled = bool(config.get("evaluate_gnn_text_model", False))
    output_dir = Path(config.get("gnn_text_output_dir", DEFAULT_GNN_TEXT_OUTPUT_DIR))
    plot_label = str(config.get("gnn_text_plot_label", DEFAULT_GNN_PLOT_LABEL)).strip() or DEFAULT_GNN_PLOT_LABEL
    if enabled and not output_dir.exists():
        raise FileNotFoundError(f"GNN text output directory does not exist: {output_dir}")
    return {
        "enabled": enabled,
        "output_dir": output_dir,
        "plot_label": plot_label,
        "metric_prefix": "gnn",
    }


@lru_cache(maxsize=2048)
def _load_gnn_text_assignments(txt_path_str):
    assignments = {}
    with Path(txt_path_str).open("r") as handle:
        for track_id, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            for token in line.split():
                assignments[int(token)] = track_id
    if not assignments:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    hit_ids = np.fromiter(assignments.keys(), dtype=np.int64, count=len(assignments))
    track_ids = np.fromiter(assignments.values(), dtype=np.int64, count=len(assignments))
    return hit_ids, track_ids


def _resolve_tracking_postprocessing_config(postprocessing_config):
    resolved = dict(DEFAULT_TRACKING_POSTPROCESSING)
    if postprocessing_config is None:
        return resolved
    if isinstance(postprocessing_config, str):
        resolved["mode"] = postprocessing_config
    elif isinstance(postprocessing_config, dict):
        resolved.update(postprocessing_config)
    else:
        raise TypeError("postprocessing must be either a string, a mapping, or null.")

    mode = str(resolved.get("mode", DEFAULT_TRACKING_POSTPROCESSING["mode"])).strip().lower()
    mode = TRACKING_POSTPROCESSING_ALIASES.get(mode, mode)
    if mode not in TRACKING_POSTPROCESSING_MODES:
        raise ValueError(
            f"Unknown postprocessing mode '{resolved.get('mode')}'. "
            f"Expected one of {sorted(TRACKING_POSTPROCESSING_MODES)}."
        )
    resolved["mode"] = mode
    resolved["kmeans_iters"] = int(resolved.get("kmeans_iters", DEFAULT_TRACKING_POSTPROCESSING["kmeans_iters"]))
    resolved["dbscan_eps"] = float(resolved.get("dbscan_eps", DEFAULT_TRACKING_POSTPROCESSING["dbscan_eps"]))
    resolved["dbscan_min_samples"] = int(
        resolved.get("dbscan_min_samples", DEFAULT_TRACKING_POSTPROCESSING["dbscan_min_samples"])
    )
    if resolved["kmeans_iters"] < 1:
        raise ValueError("postprocessing.kmeans_iters must be >= 1.")
    if resolved["dbscan_eps"] <= 0:
        raise ValueError("postprocessing.dbscan_eps must be > 0.")
    if resolved["dbscan_min_samples"] < 1:
        raise ValueError("postprocessing.dbscan_min_samples must be >= 1.")
    return resolved


def _init_dm_eff_counts(pt_bins, eta_bins):
    return {
        "pt_true": np.zeros(len(pt_bins) - 1, dtype=np.float64),
        "pt_reco": np.zeros(len(pt_bins) - 1, dtype=np.float64),
        "eta_true": np.zeros(len(eta_bins) - 1, dtype=np.float64),
        "eta_reco": np.zeros(len(eta_bins) - 1, dtype=np.float64),
    }


def _efficiency_with_err(reco_counts, true_counts):
    true_counts = true_counts.astype(np.float64, copy=False)
    reco_counts = reco_counts.astype(np.float64, copy=False)
    eff = np.divide(
        reco_counts,
        true_counts,
        out=np.zeros_like(reco_counts, dtype=np.float64),
        where=true_counts > 0,
    )
    err = np.zeros_like(eff)
    valid = true_counts > 0
    err[valid] = np.sqrt(np.clip(eff[valid] * (1.0 - eff[valid]) / true_counts[valid], 0.0, None))
    return eff, err


def _plot_dm_efficiency_from_counts(
    true_counts,
    reco_counts,
    bins,
    *,
    x_label,
    x_scale="linear",
):
    eff, err = _efficiency_with_err(reco_counts, true_counts)
    xvals = 0.5 * (bins[1:] + bins[:-1])
    xerrs = 0.5 * (bins[1:] - bins[:-1])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        xvals,
        eff,
        xerr=xerrs,
        yerr=err,
        fmt="o",
        color="black",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("DM Efficiency")
    ax.set_ylim(0.0, 1.05)
    if x_scale == "log":
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _build_efficiency_plot(eff_counts_by_model, pt_bins, eta_bins, *, y_label):
    pt_xvals = 0.5 * (pt_bins[1:] + pt_bins[:-1])
    pt_xerrs = 0.5 * (pt_bins[1:] - pt_bins[:-1])
    eta_xvals = 0.5 * (eta_bins[1:] + eta_bins[:-1])
    eta_xerrs = 0.5 * (eta_bins[1:] - eta_bins[:-1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    styles = [
        {"fmt": "o", "color": "black"},
        {"fmt": "s", "color": "tab:orange"},
        {"fmt": "^", "color": "tab:blue"},
        {"fmt": "D", "color": "tab:green"},
    ]

    for idx, (label, eff_counts) in enumerate(eff_counts_by_model.items()):
        style = styles[idx % len(styles)]
        pt_eff, pt_err = _efficiency_with_err(eff_counts["pt_reco"], eff_counts["pt_true"])
        eta_eff, eta_err = _efficiency_with_err(eff_counts["eta_reco"], eff_counts["eta_true"])
        axes[0].errorbar(
            pt_xvals,
            pt_eff,
            xerr=pt_xerrs,
            yerr=pt_err,
            fmt=style["fmt"],
            color=style["color"],
            linestyle="none",
            capsize=3,
            label=label,
        )
        axes[1].errorbar(
            eta_xvals,
            eta_eff,
            xerr=eta_xerrs,
            yerr=eta_err,
            fmt=style["fmt"],
            color=style["color"],
            linestyle="none",
            capsize=3,
            label=label,
        )

    axes[0].set_xlabel(r"$p_T$ [GeV]")
    axes[0].set_ylabel(y_label)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r"$\eta$")
    axes[1].set_ylabel(y_label)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    if len(eff_counts_by_model) > 1:
        axes[0].legend()
        axes[1].legend()

    fig.tight_layout()
    return fig


def _all_reduce_dm_eff_counts(dm_eff_counts, device):
    if dist.is_available() and dist.is_initialized():
        reduced = {}
        for key, values in dm_eff_counts.items():
            tensor_values = torch.tensor(values, device=device, dtype=torch.float64)
            dist.all_reduce(tensor_values, op=dist.ReduceOp.SUM)
            reduced[key] = tensor_values.cpu().numpy()
        return reduced
    return dm_eff_counts


def _accumulate_dm_eff_counts(dst_counts, src_counts):
    for key in dst_counts:
        dst_counts[key] += src_counts[key]


@torch.no_grad()
def _build_gnn_reference_prediction(full_data, gnn_reference_config):
    if not gnn_reference_config["enabled"]:
        return None

    evtid = int(full_data.evtid.view(-1)[0].item())
    txt_path = gnn_reference_config["output_dir"] / f"event['{evtid:09d}'].txt"
    if not txt_path.exists():
        return None

    hit_ids, track_ids = _load_gnn_text_assignments(str(txt_path))
    predicted = np.full(full_data.particle_id.shape[0], -1, dtype=np.int64)
    valid_mask_np = full_data.valid_mask.bool().detach().cpu().numpy()
    valid_positions = np.flatnonzero(valid_mask_np)
    if valid_positions.size == 0 or hit_ids.size == 0:
        return torch.as_tensor(predicted, device=full_data.particle_id.device, dtype=torch.long)

    if hasattr(full_data, "hit_id") and full_data.hit_id is not None:
        current_hit_ids = full_data.hit_id.detach().cpu().numpy().reshape(-1)[valid_positions]
        order = np.argsort(current_hit_ids)
        sorted_hit_ids = current_hit_ids[order]
        candidate_idx = np.searchsorted(sorted_hit_ids, hit_ids)
        in_range = candidate_idx < sorted_hit_ids.size
        if in_range.any():
            candidate_idx_in_range = candidate_idx[in_range]
            exact = sorted_hit_ids[candidate_idx_in_range] == hit_ids[in_range]
            if exact.any():
                matched_valid_positions = valid_positions[order[candidate_idx_in_range[exact]]]
                predicted[matched_valid_positions] = track_ids[in_range][exact]
    else:
        in_range = (hit_ids >= 0) & (hit_ids < valid_positions.size)
        if in_range.any():
            predicted[valid_positions[hit_ids[in_range]]] = track_ids[in_range]

    return torch.as_tensor(predicted, device=full_data.particle_id.device, dtype=torch.long)


def _calc_efficiency_counts(
    predicted,
    data,
    pt_bins,
    eta_bins,
    efficiency_type="dm",
    pt_thld=DM_PT_THLD,
    predicted_count_thld=DM_PREDICTED_COUNT_THLD,
    min_track_length=DM_MIN_TRACK_LENGTH,
):
    eff_counts = _init_dm_eff_counts(pt_bins, eta_bins)

    predicted = np.asarray(predicted.detach().cpu().numpy()).reshape(-1)
    truth = np.asarray(data.particle_id.detach().cpu().numpy()).reshape(-1)
    pts = np.asarray(data.pt.detach().cpu().numpy()).reshape(-1)
    reconstructable = np.asarray(data.reconstructable.detach().cpu().numpy()).reshape(-1)
    if hasattr(data, "eta") and data.eta is not None:
        etas = np.asarray(data.eta.detach().cpu().numpy()).reshape(-1)
    elif hasattr(data, "coords") and data.coords is not None:
        coords = np.asarray(data.coords.detach().cpu().numpy())
        etas = np.asarray(coords[:, 0]).reshape(-1) if coords.ndim >= 2 else np.zeros_like(predicted, dtype=np.float64)
    else:
        etas = np.zeros_like(predicted, dtype=np.float64)

    valid_mask_np = None
    if hasattr(data, "valid_mask") and data.valid_mask is not None:
        valid_mask_np = np.asarray(data.valid_mask.detach().cpu().numpy()).reshape(-1).astype(bool)

    common_len = min(predicted.size, truth.size, pts.size, etas.size, reconstructable.size)
    if valid_mask_np is not None:
        common_len = min(common_len, valid_mask_np.size)
    if common_len == 0:
        return eff_counts

    predicted = predicted[:common_len]
    truth = truth[:common_len]
    pts = pts[:common_len]
    etas = etas[:common_len]
    reconstructable = reconstructable[:common_len]

    if valid_mask_np is not None:
        valid_mask_np = valid_mask_np[:common_len]
        predicted = predicted[valid_mask_np]
        truth = truth[valid_mask_np]
        pts = pts[valid_mask_np]
        etas = etas[valid_mask_np]
        reconstructable = reconstructable[valid_mask_np]

    if predicted.size == 0:
        return eff_counts

    h_df = pd.DataFrame(
        {
            "c": predicted,
            "id": truth,
            "pt": pts,
            "eta": etas,
            "reconstructable": reconstructable,
        }
    )
    if h_df.empty:
        return eff_counts

    pid_hit_counts = h_df["id"].value_counts()
    h_df["pid_n_hits"] = h_df["id"].map(pid_hit_counts)

    particles = (
        h_df.groupby("id", sort=False)
        .agg(
            pt=("pt", "mean"),
            eta=("eta", "mean"),
            reconstructable=("reconstructable", "mean"),
            pid_n_hits=("pid_n_hits", "first"),
        )
        .reset_index()
    )
    particles["is_reconstructable"] = (
        (particles["pt"] >= pt_thld)
        & (particles["reconstructable"] > 0.5)
        & (particles["pid_n_hits"] >= min_track_length)
    )

    c_df = tracking_metric_df(
        h_df[["c", "id", "pt", "eta", "reconstructable"]],
        predicted_count_thld=predicted_count_thld,
    )
    fiducial_cluster_mask = (
        c_df["valid_cluster"]
        & (c_df["maj_pt"] >= pt_thld)
        & (c_df["maj_reconstructable"] > 0.5)
        & (c_df["maj_pid_hits"] >= min_track_length)
    )
    if efficiency_type == "dm":
        reconstructed_mask = fiducial_cluster_mask & c_df["double_majority"]
    elif efficiency_type in {"technical", "tracking"}:
        reconstructed_mask = fiducial_cluster_mask & c_df["ATLAS_matched"]
    else:
        raise ValueError(f"Unknown efficiency_type '{efficiency_type}'")
    reconstructed_pids = set(c_df.loc[reconstructed_mask, "maj_pid"].to_numpy())
    particles["is_reconstructed"] = particles["id"].isin(reconstructed_pids)

    particles = particles[np.isfinite(particles["pt"]) & np.isfinite(particles["eta"])]
    true_particles = particles[particles["is_reconstructable"]]
    reco_particles = particles[particles["is_reconstructable"] & particles["is_reconstructed"]]

    eff_counts["pt_true"], _ = np.histogram(true_particles["pt"].to_numpy(), bins=pt_bins)
    eff_counts["pt_reco"], _ = np.histogram(reco_particles["pt"].to_numpy(), bins=pt_bins)
    eff_counts["eta_true"], _ = np.histogram(true_particles["eta"].to_numpy(), bins=eta_bins)
    eff_counts["eta_reco"], _ = np.histogram(reco_particles["eta"].to_numpy(), bins=eta_bins)
    return eff_counts


def trim_pred(pred, raw_size):
    if "pred_masks" in pred:
        if pred["pred_masks"].dim() == 4:
            pred["pred_masks"] = pred["pred_masks"][:, :, :raw_size, :]
        elif pred["pred_masks"].dim() == 3:
            pred["pred_masks"] = pred["pred_masks"][:, :, :raw_size]
    if "aux_outputs" in pred:
        for aux in pred["aux_outputs"]:
            if "pred_masks" in aux:
                if aux["pred_masks"].dim() == 4:
                    aux["pred_masks"] = aux["pred_masks"][:, :, :raw_size, :]
                elif aux["pred_masks"].dim() == 3:
                    aux["pred_masks"] = aux["pred_masks"][:, :, :raw_size]
            if "_serialized_point_order" in aux:
                aux["_serialized_point_order"] = aux["_serialized_point_order"][:raw_size]
    if "_serialized_point_order" in pred:
        pred["_serialized_point_order"] = pred["_serialized_point_order"][:raw_size]
    if "clf_probs" in pred:
        pred["clf_probs"] = pred["clf_probs"][:raw_size]
    if "clf_logits" in pred:
        pred["clf_logits"] = pred["clf_logits"][:raw_size]
    if "emb_all" in pred:
        pred["emb_all"] = pred["emb_all"][:raw_size]
    if "emb_all_labels" in pred:
        pred["emb_all_labels"] = pred["emb_all_labels"][:raw_size]
    return pred


def trim_pred_by_valid_mask(pred, valid_mask):
    valid_mask = valid_mask.bool()
    if "_serialized_point_order" in pred:
        raw_size = int(valid_mask.sum().item())
        return trim_pred(pred, raw_size)
    if "pred_masks" in pred:
        if pred["pred_masks"].dim() == 4:
            pred["pred_masks"] = pred["pred_masks"][:, :, valid_mask, :]
        elif pred["pred_masks"].dim() == 3:
            pred["pred_masks"] = pred["pred_masks"][:, :, valid_mask]
    if "aux_outputs" in pred:
        for aux in pred["aux_outputs"]:
            if "pred_masks" in aux:
                if aux["pred_masks"].dim() == 4:
                    aux["pred_masks"] = aux["pred_masks"][:, :, valid_mask, :]
                elif aux["pred_masks"].dim() == 3:
                    aux["pred_masks"] = aux["pred_masks"][:, :, valid_mask]
    return pred


def trim_targets_by_valid_mask(targets, valid_mask):
    valid_mask = valid_mask.bool()
    for tgt in targets:
        if "masks" in tgt:
            tgt["masks"] = tgt["masks"][:, valid_mask]
    return targets


def _split_event_into_sub_events(
    data,
    fixed_len,
    encoder_block_size,
    decoder_block_size,
    regions,
    num_sub_events=3,
    decoder_overlap=True,
):
    """
    Split one event along phi into decoder sub-events.

    If decoder_overlap=True, create 2*num_sub_events windows:
    num_sub_events base windows + num_sub_events shifted windows.
    If decoder_overlap=False, create only num_sub_events non-overlapping base windows.
    Returns padded full-event data and a list of sub-event labels/indices.
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
        data, valid_mask = pad_to_fixed_len_all(
            data,
            valid_mask.numel() + encoder_block_size,
            encoder_block_size,
        )
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
        # Keep the number of decoder windows identical across DDP ranks.
        # Some events can be fully masked out after preprocessing, and
        # dropping the shifted windows only on those ranks causes the
        # per-sub-event loss all_reduce sequence to diverge.
        if n_valid > 0:
            shift = max(1, n_valid // (2 * num_sub_events))
            shifted_valid = torch.roll(sorted_valid, shifts=-shift, dims=0)
            shifted_idx = _build_indices(shifted_valid)
        else:
            shifted_idx = base_idx.clone()
        all_indices = torch.cat([base_idx, shifted_idx], dim=0)
    else:
        all_indices = base_idx

    sub_events = []
    for idxs in all_indices:
        valid_mask_sub = valid_mask.index_select(0, idxs)
        sub_pid = data.particle_id.index_select(0, idxs)
        sub_pt = data.pt.index_select(0, idxs)
        sub_rec = data.reconstructable.index_select(0, idxs)
        sub_data = type("SubData", (), {"particle_id": sub_pid, "pt": sub_pt, "reconstructable": sub_rec})()
        sub_labels, sub_masks, _, _ = build_cluster_masks(sub_data)
        sub_events.append(
            {
                "all_idx": idxs,
                "orig_idx": idxs[valid_mask_sub],
                "valid_mask": valid_mask_sub,
                "targets": [[{"labels": sub_labels, "masks": sub_masks}]],
            }
        )

    data.decoder_overlap = decoder_overlap
    return data, sub_events


def build_infonce_inputs(pred, data):
    if "emb_all" not in pred:
        return None

    emb_all = pred["emb_all"]
    labels_all = pred.get("emb_all_labels", data.particle_id)
    y_true = data.targets[0][0]["y_true"]

    if labels_all.device != emb_all.device:
        labels_all = labels_all.to(emb_all.device)
    if y_true.device != emb_all.device:
        y_true = y_true.to(emb_all.device)

    if emb_all.shape[0] != y_true.shape[0]:
        min_len = min(emb_all.shape[0], y_true.shape[0])
        emb_all = emb_all[:min_len]
        labels_all = labels_all[:min_len]
        y_true = y_true[:min_len]

    emb = emb_all[y_true]
    emb_labels = labels_all[y_true]
    return emb, emb_all, emb_labels, labels_all, y_true


def weighted_loss(losses, criterion):
    for k in list(losses.keys()):
        if k in criterion.weight_dict:
            losses[k] *= criterion.weight_dict[k]
        else:
            print(f"Warning: {k} not in weight_dict, removing from losses")
            # remove this loss if not specified in `weight_dict`
            losses.pop(k)
    all_losses = sum(losses.values())
    all_losses_item_dict = {k: v.item() for k, v in losses.items()}
    return all_losses, all_losses_item_dict


def calc_clf_loss(pred, data, criterion):
    y_pred = pred["clf_logits"].squeeze(-1)
    y_true = (data.targets[0][0]["y_true"]).float()

    pos_weight = y_true * (criterion.weight_dict["clf_pos_weight"] - 1.0) + 1.0
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=pos_weight)
    # loss = torch.tensor(0.0, device=y_pred.device)

    if "aux_clf_outputs" in pred:
        for aux_output in pred["aux_clf_outputs"]:
            aux_y_pred = aux_output["clf_logits"].squeeze(-1)
            aux_loss = F.binary_cross_entropy_with_logits(aux_y_pred, y_true, weight=pos_weight)
            loss += aux_loss
    try:
        auroc = roc_auc_score(y_true.detach().cpu().numpy(), y_pred.sigmoid().detach().float().cpu().numpy())
    except ValueError:
        auroc = 0.0
    # acc = ((y_pred > 0.5) == y_true).float().mean().item()
    recall = recall_score(y_true.detach().cpu().numpy(), (y_pred.sigmoid() > 0.1).detach().float().cpu().numpy())
    precision = precision_score(y_true.detach().cpu().numpy(), (y_pred.sigmoid() > 0.1).detach().float().cpu().numpy())
    return loss * criterion.weight_dict["clf_loss"], auroc, recall, precision


def calc_infonce_loss(
    embeddings, emb_all, labels, labels_all, y_true, criterion, k=256, chunk_size=5000, temperature=0.07
):
    device = embeddings.device
    N, D = embeddings.shape
    # return torch.tensor(0.0, device=device)
    # 1) L2-normalize once
    # emb_norm = F.normalize(embeddings, p=2, dim=1)  # [N, D]
    emb_norm = embeddings

    # 2) figure out which anchors actually have >1 in their cluster
    _, inv = torch.unique(labels, return_inverse=True)
    counts = torch.bincount(inv)
    cluster_size = counts[inv]  # [N]
    valid_anchor = cluster_size > 1

    # 3) storage for per-anchor loss
    loss_per_anchor = torch.empty(N, device=device)

    pos_map = torch.nonzero(y_true, as_tuple=False).squeeze(1)  # shape [N]
    # 4) chunked processing
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        M = end - start

        # a) compute similarity between this chunk and all embeddings
        chunk_emb = emb_norm[start:end]  # [M, D]
        # sim_chunk = chunk_emb @ emb_norm.t()           # [M, N]
        # l2_dist = torch.linalg.norm(chunk_emb[:, None] - emb_norm[None], ord=2, dim=-1)
        # l2_dist = torch.cdist(chunk_emb.unsqueeze(0), emb_all.unsqueeze(0), p=2).squeeze(0)  # [M, N]
        # sigma = 0.75
        # sim_chunk = torch.exp(-l2_dist / (2 * sigma**2))

        sim_chunk = torch.einsum("md,nd->mn", chunk_emb, emb_all)  # [M, N]
        sim_chunk = sim_chunk / temperature  # scale

        # b) mask out self-similarity
        self_indices = pos_map[start:end]
        rows = torch.arange(M, device=device)  # [0,1,...,M-1]
        sim_chunk[rows, self_indices] = float("-inf")

        # c) build positive mask and sum exp(sim) over **all** same-label
        #    (self is -inf, so excluded)
        lbl_chunk = labels[start:end].unsqueeze(1)  # [M, 1]
        pos_mask = lbl_chunk == labels_all.unsqueeze(0)  # [M, N]

        # d) log-sum-exp over **all** positives
        sim_pos = sim_chunk.masked_fill(~pos_mask, float("-inf"))  # [M, N]
        log_pos = torch.logsumexp(sim_pos, dim=1)  # [M]

        # e) top-k negatives among rest
        sim_neg = sim_chunk.masked_fill(pos_mask, float("-inf"))  # [M, N]
        topk_neg_vals, _ = sim_neg.topk(k, dim=1, largest=True)  # [M, k]
        log_neg = torch.logsumexp(topk_neg_vals, dim=1)  # [M]

        # f) denominator: logsumexp([log_pos, log_neg])
        log_den = torch.logsumexp(torch.stack([log_pos, log_neg], dim=1), dim=1)  # [M]

        # g) per-anchor loss = –(log_pos – log_den)
        loss_per_anchor[start:end] = -(log_pos - log_den)

    # 5) average over anchors with ≥1 positive
    if valid_anchor.sum() == 0:
        return torch.tensor(0.0, device=device)

    losses = loss_per_anchor[valid_anchor]  # [N_valid]
    lbls = labels[valid_anchor]  # [N_valid]

    # map labels→0..C-1
    _, inv_lbls = torch.unique(lbls, return_inverse=True)
    # mean loss per cluster
    cluster_losses = scatter_mean(losses, inv_lbls, dim=0)  # [C]
    # final: equal‐weight average over clusters
    loss = cluster_losses.mean()
    return loss * criterion.weight_dict["nce_weight"]

def get_pred_masks(pred):
    emb_all = pred["emb_all"].detach()
    clf_scores = pred["clf_probs"].detach().squeeze(-1)
    point_order = pred.get("_serialized_point_order")
    if point_order is not None:
        point_order = point_order.detach()
    return pred["pred_masks"].detach(), pred["pred_logits"].detach(), emb_all, clf_scores, point_order


def train_one_batch(model, optimizer, criterion, data, lr_s, amp_enabled=False, amp_dtype=torch.float16, scaler=None):
    model.train()
    optimizers = _as_list(optimizer)
    for opt in optimizers:
        opt.zero_grad()

    full_data, sub_events = data
    base_model = _unwrap_model(model)
    raw_masks = []
    total_loss = 0.0
    loss_dict_accum = defaultdict(list)

    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
        enc_out = base_model.encode(full_data)
        full_data.emb_all = enc_out["emb_all"].detach()
        full_data.encoder_features = enc_out["encoded"].detach()
        full_data.clf_scores = enc_out["clf_probs"].detach()
        for sub in sub_events:
            sub_feats = enc_out["encoded"][sub["all_idx"]]
            sub_coords = full_data.coords[sub["all_idx"]]
            pred = base_model.decode(sub_feats, sub_coords, sub["valid_mask"])
            pred = trim_pred_by_valid_mask(pred, sub["valid_mask"])
            sub_targets = trim_targets_by_valid_mask(deepcopy(sub["targets"][0]), sub["valid_mask"])

            loss = criterion(pred, sub_targets)
            loss, loss_dict = weighted_loss(loss, criterion)

            total_loss = total_loss + loss
            for k, v in loss_dict.items():
                loss_dict_accum[k].append(v)

            pred_masks = pred["pred_masks"].detach()
            pred_logits = pred["pred_logits"].detach()
            point_order = pred.get("_serialized_point_order")
            if point_order is not None:
                point_order = point_order.detach()
            raw_masks.append((pred_masks, pred_logits, point_order))

        pred_full = {
            "clf_probs": enc_out["clf_probs"],
            "clf_logits": enc_out["clf_logits"],
            "emb_all": enc_out["emb_all"],
            "emb_all_labels": enc_out["emb_all_labels"],
        }
        y_true = full_data.targets[0][0]["y_true"]
        pred_full = trim_pred(pred_full, y_true.shape[0])

        if "clf_logits" in pred_full:
            clf_loss_value, auroc, recall, prec = calc_clf_loss(pred_full, full_data, criterion)
            total_loss = total_loss + clf_loss_value
            loss_dict_accum["clf_loss"].append(clf_loss_value.item())
            loss_dict_accum["roc"].append(auroc)
            loss_dict_accum["rec"].append(recall)
            loss_dict_accum["prec"].append(prec)

        infonce_inputs = build_infonce_inputs(pred_full, full_data)
        if infonce_inputs is not None:
            emb, emb_all, emb_labels, labels_all, y_true = infonce_inputs
            infonce_loss = calc_infonce_loss(emb, emb_all, emb_labels, labels_all, y_true, criterion)
            total_loss = total_loss + infonce_loss
            loss_dict_accum["infonce"].append(infonce_loss.item())

    if scaler is not None and scaler.is_enabled():
        scaler.scale(total_loss).backward()
        for opt in optimizers:
            scaler.unscale_(opt)
        max_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
    else:
        total_loss.backward()
        max_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # print(f"Max grad norm: {max_norm:.4f}")
        for opt in optimizers:
            opt.step()

    if lr_s is not None:
        schedulers = _as_list(lr_s)
        for scheduler in schedulers:
            if isinstance(scheduler, LambdaLR):
                scheduler.step()

    loss_dict_out = {k: float(np.mean(v)) for k, v in loss_dict_accum.items()} if loss_dict_accum else {}
    loss_dict_out["total_loss"] = float(total_loss.item()) if hasattr(total_loss, "item") else float(total_loss)
    loss_dict_out["max_norm"] = max_norm.item()
    return loss_dict_out, raw_masks


@torch.no_grad()
def eval_one_batch(model, optimizer, criterion, data, lr_s, amp_enabled=False, amp_dtype=torch.float16, scaler=None):
    model.eval()
    full_data, sub_events = data
    base_model = _unwrap_model(model)
    raw_masks = []
    total_loss = 0.0
    loss_dict_accum = defaultdict(list)

    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
        enc_out = base_model.encode(full_data)
        full_data.emb_all = enc_out["emb_all"].detach()
        full_data.encoder_features = enc_out["encoded"].detach()
        full_data.clf_scores = enc_out["clf_probs"].detach()
        for sub in sub_events:
            sub_feats = enc_out["encoded"][sub["all_idx"]]
            sub_coords = full_data.coords[sub["all_idx"]]
            pred = base_model.decode(sub_feats, sub_coords, sub["valid_mask"])
            pred = trim_pred_by_valid_mask(pred, sub["valid_mask"])
            sub_targets = trim_targets_by_valid_mask(deepcopy(sub["targets"][0]), sub["valid_mask"])

            loss = criterion(pred, sub_targets)
            loss, loss_dict = weighted_loss(loss, criterion)

            total_loss = total_loss + loss
            for k, v in loss_dict.items():
                loss_dict_accum[k].append(v)

            pred_masks = pred["pred_masks"].detach()
            pred_logits = pred["pred_logits"].detach()
            point_order = pred.get("_serialized_point_order")
            if point_order is not None:
                point_order = point_order.detach()
            raw_masks.append((pred_masks, pred_logits, point_order))

        pred_full = {
            "clf_probs": enc_out["clf_probs"],
            "clf_logits": enc_out["clf_logits"],
            "emb_all": enc_out["emb_all"],
            "emb_all_labels": enc_out["emb_all_labels"],
        }
        y_true = full_data.targets[0][0]["y_true"]
        pred_full = trim_pred(pred_full, y_true.shape[0])

        if "clf_logits" in pred_full:
            clf_loss_value, auroc, recall, prec = calc_clf_loss(pred_full, full_data, criterion)
            total_loss = total_loss + clf_loss_value
            loss_dict_accum["clf_loss"].append(clf_loss_value.item())
            loss_dict_accum["roc"].append(auroc)
            loss_dict_accum["rec"].append(recall)
            loss_dict_accum["prec"].append(prec)

        infonce_inputs = build_infonce_inputs(pred_full, full_data)
        if infonce_inputs is not None:
            emb, emb_all, emb_labels, labels_all, y_true = infonce_inputs
            infonce_loss = calc_infonce_loss(emb, emb_all, emb_labels, labels_all, y_true, criterion)
            total_loss = total_loss + infonce_loss
            loss_dict_accum["infonce"].append(infonce_loss.item())

    loss_dict_out = {k: float(np.mean(v)) for k, v in loss_dict_accum.items()} if loss_dict_accum else {}
    loss_dict_out["total_loss"] = float(total_loss.item()) if hasattr(total_loss, "item") else float(total_loss)
    loss_dict_out["max_norm"] = 0.0  # No gradients in eval mode
    return loss_dict_out, raw_masks


@torch.no_grad()
def _compute_clf_keep_mask(full_data, full_clf_scores, clf_recall=0.995):
    if full_clf_scores is None:
        raise ValueError("Tracking postprocessing requires full_clf_scores.")
    clf_scores_full = full_clf_scores.squeeze(-1).float()
    valid_mask_full = full_data.valid_mask.bool()
    y_true_full = full_data.targets[0][0]["y_true"]
    clf_scores_valid = clf_scores_full[valid_mask_full]
    pos_scores = clf_scores_valid[y_true_full]
    if pos_scores.numel() == 0:
        return valid_mask_full
    clf_thres = torch.quantile(pos_scores, 1.0 - clf_recall)
    return (clf_scores_full >= clf_thres) & valid_mask_full


@torch.no_grad()
def _build_decoder_assignments(pred_list, sub_events, full_data, clf_keep, mask_thres=0.0):
    device = full_data.x.device
    n_total = full_data.x.shape[0]
    max_ids = -1
    max_prob = torch.full((n_total,), float("-inf"), device=device)
    predicted = torch.full((n_total,), max_ids, device=device, dtype=torch.long)
    total_queries = 0
    decoder_overlap = bool(getattr(full_data, "decoder_overlap", True))

    for pred, sub in zip(pred_list, sub_events):
        if len(pred) == 3:
            pred_masks, pred_logits, point_order = pred
        else:
            pred_masks, pred_logits = pred
            point_order = None
        mask_scores = (pred_logits.softmax(-1)[..., 0].unsqueeze(-1).unsqueeze(-1) * pred_masks.sigmoid()).float()
        mask_scores = mask_scores[0, ..., 0]  # [Q, N_sub]
        orig_idx = sub["orig_idx"]
        valid_mask_sub = sub["valid_mask"].bool()
        n_sub_padded = valid_mask_sub.numel()
        n_sub_valid = orig_idx.numel()
        n_scores = mask_scores.shape[1]
        if n_scores == n_sub_padded:
            mask_scores = mask_scores.masked_fill(~valid_mask_sub, float("-inf"))
            sub_mask_mode = "padded"
        elif n_scores == n_sub_valid:
            sub_mask_mode = "trimmed"
        else:
            raise RuntimeError(
                f"Incompatible sub-event shapes in metric merge: "
                f"mask_scores={n_scores}, valid_mask_sub={n_sub_padded}, orig_idx={n_sub_valid}"
            )

        if point_order is not None and sub_mask_mode == "trimmed":
            point_order = point_order.to(device=mask_scores.device, dtype=torch.long)
            if point_order.numel() != n_sub_valid:
                raise RuntimeError(
                    f"Serialized point order length {point_order.numel()} does not match trimmed mask width {n_sub_valid}."
                )
            inverse_order = torch.empty_like(point_order)
            inverse_order.scatter_(
                0,
                point_order,
                torch.arange(point_order.numel(), device=point_order.device, dtype=point_order.dtype),
            )
            mask_scores = mask_scores.index_select(1, inverse_order)

        scores = mask_scores
        if mask_thres > 0:
            valid_q = (scores > mask_thres).any(dim=1)
            if not valid_q.any():
                continue
            scores = scores[valid_q]

        scores_max, idx_max = scores.max(dim=0)
        if sub_mask_mode == "padded":
            scores_max = scores_max[valid_mask_sub]
            idx_max = idx_max[valid_mask_sub]
        offset = total_queries
        total_queries += scores.shape[0]

        if decoder_overlap:
            better = scores_max > max_prob[orig_idx]
            if better.any():
                idx = orig_idx[better]
                max_prob[idx] = scores_max[better]
                predicted[idx] = offset + idx_max[better]
        else:
            if (max_prob[orig_idx] > float("-inf")).any():
                raise RuntimeError("decoder_overlap=False expected unique orig_idx across sub-events.")
            max_prob[orig_idx] = scores_max
            predicted[orig_idx] = offset + idx_max

    thres = 0.0
    predicted[max_prob < thres] = max_ids
    predicted[~clf_keep] = max_ids
    return predicted


@torch.no_grad()
def _compute_tracking_metrics(
    predicted,
    full_data,
    pt_thld=DM_PT_THLD,
    predicted_count_thld=DM_PREDICTED_COUNT_THLD,
    min_track_length=DM_MIN_TRACK_LENGTH,
):
    valid_mask = full_data.valid_mask.bool()
    predicted = predicted[valid_mask]
    truth = full_data.particle_id[valid_mask]
    pts = full_data.pt[valid_mask]
    reconstructable = full_data.reconstructable[valid_mask]

    res = tracking_metrics(
        truth=truth.cpu().numpy(),
        predicted=predicted.cpu().numpy(),
        pts=pts.cpu().numpy(),
        reconstructable=reconstructable.cpu().numpy(),
        pt_thlds=[pt_thld],
        predicted_count_thld=predicted_count_thld,
        min_track_length=min_track_length,
    )
    metrics = res[pt_thld]
    parsed = {}
    for metric_name, metric_value in metrics.items():
        parsed_key = "dm" if metric_name == "double_majority" else metric_name
        parsed[parsed_key] = float(metric_value)
    return parsed


def _prepare_tracking_features(features):
    return F.normalize(features.float(), p=2, dim=1, eps=1e-12)


@torch.no_grad()
def _kmeans_refine_assignments(init_pred, features, iters):
    active_mask = init_pred >= 0
    if not active_mask.any():
        return init_pred
    active_ids, inv = torch.unique(init_pred[active_mask], return_inverse=True)
    k = active_ids.numel()
    if k <= 1:
        return init_pred
    points = _prepare_tracking_features(features)[active_mask]
    centroids = scatter_mean(points, inv, dim=0, dim_size=k)
    centroids = F.normalize(centroids, p=2, dim=1, eps=1e-12)
    assign = inv
    for _ in range(iters):
        sim = points @ centroids.t()
        assign = sim.argmax(dim=1)
        new_centroids = scatter_mean(points, assign, dim=0, dim_size=k)
        counts = torch.bincount(assign, minlength=k).to(points.dtype).unsqueeze(1)
        new_centroids = F.normalize(new_centroids, p=2, dim=1, eps=1e-12)
        centroids = torch.where(counts > 0, new_centroids, centroids)
    refined = init_pred.clone()
    refined[active_mask] = active_ids[assign]
    return refined


@torch.no_grad()
def _kmeans_cluster_features(features, active_mask, num_clusters, iters):
    predicted = torch.full((features.shape[0],), -1, device=features.device, dtype=torch.long)
    if not active_mask.any() or num_clusters <= 0:
        return predicted
    active_idx = active_mask.nonzero(as_tuple=False).flatten()
    num_clusters = min(int(num_clusters), active_idx.numel())
    if num_clusters <= 0:
        return predicted
    if num_clusters == 1:
        predicted[active_idx] = 0
        return predicted

    points = _prepare_tracking_features(features)[active_idx]
    init_ids = torch.div(
        torch.arange(num_clusters, device=points.device) * points.shape[0],
        num_clusters,
        rounding_mode="floor",
    )
    centroids = points.index_select(0, init_ids).clone()
    assign = torch.zeros(points.shape[0], device=points.device, dtype=torch.long)

    for _ in range(iters):
        sim = points @ centroids.t()
        assign = sim.argmax(dim=1)
        new_centroids = scatter_mean(points, assign, dim=0, dim_size=num_clusters)
        counts = torch.bincount(assign, minlength=num_clusters).to(points.dtype).unsqueeze(1)
        new_centroids = F.normalize(new_centroids, p=2, dim=1, eps=1e-12)
        centroids = torch.where(counts > 0, new_centroids, centroids)

    predicted[active_idx] = assign
    return predicted


@torch.no_grad()
def _dbscan_cluster_features(features, active_mask, eps, min_samples):
    predicted = torch.full((features.shape[0],), -1, device=features.device, dtype=torch.long)
    if not active_mask.any():
        return predicted
    active_idx = active_mask.nonzero(as_tuple=False).flatten()
    points = _prepare_tracking_features(features)[active_idx].detach().cpu().numpy()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    predicted[active_idx] = torch.as_tensor(labels, device=features.device, dtype=torch.long)
    return predicted


@torch.no_grad()
def calc_tracking_metrics_merged(
    pred_list,
    sub_events,
    mask_thres=0.0,
    full_data=None,
    full_emb=None,
    full_encoder_features=None,
    full_clf_scores=None,
    postprocessing_config=None,
    pt_thld=DM_PT_THLD,
    predicted_count_thld=DM_PREDICTED_COUNT_THLD,
    min_track_length=DM_MIN_TRACK_LENGTH,
    return_prediction=False,
):
    postprocessing_config = _resolve_tracking_postprocessing_config(postprocessing_config)
    mode = postprocessing_config["mode"]
    clf_keep = _compute_clf_keep_mask(full_data, full_clf_scores)

    decoder_assignments = None
    decoder_cluster_count = 0
    if mode in {"decoder_kmeans", "encoder_kmeans", "decoder_assignments"}:
        decoder_assignments = _build_decoder_assignments(
            pred_list,
            sub_events,
            full_data,
            clf_keep,
            mask_thres=mask_thres,
        )
        decoder_cluster_count = int(torch.unique(decoder_assignments[decoder_assignments >= 0]).numel())

    if mode == "decoder_kmeans":
        if full_emb is None:
            raise ValueError("decoder_kmeans postprocessing requires full_emb.")
        predicted = _kmeans_refine_assignments(
            decoder_assignments,
            full_emb,
            iters=postprocessing_config["kmeans_iters"],
        )
    elif mode == "encoder_kmeans":
        if full_encoder_features is None:
            raise ValueError("encoder_kmeans postprocessing requires full_encoder_features.")
        predicted = _kmeans_cluster_features(
            full_encoder_features,
            clf_keep,
            num_clusters=decoder_cluster_count,
            iters=postprocessing_config["kmeans_iters"],
        )
    elif mode == "encoder_dbscan":
        if full_encoder_features is None:
            raise ValueError("encoder_dbscan postprocessing requires full_encoder_features.")
        predicted = _dbscan_cluster_features(
            full_encoder_features,
            clf_keep,
            eps=postprocessing_config["dbscan_eps"],
            min_samples=postprocessing_config["dbscan_min_samples"],
        )
    elif mode == "decoder_assignments":
        predicted = decoder_assignments
    else:
        raise ValueError(f"Unsupported postprocessing mode '{mode}'.")

    metrics = _compute_tracking_metrics(
        predicted,
        full_data,
        pt_thld=pt_thld,
        predicted_count_thld=predicted_count_thld,
        min_track_length=min_track_length,
    )
    if return_prediction:
        return metrics, predicted
    return metrics


@torch.no_grad()
def calc_dm_merged(
    pred_list,
    sub_events,
    mask_thres=0.0,
    full_data=None,
    full_emb=None,
    full_encoder_features=None,
    full_clf_scores=None,
    postprocessing_config=None,
):
    return calc_tracking_metrics_merged(
        pred_list,
        sub_events,
        mask_thres=mask_thres,
        full_data=full_data,
        full_emb=full_emb,
        full_encoder_features=full_encoder_features,
        full_clf_scores=full_clf_scores,
        postprocessing_config=postprocessing_config,
    )["dm"]


@torch.no_grad()
def process_data(
    data,
    epoch,
    setting="all",
    epoch_to_sample=50,
    good_pt_thld=0.9,
    fixed_len=None,
    encoder_block_size=None,
    decoder_block_size=None,
    regions=None,
    decoder_overlap=True,
):
    if setting == "all":
        condition = torch.ones_like(data.pt, dtype=torch.bool)
    elif setting == "only_good":
        condition = (data.pt >= good_pt_thld) & (data.reconstructable == 1.0)
    elif setting == "sample_points":
        condition = data.pt >= good_pt_thld
        noise_samples = ~condition
        ratio_to_sample = min(1.0, (epoch + 1) / epoch_to_sample)

        num_samples = int(noise_samples.sum() * ratio_to_sample)
        noise_indices = torch.nonzero(noise_samples, as_tuple=False).squeeze()
        sampled_indices = noise_indices[torch.randperm(len(noise_indices))[:num_samples]]
        condition[sampled_indices] = True

    elif setting == "sample_pt_fmin":
        condition = data.pt >= good_pt_thld
        samples_sorted_by_pt, arg_samples_sorted_by_pt = data.pt.sort()
        bad_samples_sorted_by_pt = samples_sorted_by_pt[~condition[arg_samples_sorted_by_pt]]
        arg_bad_samples_sorted_by_pt = arg_samples_sorted_by_pt[~condition[arg_samples_sorted_by_pt]]

        ratio_to_sample = min(1.0, (epoch + 1) / epoch_to_sample)
        num_samples = int(len(bad_samples_sorted_by_pt) * ratio_to_sample)
        sampled_indices = (
            arg_bad_samples_sorted_by_pt[:num_samples]
            if num_samples < len(bad_samples_sorted_by_pt)
            else arg_bad_samples_sorted_by_pt
        )
        condition[sampled_indices] = True

    elif setting == "sample_pt_fmax":
        ratio_to_sample = min(1.0, (epoch + 1) / epoch_to_sample)
        cur_thld = good_pt_thld * (1.0 - ratio_to_sample)
        condition = data.pt >= cur_thld

    elif setting == "sample_phi":
        condition = data.pt >= good_pt_thld
        phi, arg_phi = data.pos[:, 1].sort()
        ratio_to_sample = min(1.0, (epoch + 1) / epoch_to_sample)
        num_samples = int(len(phi) * ratio_to_sample)
        sampled_indices = arg_phi[:num_samples] if num_samples < len(phi) else arg_phi
        condition[sampled_indices] = True
    else:
        raise ValueError(f"Unknown data setting: {setting}")

    condition = condition & (data.eta.abs() <= 4.0)
    data = apply_setting_mask(data, condition)

    full_data, sub_events = _split_event_into_sub_events(
        data,
        fixed_len=fixed_len,
        encoder_block_size=encoder_block_size,
        decoder_block_size=decoder_block_size,
        regions=regions,
        decoder_overlap=decoder_overlap,
    )
    return full_data, sub_events


def run_one_epoch(
    model,
    optimizer,
    criterion,
    data_loader,
    phase,
    epoch,
    device,
    lr_s,
    config,
    amp_enabled=False,
    amp_dtype=torch.float16,
    scaler=None,
    sampler=None,
    collect_dm_efficiency=False,
    collect_tracking_efficiency=False,
    dm_eff_pt_bins=None,
    dm_eff_eta_bins=None,
    gnn_reference_config=None,
):
    # save gpu memory usage
    torch.cuda.empty_cache()

    if sampler is not None and phase == "train":
        sampler.set_epoch(epoch)

    run_one_batch = train_one_batch if phase == "train" else eval_one_batch
    phase = "test " if phase == "test" else phase

    is_main_process = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    pbar = tqdm(data_loader, disable=not (is_main_process and __name__ == "__main__"))
    total_losses, total_track_metrics = defaultdict(list), defaultdict(list)
    reference_metric_totals = defaultdict(float)
    reference_metric_counts = defaultdict(int)
    reference_metric_names = None
    postprocessing_config = config["postprocessing"]
    gnn_reference_config = gnn_reference_config or _resolve_gnn_reference_config({})
    primary_plot_label = str(config.get("dm_efficiency_primary_label", config.get("model_name", "current"))).strip()
    primary_plot_label = primary_plot_label or config.get("model_name", "current")
    reference_plot_label = gnn_reference_config["plot_label"]
    if reference_plot_label == primary_plot_label:
        reference_plot_label = f"{reference_plot_label} (GNN)"
    dm_eff_counts = None
    tracking_eff_counts = None
    reference_dm_eff_event_count = 0
    collect_any_efficiency = collect_dm_efficiency or collect_tracking_efficiency
    if collect_any_efficiency:
        if dm_eff_pt_bins is None or dm_eff_eta_bins is None:
            raise ValueError("dm_eff_pt_bins and dm_eff_eta_bins must be provided when collecting efficiency plots.")
        if collect_dm_efficiency:
            dm_eff_counts = {primary_plot_label: _init_dm_eff_counts(dm_eff_pt_bins, dm_eff_eta_bins)}
            if gnn_reference_config["enabled"]:
                dm_eff_counts[reference_plot_label] = _init_dm_eff_counts(dm_eff_pt_bins, dm_eff_eta_bins)
        if collect_tracking_efficiency:
            tracking_eff_counts = {primary_plot_label: _init_dm_eff_counts(dm_eff_pt_bins, dm_eff_eta_bins)}
            if gnn_reference_config["enabled"]:
                tracking_eff_counts[reference_plot_label] = _init_dm_eff_counts(dm_eff_pt_bins, dm_eff_eta_bins)
    for idx, data in enumerate(pbar):
        data = data.clone()
        data = data.to(device)
        base_model = _unwrap_model(model)
        encoder_block_size, decoder_block_size = resolve_block_sizes(config["model_kwargs"])
        data = process_data(
            data,
            epoch,
            setting=config["data_setting"],
            epoch_to_sample=config["epoch_to_sample"],
            good_pt_thld=config["good_pt_thld"],
            fixed_len=config["model_kwargs"]["fixed_seq_len"],
            encoder_block_size=encoder_block_size,
            decoder_block_size=decoder_block_size,
            regions=getattr(base_model, "helper_funcs", {}).get("regions"),
            decoder_overlap=config["model_kwargs"].get("decoder_overlap", True),
        )

        batch_loss_dict, batch_masks = run_one_batch(
            model, optimizer, criterion, data, lr_s, amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler
        )
        full_data, sub_events = data
        if collect_any_efficiency:
            batch_track_metrics, batch_prediction = calc_tracking_metrics_merged(
                batch_masks,
                sub_events,
                full_data=full_data,
                full_emb=full_data.emb_all,
                full_encoder_features=full_data.encoder_features,
                full_clf_scores=full_data.clf_scores,
                postprocessing_config=postprocessing_config,
                return_prediction=True,
            )
            if collect_dm_efficiency and dm_eff_counts is not None:
                batch_dm_eff_counts = _calc_efficiency_counts(
                    batch_prediction,
                    full_data,
                    dm_eff_pt_bins,
                    dm_eff_eta_bins,
                    efficiency_type="dm",
                )
                _accumulate_dm_eff_counts(dm_eff_counts[primary_plot_label], batch_dm_eff_counts)
            if collect_tracking_efficiency and tracking_eff_counts is not None:
                batch_tracking_eff_counts = _calc_efficiency_counts(
                    batch_prediction,
                    full_data,
                    dm_eff_pt_bins,
                    dm_eff_eta_bins,
                    efficiency_type="technical",
                )
                _accumulate_dm_eff_counts(tracking_eff_counts[primary_plot_label], batch_tracking_eff_counts)
        else:
            batch_track_metrics = calc_tracking_metrics_merged(
                batch_masks,
                sub_events,
                full_data=full_data,
                full_emb=full_data.emb_all,
                full_encoder_features=full_data.encoder_features,
                full_clf_scores=full_data.clf_scores,
                postprocessing_config=postprocessing_config,
            )
        if reference_metric_names is None:
            reference_metric_names = list(batch_track_metrics.keys())

        gnn_prediction = _build_gnn_reference_prediction(full_data, gnn_reference_config)
        if gnn_prediction is not None:
            gnn_metrics = _compute_tracking_metrics(gnn_prediction, full_data)
            for metric_name in reference_metric_names:
                if metric_name in gnn_metrics:
                    reference_metric_totals[metric_name] += float(gnn_metrics[metric_name])
                    reference_metric_counts[metric_name] += 1
            if collect_dm_efficiency and dm_eff_counts is not None:
                gnn_dm_eff_counts = _calc_efficiency_counts(
                    gnn_prediction,
                    full_data,
                    dm_eff_pt_bins,
                    dm_eff_eta_bins,
                    efficiency_type="dm",
                )
                _accumulate_dm_eff_counts(dm_eff_counts[reference_plot_label], gnn_dm_eff_counts)
            if collect_tracking_efficiency and tracking_eff_counts is not None:
                gnn_tracking_eff_counts = _calc_efficiency_counts(
                    gnn_prediction,
                    full_data,
                    dm_eff_pt_bins,
                    dm_eff_eta_bins,
                    efficiency_type="technical",
                )
                _accumulate_dm_eff_counts(tracking_eff_counts[reference_plot_label], gnn_tracking_eff_counts)
            if collect_any_efficiency:
                reference_dm_eff_event_count += 1

        # Accumulate losses by name
        for loss_name, loss_value in batch_loss_dict.items():
            total_losses[loss_name].append(loss_value)
        for metric_name, metric_value in batch_track_metrics.items():
            total_track_metrics[metric_name].append(metric_value)

        # Create description with all losses
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in batch_loss_dict.items() if not k.split("_")[-1].isdigit()])
        track_str = _format_tracking_metrics_for_log(batch_track_metrics)
        desc = (
            f"[Epoch {epoch}] {phase}, {track_str}, {loss_str}"
            if track_str
            else f"[Epoch {epoch}] {phase}, {loss_str}"
        )

        if idx == len(data_loader) - 1:
            # Average all losses for final description
            avg_losses = {k: np.mean(v) for k, v in total_losses.items()}
            avg_track_metrics = {k: np.mean(v) for k, v in total_track_metrics.items()}
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items() if not k.split("_")[-1].isdigit()])
            track_str = _format_tracking_metrics_for_log(avg_track_metrics)
            desc = (
                f"[Epoch {epoch}] {phase}, {loss_str}, {track_str}"
                if track_str
                else f"[Epoch {epoch}] {phase}, {loss_str}"
            )
        if is_main_process:
            pbar.set_description(desc)

    # Return averaged losses and tracking metrics
    result = {k: np.mean(v) for k, v in total_losses.items()}
    for metric_name, metric_values in total_track_metrics.items():
        result[metric_name] = np.mean(metric_values)

    if dist.is_available() and dist.is_initialized():
        for key, value in list(result.items()):
            tensor_v = torch.tensor(float(value), device=device)
            dist.all_reduce(tensor_v, op=dist.ReduceOp.SUM)
            result[key] = tensor_v.item() / dist.get_world_size()

    if gnn_reference_config["enabled"] and reference_metric_names is not None:
        for metric_name in reference_metric_names:
            total_value = float(reference_metric_totals.get(metric_name, 0.0))
            count_value = float(reference_metric_counts.get(metric_name, 0))
            if dist.is_available() and dist.is_initialized():
                total_tensor = torch.tensor(total_value, device=device, dtype=torch.float64)
                count_tensor = torch.tensor(count_value, device=device, dtype=torch.float64)
                dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                total_value = total_tensor.item()
                count_value = count_tensor.item()
            if count_value > 0:
                result[f"{gnn_reference_config['metric_prefix']}_{metric_name}"] = total_value / count_value

    if is_main_process:
        pbar.close()

    extra = {}
    if collect_any_efficiency and dist.is_available() and dist.is_initialized():
        reference_count_tensor = torch.tensor(reference_dm_eff_event_count, device=device, dtype=torch.float64)
        dist.all_reduce(reference_count_tensor, op=dist.ReduceOp.SUM)
        reference_dm_eff_event_count = int(reference_count_tensor.item())
    if collect_dm_efficiency and dm_eff_counts is not None:
        dm_eff_counts = {label: _all_reduce_dm_eff_counts(counts, device) for label, counts in dm_eff_counts.items()}
        if is_main_process:
            plot_counts = {primary_plot_label: dm_eff_counts[primary_plot_label]}
            if gnn_reference_config["enabled"] and reference_dm_eff_event_count > 0:
                plot_counts[reference_plot_label] = dm_eff_counts[reference_plot_label]
            extra["dm_efficiency_plot"] = _build_efficiency_plot(
                plot_counts,
                dm_eff_pt_bins,
                dm_eff_eta_bins,
                y_label="DM Efficiency",
            )
    if collect_tracking_efficiency and tracking_eff_counts is not None:
        tracking_eff_counts = {
            label: _all_reduce_dm_eff_counts(counts, device) for label, counts in tracking_eff_counts.items()
        }
        if is_main_process:
            plot_counts = {primary_plot_label: tracking_eff_counts[primary_plot_label]}
            if gnn_reference_config["enabled"] and reference_dm_eff_event_count > 0:
                plot_counts[reference_plot_label] = tracking_eff_counts[reference_plot_label]
            extra["tracking_efficiency_plot"] = _build_efficiency_plot(
                plot_counts,
                dm_eff_pt_bins,
                dm_eff_eta_bins,
                y_label="Tracking Efficiency",
            )

    return result, extra


def run_one_seed(config):
    torch.set_float32_matmul_precision("high")
    config["postprocessing"] = _resolve_tracking_postprocessing_config(config.get("postprocessing"))
    config.setdefault("evaluate_gnn_text_model", False)
    config.setdefault("gnn_text_output_dir", str(DEFAULT_GNN_TEXT_OUTPUT_DIR))
    config.setdefault("gnn_text_plot_label", DEFAULT_GNN_PLOT_LABEL)
    gnn_reference_config = _resolve_gnn_reference_config(config)
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            visible_gpus = torch.cuda.device_count()
            if local_rank < 0 or local_rank >= visible_gpus:
                raise RuntimeError(
                    f"Invalid LOCAL_RANK={local_rank} for visible_gpus={visible_gpus}. "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
                )
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        world_size = 1
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

    torch.set_num_threads(config["num_threads"])
    amp_enabled, amp_dtype, scaler = setup_amp(config, device)

    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    dataset_dir = Path(config["data_dir"]) / "tracking"

    time = datetime.now().strftime("%m_%d-%H_%M_%S.%f")[:-4]
    rand_num = np.random.randint(10, 100)
    log_dir = dataset_dir / "logs" / f"{time}{rand_num}_{model_name}_{config['seed']}_{config['note']}"
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=False)
        resolved_config_path = dump_yaml_config(config, log_dir / "resolved_config.yaml")

    # Initialize wandb
    run_name = f"{model_name}-{dataset_name}-{config['seed']}-{config['note']}"
    run = (
        wandb.init(entity="g-com", project="tracking-hept", name=run_name, config=config)
        if config.get("log_wandb", True) and rank == 0
        else None
    )
    if rank == 0:
        log(
            f"Device: {device}, Model: {model_name}, Dataset: {dataset_name}, "
            f"Note: {config['note']}, World Size: {world_size}"
        )
        log(f"Log dir: {log_dir}")
        log(f"Resolved config: {resolved_config_path}")
        log(f"Tracking postprocessing: {config['postprocessing']}")
        log(
            f"GNN text evaluation: enabled={gnn_reference_config['enabled']}, "
            f"output_dir={gnn_reference_config['output_dir']}"
        )
    if run is not None:
        run.config.update(
            {
                "log_dir": str(log_dir),
                "evaluate_gnn_text_model": gnn_reference_config["enabled"],
                "gnn_text_output_dir": str(gnn_reference_config["output_dir"]),
                "gnn_text_plot_label": gnn_reference_config["plot_label"],
            }
        )

        # Log transformer.py file and tracking_trainer.py file as artifacts
        for file_path in ["./tracking_trainer.py", "./models/baselines/transformer.py"]:
            path = Path(file_path)
            if path.exists():
                artifact = wandb.Artifact(path.stem + "-code", type="code")
                artifact.add_file(str(path), name=path.name)
                run.log_artifact(artifact)
                log(f"Logged {path.name} to wandb as artifact")

    set_seed(config["seed"] + rank)
    dataset = get_dataset(dataset_name, dataset_dir)
    loader_out = get_data_loader(dataset, dataset.idx_split, batch_size=config["batch_size"], ddp=ddp)
    if ddp:
        loaders, samplers = loader_out
    else:
        loaders = loader_out
        samplers = {"train": None, "valid": None, "test": None}

    model = get_model(model_name, config["model_kwargs"], dataset)
    load_target_model = _unwrap_model(model)
    if config.get("only_flops", False):
        raise RuntimeError
    if config.get("resume", False):
        if rank == 0:
            log(f"Resume from {config['resume']}")
        checkpoint_path = dataset_dir / "logs" / (config["resume"] + "/best_checkpoint.pt")
        # For backward compatibility, also try the old model-only path
        if not checkpoint_path.exists():
            checkpoint_path = dataset_dir / "logs" / (config["resume"] + "/best_model.pt")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle both old format (state_dict only) and new format (full checkpoint)
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint
        model_state_dict_raw = model_state_dict

        # Handle 'module.' prefix from DDP checkpoints
        if any(k.startswith("module.") for k in model_state_dict.keys()):
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

        # Check if model supports region reinitialization and if num_regions differs
        if config["model_kwargs"]["reinit_regions"]:
            normalized_state_dict = _strip_checkpoint_prefixes(model_state_dict_raw)
            current_regions_shape = normalized_state_dict["regions"].shape if "regions" in normalized_state_dict else None
            new_num_regions = config["model_kwargs"].get("num_regions")
            new_n_hashes = config["model_kwargs"].get("n_hashes")
            new_num_heads = config["model_kwargs"].get("num_heads")

            # Reinitialize regions with new parameters
            load_target_model.reinit_regions(new_num_regions, new_n_hashes, new_num_heads)
            if rank == 0:
                log(f"Reinitialized regions: old shape {current_regions_shape}, new num_regions={new_num_regions}")
            # Remove regions from checkpoint to avoid size mismatch
            model_state_dict_raw = {k: v for k, v in model_state_dict_raw.items() if not k.endswith("regions")}

        if config["model_kwargs"]["reinit_queries"]:
            load_target_model.mask3d_head.reinit_queries(
                num_queries=config["model_kwargs"]["num_queries"],
                hidden_dim=load_target_model.mask3d_head.instance_queries.shape[-1],
            )
            drop_keys = {
                "mask3d_head.instance_queries",
                "mask3d_head.instance_pos_emb",
                "_orig_mod.mask3d_head.instance_queries",
                "_orig_mod.mask3d_head.instance_pos_emb",
                "module.mask3d_head.instance_queries",
                "module.mask3d_head.instance_pos_emb",
                "module._orig_mod.mask3d_head.instance_queries",
                "module._orig_mod.mask3d_head.instance_pos_emb",
            }
            model_state_dict_raw = {k: v for k, v in model_state_dict_raw.items() if k not in drop_keys}

        # Try dev-siqi style first: load directly into compiled model wrapper.
        load_result = model.load_state_dict(model_state_dict_raw, strict=False)
        matched_count, missing_count, unexpected_count, is_zero_match = _count_state_dict_matches(
            load_result,
            target_state_dict_len=len(model.state_dict()),
            source_state_dict_len=len(model_state_dict_raw),
        )
        load_method = "direct"

        # Fallback for prefix/layout mismatches: normalize keys and load unwrapped model.
        if is_zero_match:
            model_state_dict = _strip_checkpoint_prefixes(model_state_dict_raw)
            load_result = load_target_model.load_state_dict(model_state_dict, strict=False)
            matched_count, missing_count, unexpected_count, is_zero_match = _count_state_dict_matches(
                load_result,
                target_state_dict_len=len(load_target_model.state_dict()),
                source_state_dict_len=len(model_state_dict),
            )
            load_method = "normalized_unwrapped"

        if rank == 0:
            log(
                f"Loaded model checkpoint ({load_method}): "
                f"matched={matched_count}, missing={missing_count}, unexpected={unexpected_count}"
            )
            if missing_count > 0:
                log(f"Missing keys sample: {load_result.missing_keys[:5]}")
            if unexpected_count > 0:
                log(f"Unexpected keys sample: {load_result.unexpected_keys[:5]}")
        if is_zero_match:
            raise RuntimeError("Failed to load checkpoint weights: no parameter keys matched the current model.")

    model = model.to(device)
    if ddp:
        ddp_device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=ddp_device_ids, find_unused_parameters=True)

    base_model = _unwrap_model(model)
    if config["model_kwargs"]["freeze_decoder"]:
        if rank == 0:
            log("Freezing decoder layers")
        base_model.freeze_mask3d_head()
    if config["model_kwargs"]["freeze_encoder"]:
        if rank == 0:
            log("Freezing encoder layers")
        base_model.freeze_encoder_modules()

    opt = get_optimizer(model.parameters(), config["optimizer_name"], config["optimizer_kwargs"])
    optimizers = _as_list(opt)
    lr_schedulers = []
    if config["lr_scheduler_name"] is not None:
        for optimizer in optimizers:
            scheduler_cfg = dict(config["lr_scheduler_kwargs"])
            scheduler_cfg["num_training_steps"] = config["num_epochs"] * len(loaders["train"])
            lr_schedulers.append(get_lr_scheduler(optimizer, config["lr_scheduler_name"], scheduler_cfg))
    lr_s = None if len(lr_schedulers) == 0 else lr_schedulers[0] if len(lr_schedulers) == 1 else lr_schedulers
    criterion = get_loss(config["loss_name"], config["loss_kwargs"], config["model_kwargs"])

    # Load optimizer and lr_scheduler state if resuming
    start_epoch = 0
    resume_best_train = None
    resume_best_valid = None
    resume_best_test = None
    resume_best_epoch = None
    if config.get("resume", False):
        checkpoint_path = dataset_dir / "logs" / (config["resume"] + "/best_checkpoint.pt")
        if not checkpoint_path.exists():
            checkpoint_path = dataset_dir / "logs" / (config["resume"] + "/best_model.pt")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        loaded_optimizer = False
        resume_optimizer = config.get("resume_optimizer", True)
        resume_lr_scheduler = config.get("resume_lr_scheduler", True)

        # Load optimizer state if available (skip on mismatch when model shape changed)
        if "optimizer_state_dict" in checkpoint:
            if resume_optimizer:
                try:
                    optimizer_state = checkpoint["optimizer_state_dict"]
                    if isinstance(optimizer_state, list):
                        for optimizer, state in zip(optimizers, optimizer_state):
                            optimizer.load_state_dict(state)
                    else:
                        optimizers[0].load_state_dict(optimizer_state)
                    loaded_optimizer = True
                    if rank == 0:
                        log("Loaded optimizer state")
                except ValueError as exc:
                    if rank == 0:
                        log(f"Skipping optimizer state due to mismatch: {exc}")
            else:
                if rank == 0:
                    log("Skipping optimizer state (resume_optimizer=false)")

        # Preserve resumed optimizer LR by default for true checkpoint continuation.
        # Set reset_lr_from_yaml_on_resume=true to force YAML LR overrides.
        yaml_lr = config.get("optimizer_kwargs", {}).get("lr")
        reset_lr_from_yaml_on_resume = config.get("reset_lr_from_yaml_on_resume", False)
        if yaml_lr is not None and reset_lr_from_yaml_on_resume:
            adamw_lr = config.get("optimizer_kwargs", {}).get("adamw_lr", yaml_lr)
            target_lrs = [yaml_lr] if len(optimizers) == 1 else [yaml_lr, adamw_lr]
            for optimizer, target_lr in zip(optimizers, target_lrs):
                for group in optimizer.param_groups:
                    group["lr"] = target_lr
            if rank == 0:
                log(f"Reset optimizer lr from YAML (forced): {target_lrs}")
        elif rank == 0 and loaded_optimizer:
            lrs = [[group["lr"] for group in optimizer.param_groups] for optimizer in optimizers]
            log(f"Keeping resumed optimizer lr values: {lrs}")

        if resume_optimizer and config["model_kwargs"].get("reinit_queries", False):
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        optimizer.state.pop(param, None)
            loaded_optimizer = False
            if rank == 0:
                log("Cleared optimizer state after reinitializing queries")

        # Load lr_scheduler state if available (only when optimizer state is loaded)
        if "lr_scheduler_state_dict" in checkpoint and lr_s is not None:
            if resume_lr_scheduler and loaded_optimizer:
                try:
                    scheduler_state = checkpoint["lr_scheduler_state_dict"]
                    schedulers = _as_list(lr_s)
                    if isinstance(scheduler_state, list):
                        for scheduler, state in zip(schedulers, scheduler_state):
                            scheduler.load_state_dict(state)
                    else:
                        schedulers[0].load_state_dict(scheduler_state)
                    if rank == 0:
                        log("Loaded lr_scheduler state")
                except Exception as exc:
                    if rank == 0:
                        log(f"Skipping lr_scheduler state due to mismatch: {exc}")
            else:
                if rank == 0:
                    if not resume_lr_scheduler:
                        log("Skipping lr_scheduler state (resume_lr_scheduler=false)")
                    elif not loaded_optimizer:
                        log("Skipping lr_scheduler state (optimizer state not loaded)")

        # Load epoch info if available
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            if rank == 0:
                log(f"Resuming from epoch {start_epoch}")
        if config.get("resume_start_epoch") is not None:
            start_epoch = int(config["resume_start_epoch"])
            if rank == 0:
                log(f"Overriding resume start epoch from config: start_epoch={start_epoch}")

        # Load best metrics from the checkpoint for resume context only.
        # Current-run best tracking is reset below so new "best" decisions
        # use only post-resume epochs.
        if "best_train" in checkpoint:
            resume_best_train = checkpoint["best_train"]
            resume_best_valid = checkpoint["best_valid"]
            resume_best_test = checkpoint["best_test"]
            resume_best_epoch = checkpoint.get("epoch", 0)
            if rank == 0:
                log(
                    f"Loaded resume checkpoint best metrics from epoch {resume_best_epoch}: "
                    f"train={_format_tracking_metrics_for_log(resume_best_train)}, "
                    f"valid={_format_tracking_metrics_for_log(resume_best_valid)}, "
                    f"test={_format_tracking_metrics_for_log(resume_best_test)}"
                )

    # Always track "best" for the current run only, even when resuming.
    best_epoch, best_train = 0, {"dm": float("-inf")}
    best_valid, best_test = deepcopy(best_train), deepcopy(best_train)
    if rank == 0 and resume_best_train is not None:
        log("Reset current-run best tracking after resume; new best checkpoint decisions use post-resume metrics only.")

    should_log_test_dm_efficiency = bool(
        config.get("log_test_dm_efficiency_plot", True) and config.get("log_wandb", True)
    )
    should_log_test_tracking_efficiency = bool(
        config.get("log_test_tracking_efficiency_plot", True) and config.get("log_wandb", True)
    )
    dm_eff_pt_bins, dm_eff_eta_bins = (None, None)
    if should_log_test_dm_efficiency or should_log_test_tracking_efficiency:
        dm_eff_pt_bins, dm_eff_eta_bins = _get_dm_eff_bins(config)

    for epoch in range(start_epoch, config["num_epochs"]):
        # Set current epoch for noise scheduling if model supports it
        if hasattr(base_model, "set_epoch"):
            base_model.set_epoch(epoch)

        if not config.get("only_eval", False):
            train_res, _ = run_one_epoch(
                model,
                optimizers,
                criterion,
                loaders["train"],
                "train",
                epoch,
                device,
                lr_s,
                config,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
                sampler=samplers["train"],
                gnn_reference_config=gnn_reference_config,
            )
        else:
            train_res = {"dm": 0.0}
        valid_res, _ = run_one_epoch(
            model,
            optimizers,
            criterion,
            loaders["valid"],
            "valid",
            epoch,
            device,
            lr_s,
            config,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=None,
            sampler=samplers["valid"],
            gnn_reference_config=gnn_reference_config,
        )
        test_res, test_extra = run_one_epoch(
            model,
            optimizers,
            criterion,
            loaders["test"],
            "test",
            epoch,
            device,
            lr_s,
            config,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=None,
            sampler=samplers["test"],
            collect_dm_efficiency=should_log_test_dm_efficiency,
            collect_tracking_efficiency=should_log_test_tracking_efficiency,
            dm_eff_pt_bins=dm_eff_pt_bins,
            dm_eff_eta_bins=dm_eff_eta_bins,
            gnn_reference_config=gnn_reference_config,
        )

        if lr_s is not None:
            schedulers = _as_list(lr_s)
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(valid_res[config["lr_scheduler_metric"]])
                elif isinstance(scheduler, StepLR):
                    scheduler.step()

        if (valid_res["dm"]) > (best_valid["dm"]):
            best_epoch, best_train, best_valid, best_test = epoch, train_res, valid_res, test_res

            if rank == 0:
                # Save complete checkpoint including model, optimizer, and lr_scheduler states
                checkpoint = {
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": [optimizer.state_dict() for optimizer in optimizers]
                    if len(optimizers) > 1
                    else optimizers[0].state_dict(),
                    "epoch": epoch,
                    "best_train": best_train,
                    "best_valid": best_valid,
                    "best_test": best_test,
                }

                # Save lr_scheduler state if it exists
                if lr_s is not None:
                    schedulers = _as_list(lr_s)
                    checkpoint["lr_scheduler_state_dict"] = (
                        [scheduler.state_dict() for scheduler in schedulers]
                        if len(schedulers) > 1
                        else schedulers[0].state_dict()
                    )

                torch.save(checkpoint, log_dir / "best_checkpoint.pt")

                # Also save just the model state dict for backward compatibility
                torch.save(base_model.state_dict(), log_dir / "best_model.pt")

        # Save periodic checkpoint every 50 epochs or at the last epoch
        if rank == 0 and (epoch % 50 == 0 or epoch == config["num_epochs"] - 1):
            checkpoint = {
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": [optimizer.state_dict() for optimizer in optimizers]
                if len(optimizers) > 1
                else optimizers[0].state_dict(),
                "epoch": epoch,
                "best_train": best_train,
                "best_valid": best_valid,
                "best_test": best_test,
            }

            if lr_s is not None:
                schedulers = _as_list(lr_s)
                checkpoint["lr_scheduler_state_dict"] = (
                    [scheduler.state_dict() for scheduler in schedulers]
                    if len(schedulers) > 1
                    else schedulers[0].state_dict()
                )

            torch.save(checkpoint, log_dir / f"checkpoint_epoch_{epoch}.pt")

        if rank == 0:
            print(
                f"[Epoch {epoch}] Best epoch: {best_epoch}, "
                f"train: {_format_tracking_metrics_for_log(best_train)}, "
                f"valid: {_format_tracking_metrics_for_log(best_valid)}, "
                f"test: {_format_tracking_metrics_for_log(best_test)}"
            )
            print("=" * 50), print("=" * 50)

        # Log to wandb
        if run is not None and rank == 0:
            wandb_payload = {"lr": optimizers[0].param_groups[0]["lr"]}

            # Log current epoch metrics in one payload per epoch.
            for phase, res in zip(["train", "valid", "test"], [train_res, valid_res, test_res]):
                for k, v in res.items():
                    wandb_payload[f"{phase}/{k}"] = v

            # Log best metrics
            for phase, res in zip(["train", "valid", "test"], [best_train, best_valid, best_test]):
                for k, v in res.items():
                    wandb_payload[f"best_{phase}/{k}"] = v

            plot_figure = test_extra.get("dm_efficiency_plot")
            tracking_plot_figure = test_extra.get("tracking_efficiency_plot")
            if plot_figure is not None:
                wandb_payload["test/dm_efficiency"] = wandb.Image(plot_figure)
            if tracking_plot_figure is not None:
                wandb_payload["test/tracking_efficiency"] = wandb.Image(tracking_plot_figure)

            run.log(wandb_payload, step=epoch)

            if plot_figure is not None:
                plt.close(plot_figure)
            if tracking_plot_figure is not None:
                plt.close(tracking_plot_figure)

    if ddp and dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train a model for tracking.")
    parser.add_argument("-m", "--model", type=str, default="hept")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Defaults to the model-specific tracking config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-like dot-path overrides, e.g. model_kwargs.encoder_block_size=255",
    )
    args = parser.parse_args()

    config_dir = resolve_tracking_config_path(args.model, args.config)
    config = load_yaml_config(config_dir)
    config = apply_config_overrides(config, args.overrides)
    config["_config_path"] = str(config_dir)
    config["_cli_overrides"] = list(args.overrides)
    run_one_seed(config)


if __name__ == "__main__":
    main()
