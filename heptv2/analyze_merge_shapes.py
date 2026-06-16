"""Inspect dynamic shapes used by the fast post-processing merge path."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from torch_geometric.loader import DataLoader

from .data.dataset import TrackmlLarge, TrackingTransform
from .data.preprocess import process_data
from .eval.postprocess import (
    _build_decoder_assignments,
    _compute_clf_keep_mask,
    _l2_normalize,
    trim_pred_by_valid_mask,
)
from .model import Transformer
from .run_inference import _load_checkpoint, _patch_dynamo_nvcc_probe
from .utils.block_size_utils import resolve_block_sizes
from .utils.config_utils import apply_overrides


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="heptv2/configs/infer.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, args.override)

    torch.set_num_threads(int(cfg.get("num_threads", 1)))
    try:
        torch.set_num_interop_threads(int(cfg.get("num_interop_threads", 1)))
    except RuntimeError:
        pass
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    dataset = TrackmlLarge(root=Path(cfg["data_root"]), transform=TrackingTransform())
    subset = dataset[dataset.idx_split[cfg.get("split", "test")]]
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 0))

    model = Transformer(
        attn_type="hept",
        in_dim=dataset.x_dim,
        coords_dim=dataset.coords_dim,
        task=cfg["dataset_name"],
        **cfg["model_kwargs"],
    ).to(device)
    model.eval()
    _load_checkpoint(model, cfg["checkpoint_path"], device)
    _patch_dynamo_nvcc_probe()
    model.encode = torch.compile(model.encode, dynamic=False)
    model.decode = torch.compile(model.decode, dynamic=False)
    model = torch.compile(model, dynamic=False)

    enc_cfg = cfg["model_kwargs"]
    enc_block, dec_block = resolve_block_sizes(enc_cfg)
    pp = cfg["postprocessing"]
    limit = cfg["eval"].get("limit_events")

    rows = []
    for event_i, data in enumerate(loader):
        if limit is not None and event_i >= int(limit):
            break
        data = data.clone().to(device)
        full_data, sub_events = process_data(
            data,
            fixed_len=enc_cfg["fixed_seq_len"],
            encoder_block_size=enc_block,
            decoder_block_size=dec_block,
            regions=model.helper_funcs["regions"],
            num_sub_events=cfg["eval"].get("num_sub_events", 3),
            decoder_overlap=enc_cfg.get("decoder_overlap", True),
            eta_abs_max=cfg["eval"]["eta_abs_max"],
        )

        raw_masks = []
        with autocast(device_type="cuda", enabled=bool(cfg.get("amp", {}).get("enabled", False)), dtype=torch.bfloat16):
            enc_out = model.encode(full_data)
            full_data.emb_all = enc_out["emb_all"].detach()
            full_data.clf_scores = enc_out["clf_probs"].detach()
            for sub in sub_events:
                pred = model.decode(
                    enc_out["encoded"][sub["all_idx"]],
                    full_data.coords[sub["all_idx"]],
                    sub["valid_mask"],
                )
                if not pp.get("decoder_keep_padded", False):
                    pred = trim_pred_by_valid_mask(pred, sub["valid_mask"])
                if pp.get("decoder_output_bf16", False) and pred["pred_masks"].is_cuda:
                    pred["pred_masks"] = pred["pred_masks"].to(torch.bfloat16)
                raw_masks.append(
                    (
                        pred["pred_masks"].detach(),
                        pred["pred_logits"].detach(),
                        pred.get("_serialized_point_order", None),
                    )
                )

        clf_keep = _compute_clf_keep_mask(full_data, full_data.clf_scores, clf_recall=pp.get("clf_recall", 1.0))
        predicted = _build_decoder_assignments(
            raw_masks,
            sub_events,
            full_data,
            clf_keep,
            mask_thres=pp.get("decoder_score_threshold", 0.0),
            active_prob_threshold=pp.get("decoder_active_prob_threshold", 0.0),
            active_prob_power=pp.get("decoder_active_prob_power", 1.0),
            active_prob_floor=pp.get("decoder_active_prob_floor", 0.0),
            active_query_topk=pp.get("decoder_active_query_topk", 0),
            use_logit_scores=pp.get("decoder_use_logit_scores", False),
            score_mode=pp.get("decoder_score_mode", "prob"),
            score_bf16=pp.get("decoder_score_bf16", False),
            compile_assign=pp.get("decoder_compile_assign", False),
        )

        valid = full_data.valid_mask.bool()
        active = valid & (predicted >= 0)
        if not active.any():
            rows.append((int(valid.sum()), 0, 0, 0, 0, 0, 0.0))
            continue
        labels = predicted[active]
        active_ids, inv, counts = torch.unique(labels, return_inverse=True, return_counts=True)
        k = int(active_ids.numel())
        small_mask = counts <= int(pp.get("merge_similar_small_cluster_size", 2))
        target_min = int(pp.get("merge_similar_direct_target_min_size", 0))
        if target_min <= 0:
            target_min = int(pp.get("merge_similar_small_cluster_size", 2)) + 1
        target_mask = counts >= target_min
        max_merged = int(pp.get("merge_similar_max_merged_size", 12))
        if max_merged > 0:
            target_mask &= counts < max_merged
        small_rows = torch.nonzero(small_mask, as_tuple=False).squeeze(1)
        target_rows = torch.nonzero(target_mask, as_tuple=False).squeeze(1)

        accepted = 0
        mean_sim = 0.0
        if small_rows.numel() > 0 and target_rows.numel() > 0:
            emb = _l2_normalize(full_data.emb_all)
            active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
            sums = torch.zeros((k, emb.shape[1]), device=device, dtype=emb.dtype)
            sums.scatter_add_(0, inv[:, None].expand(-1, emb.shape[1]), emb[active_idx])
            centroids = F.normalize(
                sums / counts.to(emb.dtype).clamp_min(1).unsqueeze(1),
                p=2,
                dim=1,
                eps=1e-12,
            )
            sim = centroids[small_rows] @ centroids[target_rows].t()
            sim = sim.masked_fill(small_rows[:, None] == target_rows[None, :], float("-inf"))
            if max_merged > 0:
                too_large = counts[small_rows][:, None] + counts[target_rows][None, :] > max_merged
                sim = sim.masked_fill(too_large, float("-inf"))
            vals = sim.max(dim=1).values
            keep = vals >= float(pp.get("merge_similar_min_sim", 0.9))
            accepted = int(keep.sum())
            if accepted:
                mean_sim = float(vals[keep].mean().item())

        row = (
            int(valid.sum()),
            int(active.sum()),
            k,
            int(small_rows.numel()),
            int(target_rows.numel()),
            accepted,
            mean_sim,
        )
        rows.append(row)
        print(
            f"event={event_i} valid={row[0]} active={row[1]} labels={row[2]} "
            f"small={row[3]} target={row[4]} accepted={row[5]} sim={row[6]:.4f}",
            flush=True,
        )

    arr = np.asarray(rows, dtype=np.float64)
    print("\nSUMMARY")
    for col, name in enumerate(["valid", "active", "labels", "small", "target", "accepted", "acc_sim_mean"]):
        values = arr[:, col]
        print(
            f"{name:12s} mean={values.mean():.2f} p50={np.percentile(values, 50):.2f} "
            f"p90={np.percentile(values, 90):.2f} max={values.max():.2f}"
        )


if __name__ == "__main__":
    main()
