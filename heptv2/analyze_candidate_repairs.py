"""Truth-only diagnostics for designing inference-safe post-processing.

This script must not be used inside inference. It reports whether missed
particles have geometry/embedding patterns that a label-free repair could
plausibly exploit.
"""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .data.dataset import TrackmlLarge, TrackingTransform
from .data.preprocess import process_data
from .eval.metrics import compute_tracking_metrics_from_prediction
from .eval.postprocess import decoder_kmeans_postprocess, trim_pred_by_valid_mask
from .model import Transformer
from .run_inference import _load_checkpoint, _patch_dynamo_nvcc_probe
from .sweep_postprocess import _pp_kwargs, _variant_presets
from .utils.block_size_utils import resolve_block_sizes
from .utils.config_utils import apply_overrides


def _cat_stats(values):
    if not values:
        return "n=0"
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"n={arr.size} mean={arr.mean():.4f} p50={np.quantile(arr, 0.5):.4f} "
        f"p90={np.quantile(arr, 0.9):.4f}"
    )


def _wrap_phi_diff(x):
    return ((x + 1.0) % 2.0) - 1.0


def _size2_geometry_residual(coords, seed_locs, cand_loc):
    c = coords
    r1, r2 = c[seed_locs[0], 2], c[seed_locs[1], 2]
    z1, z2 = c[seed_locs[0], 4], c[seed_locs[1], 4]
    p1, p2 = c[seed_locs[0], 3], c[seed_locs[1], 3]
    rc = c[cand_loc, 2]
    zc = c[cand_loc, 4]
    pc = c[cand_loc, 3]
    denom = r2 - r1
    if abs(float(denom)) < 1e-6:
        return float("inf"), float("inf")
    t = (rc - r1) / denom
    z_pred = z1 + (z2 - z1) * t
    dphi12 = _wrap_phi_diff(p2 - p1)
    dphi1c = _wrap_phi_diff(pc - p1)
    p_res = abs(_wrap_phi_diff(dphi1c - dphi12 * t))
    return abs(float(zc - z_pred)), abs(float(p_res))


def _best_partition(points, group_size):
    n = points.shape[0]
    best = None
    all_idx = set(range(n))
    for combo in combinations(range(n), group_size):
        group = set(combo)
        other = sorted(all_idx - group)
        if not other:
            continue
        g = sorted(group)
        sim = points @ points.T
        if len(g) >= 2:
            wg = sim[np.ix_(g, g)]
            wg_score = (wg.sum() - np.trace(wg)) / max(1, len(g) * (len(g) - 1))
        else:
            wg_score = 0.0
        if len(other) >= 2:
            wo = sim[np.ix_(other, other)]
            wo_score = (wo.sum() - np.trace(wo)) / max(1, len(other) * (len(other) - 1))
        else:
            wo_score = 0.0
        cross = sim[np.ix_(g, other)].mean()
        score = wg_score + wo_score - cross
        if best is None or score > best[0]:
            best = (score, tuple(g), wg_score, wo_score, cross)
    return best


@torch.no_grad()
def _run_event(model, data, cfg, device, amp_enabled, amp_dtype, pp_cfg):
    enc_cfg = cfg["model_kwargs"]
    enc_block, dec_block = resolve_block_sizes(enc_cfg)
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
    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
        enc_out = model.encode(full_data)
        full_data.emb_all = enc_out["emb_all"].detach()
        full_data.clf_scores = enc_out["clf_probs"].detach()
        for sub in sub_events:
            sub_feats = enc_out["encoded"][sub["all_idx"]]
            sub_coords = full_data.coords[sub["all_idx"]]
            pred = model.decode(sub_feats, sub_coords, sub["valid_mask"])
            pred = trim_pred_by_valid_mask(pred, sub["valid_mask"])
            raw_masks.append(
                (
                    pred["pred_masks"].detach(),
                    pred["pred_logits"].detach(),
                    pred.get("_serialized_point_order", None),
                )
            )

    predicted = decoder_kmeans_postprocess(
        raw_masks,
        sub_events,
        full_data,
        full_emb=full_data.emb_all,
        full_clf_scores=full_data.clf_scores,
        **_pp_kwargs(pp_cfg),
    )
    return predicted, full_data


def _analyze_event(predicted, full_data, cfg, accum):
    valid = full_data.valid_mask.bool()
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    pred = predicted[valid].detach().cpu().numpy()
    pid = full_data.particle_id[valid].detach().cpu().numpy()
    pt = full_data.pt[valid].detach().cpu().numpy()
    rec = full_data.reconstructable[valid].detach().cpu().numpy().astype(bool)
    layers = full_data.layer[valid].detach().cpu().numpy() if hasattr(full_data, "layer") else None
    coords = full_data.coords[valid].detach().float().cpu().numpy()
    emb = F.normalize(full_data.emb_all.float(), p=2, dim=1, eps=1e-12)
    emb_v = emb[valid_idx].detach().cpu().numpy()
    clf = full_data.clf_scores.squeeze(-1)[valid].detach().float().cpu().numpy()

    cluster_sizes = Counter(int(c) for c in pred.tolist() if int(c) >= 0)
    pids = sorted(set(int(p) for p, p_pt, p_rec in zip(pid, pt, rec) if p_pt >= cfg["eval"]["pt_thld"] and p_rec))
    for p in pids:
        hit_locs = np.flatnonzero(pid == p)
        total = int(hit_locs.size)
        if total == 0:
            continue
        counts = Counter(int(c) for c in pred[hit_locs].tolist())
        chunks = sorted(
            [(n, c, cluster_sizes.get(c, 0)) for c, n in counts.items() if c >= 0],
            reverse=True,
        )
        best_c, best_hits = (-1, 0)
        if chunks:
            best_hits, best_c, best_size = chunks[0]
        else:
            best_size = 0
        matched = (
            best_c >= 0
            and best_size >= 3
            and (best_hits / max(1, total)) > 0.5
            and (best_hits / max(1, best_size)) > 0.5
        )
        if matched:
            continue

        if chunks and chunks[0][0] == 2 and chunks[0][2] == 2:
            c = chunks[0][1]
            in_frag = hit_locs[pred[hit_locs] == c]
            missing = [loc for loc in hit_locs.tolist() if loc not in set(in_frag.tolist())]
            if missing:
                frag_cent = emb_v[in_frag].mean(axis=0)
                frag_cent = frag_cent / max(1e-12, np.linalg.norm(frag_cent))
                layer_block = set(layers[in_frag].tolist()) if layers is not None else set()
                cand = np.flatnonzero(pred < 0)
                if layers is not None and cand.size:
                    cand = cand[~np.isin(layers[cand], list(layer_block))]
                if cand.size:
                    sims = emb_v[cand] @ frag_cent
                    order = np.argsort(-sims)
                    cand_sorted = cand[order]
                    for miss in missing:
                        where = np.flatnonzero(cand_sorted == miss)
                        if where.size:
                            rank = int(where[0]) + 1
                            accum["size2_true_missing_rank"].append(rank)
                            accum["size2_true_missing_sim"].append(float(sims[order[rank - 1]]))
                            accum["size2_true_missing_clf"].append(float(clf[miss]))
                            z_res, phi_res = _size2_geometry_residual(coords, in_frag.tolist(), miss)
                            accum["size2_true_missing_z_res"].append(z_res)
                            accum["size2_true_missing_phi_res"].append(phi_res)
                        else:
                            accum["size2_missing_not_unassigned"] += 1

        top2_hits = sum(n for n, _c, _s in chunks[:2])
        if chunks and top2_hits / max(1, total) > 0.5:
            accum["split_mergeable"] += 1
            for n_pid, c, c_size in chunks[:3]:
                if n_pid < 2 or c_size < 4 or c_size > 8:
                    continue
                locs = np.flatnonzero(pred == c)
                true_locs_in_cluster = np.flatnonzero(pid[locs] == p)
                if true_locs_in_cluster.size < 2:
                    continue
                accum["split_cluster_checked"] += 1
                if layers is not None:
                    l = layers[locs]
                    accum["split_cluster_has_dup_layer"] += int(len(np.unique(l)) < len(l))
                points = emb_v[locs]
                sim = points @ points.T
                pair_scores = []
                for a, b in combinations(range(locs.size), 2):
                    pair_scores.append((float(sim[a, b]), a, b))
                pair_scores.sort(reverse=True)
                true_pair = set(true_locs_in_cluster.tolist()[:2])
                for rank_i, (_score, a, b) in enumerate(pair_scores, start=1):
                    if {a, b} == true_pair:
                        accum["true_pair_rank"].append(rank_i)
                        accum["true_pair_sim"].append(float(sim[a, b]))
                        if rank_i > 1:
                            accum["true_pair_gap_to_best"].append(float(pair_scores[0][0] - sim[a, b]))
                        else:
                            accum["true_pair_gap_to_best"].append(0.0)
                        break
                best = _best_partition(points, int(true_locs_in_cluster.size))
                if best is not None:
                    _score, group, _wg, _wo, cross = best
                    accum["partition_score"].append(float(_score))
                    accum["partition_cross"].append(float(cross))
                    accum["partition_exact"].append(int(set(group) == set(true_locs_in_cluster.tolist())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "infer.yaml"))
    parser.add_argument("--preset", type=str, default="kmeans_order")
    parser.add_argument("--variant", type=str, default="post_k1_sg010")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, args.override)

    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_enabled = bool(cfg.get("amp", {}).get("enabled", False))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg.get("amp", {}).get("dtype", "bfloat16")
    ]

    dataset = TrackmlLarge(root=Path(cfg["data_root"]), transform=TrackingTransform())
    subset = dataset[dataset.idx_split[cfg.get("split", "test")]]
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 4))

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

    variants = dict(_variant_presets(args.preset, cfg["postprocessing"]))
    pp_cfg = variants[args.variant]
    accum = Counter()
    for key in [
        "size2_true_missing_rank",
        "size2_true_missing_sim",
        "size2_true_missing_clf",
        "size2_true_missing_z_res",
        "size2_true_missing_phi_res",
        "true_pair_rank",
        "true_pair_sim",
        "true_pair_gap_to_best",
        "partition_score",
        "partition_cross",
        "partition_exact",
    ]:
        accum[key] = []
    metric_sums = []
    limit = cfg["eval"].get("limit_events")
    for i, data in enumerate(tqdm(loader, desc=f"analyze:{args.variant}")):
        if limit is not None and i >= int(limit):
            break
        predicted, full_data = _run_event(model, data, cfg, device, amp_enabled, amp_dtype, pp_cfg)
        metrics = compute_tracking_metrics_from_prediction(
            predicted,
            full_data,
            pt_thld=cfg["eval"]["pt_thld"],
            predicted_count_thld=cfg["eval"]["predicted_count_thld"],
            min_track_length=cfg["eval"]["min_track_length"],
        )
        metric_sums.append(metrics)
        _analyze_event(predicted, full_data, cfg, accum)

    dm = np.mean([m["dm"] for m in metric_sums])
    fake = np.mean([m["fake_rate"] for m in metric_sums])
    print(f"metrics dm={dm:.5f} fake={fake:.5f} n={len(metric_sums)}")
    print(f"split_mergeable={accum['split_mergeable']}")
    print(f"split_cluster_checked={accum['split_cluster_checked']}")
    if accum["split_cluster_checked"]:
        dup_frac = accum["split_cluster_has_dup_layer"] / accum["split_cluster_checked"]
        print(f"split_cluster_dup_layer_frac={dup_frac:.4f}")
    ranks = accum["true_pair_rank"]
    if ranks:
        arr = np.asarray(ranks)
        print(
            f"true_pair_rank n={arr.size} top1={(arr <= 1).mean():.4f} "
            f"top2={(arr <= 2).mean():.4f} top3={(arr <= 3).mean():.4f} "
            f"mean={arr.mean():.2f}"
        )
    print(f"true_pair_sim {_cat_stats(accum['true_pair_sim'])}")
    print(f"true_pair_gap_to_best {_cat_stats(accum['true_pair_gap_to_best'])}")
    exact = accum["partition_exact"]
    if exact:
        print(f"partition_exact n={len(exact)} frac={np.mean(exact):.4f}")
    print(f"partition_score {_cat_stats(accum['partition_score'])}")
    print(f"partition_cross {_cat_stats(accum['partition_cross'])}")
    s2_ranks = accum["size2_true_missing_rank"]
    if s2_ranks:
        arr = np.asarray(s2_ranks)
        print(
            f"size2_missing_rank n={arr.size} top1={(arr <= 1).mean():.4f} "
            f"top4={(arr <= 4).mean():.4f} top8={(arr <= 8).mean():.4f} "
            f"mean={arr.mean():.2f}"
        )
    print(f"size2_true_missing_sim {_cat_stats(accum['size2_true_missing_sim'])}")
    print(f"size2_true_missing_clf {_cat_stats(accum['size2_true_missing_clf'])}")
    print(f"size2_true_missing_z_res {_cat_stats(accum['size2_true_missing_z_res'])}")
    print(f"size2_true_missing_phi_res {_cat_stats(accum['size2_true_missing_phi_res'])}")
    print(f"size2_missing_not_unassigned={accum['size2_missing_not_unassigned']}")


if __name__ == "__main__":
    main()
