"""Diagnose where post-processing loses double-majority tracks.

This script uses truth labels only for offline analysis. It must not inform
per-hit decisions inside inference-time post-processing.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import autocast
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .data.dataset import TrackmlLarge, TrackingTransform
from .data.preprocess import process_data
from .eval.postprocess import decoder_kmeans_postprocess, trim_pred_by_valid_mask
from .model import Transformer
from .run_inference import _load_checkpoint, _patch_dynamo_nvcc_probe
from .sweep_postprocess import _pp_kwargs, _variant_presets
from .utils.block_size_utils import resolve_block_sizes
from .utils.config_utils import apply_overrides


def _pid_stats(predicted, full_data, pt_thld=0.9, min_track_length=0):
    valid = full_data.valid_mask.bool()
    pred = predicted[valid].detach().cpu().numpy()
    pid = full_data.particle_id[valid].detach().cpu().numpy()
    pt = full_data.pt[valid].detach().cpu().numpy()
    rec = full_data.reconstructable[valid].detach().cpu().numpy().astype(bool)

    pid_counts = Counter(pid.tolist())
    cluster_sizes = Counter(int(c) for c in pred.tolist() if int(c) >= 0)
    per_pid = defaultdict(Counter)
    for c, p in zip(pred, pid):
        per_pid[int(p)][int(c)] += 1

    out = Counter()
    examples = {}
    good_pids = sorted(
        {
            int(p)
            for p, p_pt, p_rec in zip(pid, pt, rec)
            if p_pt >= pt_thld and p_rec and pid_counts[int(p)] >= min_track_length
        }
    )
    for p in good_pids:
        total = pid_counts[p]
        counts = per_pid[p]
        assigned = sum(v for c, v in counts.items() if c >= 0)
        unassigned = counts.get(-1, 0)
        best_c, best_hits = max(
            ((c, n) for c, n in counts.items() if c >= 0),
            key=lambda item: item[1],
            default=(-1, 0),
        )
        best_size = cluster_sizes.get(best_c, 0) if best_c >= 0 else 0
        matched = (
            best_c >= 0
            and best_size >= 3
            and (best_hits / max(1, total)) > 0.5
            and (best_hits / max(1, best_size)) > 0.5
        )
        if matched:
            out["matched"] += 1
            continue

        chunks = sorted(
            [(n, c, cluster_sizes.get(c, 0)) for c, n in counts.items() if c >= 0],
            reverse=True,
        )
        top_hits = chunks[0][0] if chunks else 0
        top_size = chunks[0][2] if chunks else 0
        top2_hits = sum(n for n, _c, _s in chunks[:2])
        if top_hits == 2 and top_size == 2 and total == 3:
            key = "miss_size2_of_3"
        elif top_hits == 2 and top_size == 2:
            key = "miss_size2_other"
        elif chunks and top2_hits / max(1, total) > 0.5:
            key = "miss_split_mergeable"
        elif assigned == 0:
            key = "miss_all_unassigned"
        elif unassigned / max(1, total) >= 0.5:
            key = "miss_mostly_unassigned"
        elif chunks:
            key = "miss_impure_or_short"
        else:
            key = "miss_other"
        out[key] += 1
        examples.setdefault(
            key,
            {
                "pid": p,
                "total": total,
                "assigned": assigned,
                "unassigned": unassigned,
                "chunks": chunks[:4],
            },
        )
    out["total"] = len(good_pids)
    return out, examples


@torch.no_grad()
def _run_event(model, data, cfg, device, amp_enabled, amp_dtype, variants):
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

    out = {}
    for name, pp_cfg in variants:
        predicted = decoder_kmeans_postprocess(
            raw_masks,
            sub_events,
            full_data,
            full_emb=full_data.emb_all,
            full_clf_scores=full_data.clf_scores,
            **_pp_kwargs(pp_cfg),
        )
        out[name] = _pid_stats(
            predicted,
            full_data,
            pt_thld=cfg["eval"]["pt_thld"],
            min_track_length=cfg["eval"]["min_track_length"],
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "infer.yaml"))
    parser.add_argument("--preset", type=str, default="residual_tight")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, args.override)

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

    variants = _variant_presets(args.preset, cfg["postprocessing"])
    if args.variant:
        keep = set(args.variant)
        variants = [(name, pp) for name, pp in variants if name in keep]
    sums = {name: Counter() for name, _pp in variants}
    examples = {name: {} for name, _pp in variants}
    limit = cfg["eval"].get("limit_events")
    for i, data in enumerate(tqdm(loader, desc=f"diagnose:{args.preset}")):
        if limit is not None and i >= int(limit):
            break
        per_variant = _run_event(model, data, cfg, device, amp_enabled, amp_dtype, variants)
        for name, (stats, ex) in per_variant.items():
            sums[name].update(stats)
            for key, value in ex.items():
                examples[name].setdefault(key, value)

    for name, _pp in variants:
        total = max(1, sums[name]["total"])
        missed = total - sums[name]["matched"]
        print(f"\n=== {name} === total={total} matched={sums[name]['matched']} missed={missed}")
        for key, value in sums[name].most_common():
            if key in {"total", "matched"}:
                continue
            print(f"{key:24s} {value:5d}  frac_total={value / total:.5f}")
        for key, ex in sorted(examples[name].items()):
            print(f"example {key}: {ex}")


if __name__ == "__main__":
    main()
