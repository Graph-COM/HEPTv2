"""Time incremental post-processing passes on one real event.

The model forward is run once to materialize decoder outputs. Timings below
reuse those tensors and synchronize CUDA around each repeated post-process
call, so the reported numbers isolate post-processing latency.
"""
import argparse
import statistics
import time
from pathlib import Path

import torch
import yaml
from torch.amp import autocast

from .data.dataset import TrackmlLarge, TrackingTransform
from .data.preprocess import process_data
from .eval.metrics import compute_tracking_metrics_from_prediction
from .eval.postprocess import (
    _build_decoder_assignments,
    _build_decoder_overlap_stitched_assignments,
    _compute_clf_keep_mask,
    _dedup_same_layer_clusters,
    _merge_similar_clusters,
    _pad_size2_clusters,
    _regroup_residual_hits,
    _split_overmerged_clusters,
    decoder_kmeans_postprocess,
    trim_pred_by_valid_mask,
)
from .model import Transformer
from .run_inference import _load_checkpoint, _patch_dynamo_nvcc_probe
from .utils.block_size_utils import resolve_block_sizes
from .utils.config_utils import apply_overrides


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, device, repeats, warmup):
    values = []
    for i in range(warmup + repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if i >= warmup:
            values.append(elapsed)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def _pp_base(cfg):
    pp = dict(cfg.get("postprocessing", {}) or {})
    pp.update(
        {
            "kmeans_iters": 0,
            "stitch_overlap": False,
            "pad_size2": False,
            "merge_similar": False,
            "split_overmerged": False,
            "residual_regroup": False,
            "dedup_same_layer": False,
        }
    )
    return pp


def _pp_main(cfg):
    pp = dict(_pp_base(cfg))
    src = cfg.get("postprocessing", {}) or {}
    pp.update(
        {
            "stitch_overlap": True,
            "stitch_score_gate": src.get("stitch_score_gate", 0.005),
            "stitch_min_overlap": src.get("stitch_min_overlap", 1),
            "stitch_min_overlap_frac": src.get("stitch_min_overlap_frac", 0.6),
            "merge_similar": True,
            "merge_similar_min_sim": src.get("merge_similar_min_sim", 0.75),
            "merge_similar_small_cluster_size": src.get("merge_similar_small_cluster_size", 2),
            "merge_similar_max_merged_size": src.get("merge_similar_max_merged_size", 12),
            "merge_similar_topk": src.get("merge_similar_topk", 4),
            "merge_similar_require_unique_layer": src.get("merge_similar_require_unique_layer", False),
            "pad_size2": True,
            "pad_size2_min_sim": src.get("pad_size2_min_sim", 0.3),
            "pad_size2_clf_floor": src.get("pad_size2_clf_floor", 0.0),
            "pad_size2_clf_ceiling": src.get("pad_size2_clf_ceiling", 0.0),
            "pad_size2_unique_layer": src.get("pad_size2_unique_layer", True),
            "pad_size2_topk": src.get("pad_size2_topk", 8),
        }
    )
    return pp


def _pp_residual_fast(cfg):
    pp = _pp_main(cfg)
    pp.update(
        {
            "residual_regroup": True,
            "residual_regroup_clf_threshold": 0.015,
            "residual_regroup_max_cluster_size": 2,
            "residual_regroup_topk": 16,
            "residual_regroup_seed_sim": 0.55,
            "residual_regroup_pair_sim": 0.15,
            "residual_regroup_min_group_size": 3,
            "residual_regroup_max_group_size": 4,
            "residual_regroup_require_unique_layer": True,
            "residual_regroup_clear_pool": False,
            "residual_regroup_pair_repair": False,
            "residual_regroup_max_hits": 2048,
            "pad_size2_min_sim": 0.4,
        }
    )
    return pp


def _pipeline(raw_masks, sub_events, full_data, pp):
    return decoder_kmeans_postprocess(
        raw_masks,
        sub_events,
        full_data,
        full_emb=full_data.emb_all,
        full_clf_scores=full_data.clf_scores,
        **pp,
    )


def _metrics(predicted, full_data, cfg):
    return compute_tracking_metrics_from_prediction(
        predicted,
        full_data,
        pt_thld=cfg["eval"]["pt_thld"],
        predicted_count_thld=cfg["eval"]["predicted_count_thld"],
        min_track_length=cfg["eval"]["min_track_length"],
    )


@torch.no_grad()
def _materialize_event(cfg, device, event_index):
    dataset = TrackmlLarge(root=Path(cfg["data_root"]), transform=TrackingTransform())
    split_indices = dataset.idx_split[cfg.get("split", "test")]
    data = dataset[int(split_indices[int(event_index)])].clone().to(device)

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

    enc_cfg = cfg["model_kwargs"]
    enc_block, dec_block = resolve_block_sizes(enc_cfg)
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

    amp_enabled = bool(cfg.get("amp", {}).get("enabled", False))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg.get("amp", {}).get("dtype", "bfloat16")
    ]
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
    _sync(device)
    return raw_masks, sub_events, full_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "infer.yaml"))
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
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

    raw_masks, sub_events, full_data = _materialize_event(cfg, device, args.event_index)
    valid_hits = int(full_data.valid_mask.sum().item())
    print(
        f"event_index={args.event_index} valid_hits={valid_hits} "
        f"sub_events={len(sub_events)} repeats={args.repeats} warmup={args.warmup}"
    )

    base_pp = _pp_base(cfg)
    main_pp = _pp_main(cfg)
    residual_pp = _pp_residual_fast(cfg)
    pair_pp = dict(residual_pp)
    pair_pp.update(
        {
            "residual_regroup_pair_repair": True,
            "residual_regroup_pair_repair_sim": 0.30,
            "residual_regroup_pair_repair_topk": 16,
        }
    )
    dedup_pp = dict(main_pp)
    dedup_pp.update(
        {
            "dedup_same_layer": True,
            "dedup_same_layer_min_cluster_size": 4,
            "dedup_same_layer_min_size_after": 3,
            "dedup_same_layer_min_sim_gap": 0.02,
        }
    )
    split_pp = dict(main_pp)
    split_pp.update(
        {
            "split_overmerged": True,
            "split_overmerged_min_size": 6,
            "split_overmerged_max_size": 12,
            "split_overmerged_min_part_size": 3,
            "split_overmerged_min_duplicate_layers": 1,
            "split_overmerged_max_part_duplicate_layers": 0,
            "split_overmerged_min_improve": 0.04,
            "split_overmerged_max_centroid_sim": 0.85,
            "split_overmerged_iters": 4,
            "split_overmerged_max_clusters": 64,
        }
    )

    variants = [
        ("base_9801_like", base_pp),
        ("+overlap_stitch+merge+pad", main_pp),
        ("+residual_fast_before_pad", residual_pp),
        ("+residual_pair_repair", pair_pp),
        ("+dedup_same_layer_gap02", dedup_pp),
        ("+split_overmerged_guarded", split_pp),
    ]

    print("\nPipeline totals:")
    base_mean = None
    for name, pp in variants:
        pred = _pipeline(raw_masks, sub_events, full_data, pp)
        metrics = _metrics(pred, full_data, cfg)
        mean, stdev = _time_ms(lambda pp=pp: _pipeline(raw_masks, sub_events, full_data, pp), device, args.repeats, args.warmup)
        if base_mean is None:
            base_mean = mean
        print(
            f"{name:28s} mean_ms={mean:8.2f} stdev={stdev:7.2f} "
            f"over_base={mean - base_mean:8.2f} "
            f"event_dm={metrics['dm']:.5f} event_fake={metrics['fake_rate']:.5f}"
        )

    print("\nIndividual pass timings on the same tensors:")
    clf_keep = _compute_clf_keep_mask(full_data, full_data.clf_scores, clf_recall=main_pp["clf_recall"])
    base_pred = _build_decoder_assignments(raw_masks, sub_events, full_data, clf_keep)
    stitch_pred = _build_decoder_overlap_stitched_assignments(
        raw_masks,
        sub_events,
        full_data,
        clf_keep,
        score_gate=main_pp["stitch_score_gate"],
        min_overlap=main_pp["stitch_min_overlap"],
        min_overlap_frac=main_pp["stitch_min_overlap_frac"],
    )
    merged_pred = _merge_similar_clusters(
        stitch_pred,
        full_data,
        full_data.emb_all,
        min_sim=main_pp["merge_similar_min_sim"],
        small_cluster_size=main_pp["merge_similar_small_cluster_size"],
        max_merged_size=main_pp["merge_similar_max_merged_size"],
        topk=main_pp["merge_similar_topk"],
        require_unique_layer=main_pp["merge_similar_require_unique_layer"],
    )
    residual_pred = _regroup_residual_hits(
        merged_pred,
        full_data,
        full_data.emb_all,
        full_data.clf_scores,
        clf_threshold=residual_pp["residual_regroup_clf_threshold"],
        max_cluster_size=residual_pp["residual_regroup_max_cluster_size"],
        topk=residual_pp["residual_regroup_topk"],
        seed_sim=residual_pp["residual_regroup_seed_sim"],
        pair_sim=residual_pp["residual_regroup_pair_sim"],
        min_group_size=residual_pp["residual_regroup_min_group_size"],
        max_group_size=residual_pp["residual_regroup_max_group_size"],
        require_unique_layer=residual_pp["residual_regroup_require_unique_layer"],
        clear_pool=residual_pp["residual_regroup_clear_pool"],
        pair_repair=False,
        max_hits=residual_pp["residual_regroup_max_hits"],
    )

    pass_fns = [
        (
            "clf_keep_quantile",
            lambda: _compute_clf_keep_mask(full_data, full_data.clf_scores, clf_recall=main_pp["clf_recall"]),
        ),
        (
            "base_decoder_assignment",
            lambda: _build_decoder_assignments(raw_masks, sub_events, full_data, clf_keep),
        ),
        (
            "overlap_stitched_assignment",
            lambda: _build_decoder_overlap_stitched_assignments(
                raw_masks,
                sub_events,
                full_data,
                clf_keep,
                score_gate=main_pp["stitch_score_gate"],
                min_overlap=main_pp["stitch_min_overlap"],
                min_overlap_frac=main_pp["stitch_min_overlap_frac"],
            ),
        ),
        (
            "merge_similar",
            lambda: _merge_similar_clusters(
                stitch_pred,
                full_data,
                full_data.emb_all,
                min_sim=main_pp["merge_similar_min_sim"],
                small_cluster_size=main_pp["merge_similar_small_cluster_size"],
                max_merged_size=main_pp["merge_similar_max_merged_size"],
                topk=main_pp["merge_similar_topk"],
                require_unique_layer=main_pp["merge_similar_require_unique_layer"],
            ),
        ),
        (
            "pad_size2",
            lambda: _pad_size2_clusters(
                merged_pred,
                full_data,
                full_data.emb_all,
                full_clf_scores=full_data.clf_scores,
                min_sim=main_pp["pad_size2_min_sim"],
                clf_floor=main_pp["pad_size2_clf_floor"],
                clf_ceiling=main_pp.get("pad_size2_clf_ceiling", 0.0),
                unique_layer=main_pp["pad_size2_unique_layer"],
                topk=main_pp["pad_size2_topk"],
            ),
        ),
        (
            "residual_regroup_fast",
            lambda: _regroup_residual_hits(
                merged_pred,
                full_data,
                full_data.emb_all,
                full_data.clf_scores,
                clf_threshold=residual_pp["residual_regroup_clf_threshold"],
                max_cluster_size=residual_pp["residual_regroup_max_cluster_size"],
                topk=residual_pp["residual_regroup_topk"],
                seed_sim=residual_pp["residual_regroup_seed_sim"],
                pair_sim=residual_pp["residual_regroup_pair_sim"],
                min_group_size=residual_pp["residual_regroup_min_group_size"],
                max_group_size=residual_pp["residual_regroup_max_group_size"],
                require_unique_layer=residual_pp["residual_regroup_require_unique_layer"],
                clear_pool=residual_pp["residual_regroup_clear_pool"],
                pair_repair=False,
                max_hits=residual_pp["residual_regroup_max_hits"],
            ),
        ),
        (
            "residual_pair_repair_extra",
            lambda: _regroup_residual_hits(
                merged_pred,
                full_data,
                full_data.emb_all,
                full_data.clf_scores,
                clf_threshold=residual_pp["residual_regroup_clf_threshold"],
                max_cluster_size=residual_pp["residual_regroup_max_cluster_size"],
                topk=residual_pp["residual_regroup_topk"],
                seed_sim=residual_pp["residual_regroup_seed_sim"],
                pair_sim=residual_pp["residual_regroup_pair_sim"],
                min_group_size=residual_pp["residual_regroup_min_group_size"],
                max_group_size=residual_pp["residual_regroup_max_group_size"],
                require_unique_layer=residual_pp["residual_regroup_require_unique_layer"],
                clear_pool=residual_pp["residual_regroup_clear_pool"],
                pair_repair=True,
                pair_repair_sim=0.30,
                pair_repair_topk=16,
                max_hits=residual_pp["residual_regroup_max_hits"],
            ),
        ),
        (
            "pad_after_residual_fast",
            lambda: _pad_size2_clusters(
                residual_pred,
                full_data,
                full_data.emb_all,
                full_clf_scores=full_data.clf_scores,
                min_sim=residual_pp["pad_size2_min_sim"],
                clf_floor=residual_pp["pad_size2_clf_floor"],
                clf_ceiling=residual_pp.get("pad_size2_clf_ceiling", 0.0),
                unique_layer=residual_pp["pad_size2_unique_layer"],
                topk=residual_pp["pad_size2_topk"],
            ),
        ),
        (
            "dedup_same_layer_gap02",
            lambda: _dedup_same_layer_clusters(
                merged_pred,
                full_data,
                full_data.emb_all,
                min_cluster_size=4,
                min_size_after=3,
                min_sim_gap=0.02,
            ),
        ),
        (
            "split_overmerged_guarded",
            lambda: _split_overmerged_clusters(
                merged_pred,
                full_data,
                full_data.emb_all,
                min_size=6,
                max_size=12,
                min_part_size=3,
                min_duplicate_layers=1,
                max_part_duplicate_layers=0,
                min_improve=0.04,
                max_centroid_sim=0.85,
                iters=4,
                max_clusters=64,
            ),
        ),
    ]
    for name, fn in pass_fns:
        mean, stdev = _time_ms(fn, device, args.repeats, args.warmup)
        print(f"{name:28s} mean_ms={mean:8.2f} stdev={stdev:7.2f}")


if __name__ == "__main__":
    main()
