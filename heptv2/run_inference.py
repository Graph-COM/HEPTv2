"""Entry point for heptv2 inference.

Usage:
    python -m heptv2.run_inference --config heptv2/config.yaml

Loads the checkpoint trained by the full `src/tracking_trainer.py` pipeline,
runs encoder → per-sub-event decoder → decoder_kmeans post-processing →
tracking metrics over the test split, and prints aggregated metrics.
"""
import argparse
import functools
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


def _patch_dynamo_nvcc_probe():
    """Guard torch._dynamo's nvcc --version probe (called from save_graph_repro
    on every compile). Patches both debug_utils and after_aot namespaces
    since after_aot rebinds the name at import time."""
    try:
        from torch._dynamo import debug_utils as dynamo_debug_utils
    except Exception:
        return
    original = getattr(dynamo_debug_utils, "_cuda_system_info_comment", None)
    if original is None:
        return
    if getattr(original, "__hept_nvcc_safe__", False):
        safe = original
    else:
        @functools.cache
        def _safe():
            try:
                return original()
            except PermissionError as exc:
                return f"# nvcc is not executable in this environment ({exc})\n"
        _safe.__hept_nvcc_safe__ = True
        safe = _safe
    dynamo_debug_utils._cuda_system_info_comment = safe
    # after_aot imports the name at module load; force-import and rebind.
    try:
        from torch._dynamo.repro import after_aot as dynamo_after_aot
        dynamo_after_aot._cuda_system_info_comment = safe
    except Exception:
        pass
from torch.amp import autocast
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .utils.block_size_utils import resolve_block_sizes
from .utils.config_utils import apply_overrides
from .data.dataset import TrackingTransform, TrackmlLarge
from .data.preprocess import process_data, process_inference_data
from .eval.compact_postprocess import postprocess_compact_assignment
from .eval.metrics import compute_tracking_metrics_from_prediction
from .eval.postprocess import decoder_kmeans_postprocess, trim_pred_by_valid_mask
from .model import Transformer


def _mark_cudagraph_step():
    if os.environ.get("HEPTV2_CODEX_MARK_CUDAGRAPH_STEP", "0") != "1":
        return
    marker = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def _compile_kwargs_from_mode(mode: str):
    kwargs = {"dynamic": False}
    if mode and mode != "default":
        kwargs["mode"] = mode
    return kwargs


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _strip_checkpoint_prefixes(state_dict):
    """Strip `module.` (DDP) and `_orig_mod.` (torch.compile) prefixes."""
    out = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "_orig_mod."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


def _load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = _strip_checkpoint_prefixes(state_dict)
    # `_orig_mod.` may also appear nested (e.g. after torch.compile wrapped
    # submodules). Strip any remaining prefix that survives after the first pass.
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:10]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:10]}")
    return model


def _apply_torchao_quant(model):
    mode = os.environ.get("HEPTV2_TORCHAO_QUANT", "").strip().lower()
    if not mode:
        return model
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8WeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        quantize_,
    )

    configs = {
        "int8wo": Int8WeightOnlyConfig,
        "int8dynamic": Int8DynamicActivationInt8WeightConfig,
        "fp8wo": Float8WeightOnlyConfig,
        "fp8dynamic": Float8DynamicActivationFloat8WeightConfig,
        "int4wo": Int4WeightOnlyConfig,
    }
    if mode not in configs:
        raise ValueError(f"Unknown HEPTV2_TORCHAO_QUANT={mode!r}; expected one of {sorted(configs)}")
    includes = [p.strip() for p in os.environ.get("HEPTV2_TORCHAO_INCLUDE", "").split(",") if p.strip()]
    excludes = [p.strip() for p in os.environ.get("HEPTV2_TORCHAO_EXCLUDE", "").split(",") if p.strip()]

    def filter_fn(mod, fqn):
        if not isinstance(mod, nn.Linear):
            return False
        if includes and not any(p in fqn for p in includes):
            return False
        if excludes and any(p in fqn for p in excludes):
            return False
        return True

    quantize_(model, configs[mode](), filter_fn=filter_fn)
    print(
        f"[torchao] quant={mode} include={includes or ['*']} "
        f"exclude={excludes or []}"
    )
    return model


def _apply_fp16_include(model):
    includes = [p.strip() for p in os.environ.get("HEPTV2_MODEL_FP16_INCLUDE", "").split(",") if p.strip()]
    if not includes:
        return model
    for name, module in model.named_modules():
        if name and any(p in name for p in includes):
            module.to(torch.float16)
    print(f"[model] fp16 include={includes}")
    return model


def _needs_encoder_clf(cfg):
    pp = cfg.get("postprocessing", {})
    if float(pp.get("clf_recall", 0.995)) < 1.0:
        return True
    return any(
        bool(pp.get(key, False))
        for key in (
            "codex_fused_nomask_target",
            "pad_size2",
            "pad_window_size2",
            "residual_regroup",
        )
    )


def _get_inference_path_cfg(cfg):
    return cfg.get("inference_path", {})


def _compact_assignment_enabled(cfg):
    return bool(_get_inference_path_cfg(cfg).get("compact_assignment", False))


def _attn_mask_enabled(cfg, model=None):
    head = getattr(model, "mask3d_head", None)
    if head is not None and hasattr(head, "use_attn_mask"):
        return bool(head.use_attn_mask)
    return bool(cfg.get("model_kwargs", {}).get("use_attn_mask", False))


def _compact_assignment_active(cfg, model=None):
    return _compact_assignment_enabled(cfg) and not _attn_mask_enabled(cfg, model)


def _parse_pad_bucket_lengths(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [int(part) for part in value.split(",") if part.strip()]
    return [int(part) for part in value]


@torch.inference_mode()
def predict_compact_event(model, data, cfg, device, amp_enabled, amp_dtype, build_targets=True):
    enc_cfg = cfg["model_kwargs"]
    enc_block, dec_block = resolve_block_sizes(enc_cfg)
    path_cfg = _get_inference_path_cfg(cfg)
    data = data.clone().to(device)

    full_data, _ = process_inference_data(
        data,
        fixed_len=enc_cfg["fixed_seq_len"],
        encoder_block_size=enc_block,
        decoder_block_size=dec_block,
        regions=model.helper_funcs["regions"],
        num_sub_events=cfg["eval"].get("num_sub_events", 3),
        decoder_overlap=enc_cfg.get("decoder_overlap", True),
        eta_abs_max=cfg["eval"]["eta_abs_max"],
        return_sub_events=False,
        dynamic_pad=bool(path_cfg.get("dynamic_pad", False)),
        pad_bucket_lengths=_parse_pad_bucket_lengths(path_cfg.get("pad_buckets")),
        match_fixed_region_bins=bool(path_cfg.get("match_fixed_region_bins", False)),
        build_targets=build_targets,
    )

    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
        need_clf = _needs_encoder_clf(cfg)
        if hasattr(model, "encode_online"):
            _mark_cudagraph_step()
            enc_out = model.encode_online(full_data, need_clf=need_clf, need_emb=True)
        else:
            _mark_cudagraph_step()
            enc_out = model.encode(full_data)
        full_data.emb_all = enc_out["emb_all"].detach()
        if cfg["postprocessing"].get("embedding_output_bf16", False) and full_data.emb_all.is_cuda:
            full_data.emb_all = full_data.emb_all.to(torch.bfloat16)
        clf_probs = enc_out.get("clf_probs")
        full_data.clf_scores = None if clf_probs is None else clf_probs.detach()

        all_idx = full_data.decoder_sub_all_idx
        valid_masks = full_data.decoder_sub_valid_mask
        point_coords = (
            full_data.coords[all_idx]
            if getattr(model.mask3d_head, "needs_point_coords", True)
            else None
        )
        _mark_cudagraph_step()
        compact_pred = model.decode_assignment_inputs(
            enc_out["encoded"][all_idx],
            point_coords,
            valid_masks,
        )

    _mark_cudagraph_step()
    predicted = postprocess_compact_assignment(compact_pred, full_data, cfg)
    return predicted.clone(), full_data


@torch.inference_mode()
def run_event(model, data, cfg, device, amp_enabled, amp_dtype):
    if _compact_assignment_active(cfg, model):
        predicted, full_data = predict_compact_event(
            model,
            data,
            cfg,
            device,
            amp_enabled,
            amp_dtype,
            build_targets=True,
        )
        return compute_tracking_metrics_from_prediction(
            predicted,
            full_data,
            pt_thld=cfg["eval"]["pt_thld"],
            predicted_count_thld=cfg["eval"]["predicted_count_thld"],
            min_track_length=cfg["eval"]["min_track_length"],
        )

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
        need_clf = _needs_encoder_clf(cfg)
        if hasattr(model, "encode_online"):
            _mark_cudagraph_step()
            enc_out = model.encode_online(full_data, need_clf=need_clf, need_emb=True)
        else:
            _mark_cudagraph_step()
            enc_out = model.encode(full_data)
        full_data.emb_all = enc_out["emb_all"].detach()
        if cfg["postprocessing"].get("embedding_output_bf16", False) and full_data.emb_all.is_cuda:
            full_data.emb_all = full_data.emb_all.to(torch.bfloat16)
        clf_probs = enc_out.get("clf_probs")
        full_data.clf_scores = None if clf_probs is None else clf_probs.detach()
        if cfg["postprocessing"].get("decoder_batched_decode", False):
            all_idx = getattr(full_data, "decoder_sub_all_idx", None)
            valid_masks = getattr(full_data, "decoder_sub_valid_mask", None)
            if all_idx is None or valid_masks is None:
                all_idx = torch.stack([sub["all_idx"] for sub in sub_events], dim=0)
                valid_masks = torch.stack([sub["valid_mask"] for sub in sub_events], dim=0)
            point_coords = (
                full_data.coords[all_idx]
                if getattr(model.mask3d_head, "needs_point_coords", True)
                else None
            )
            _mark_cudagraph_step()
            pred = model.decode(
                enc_out["encoded"][all_idx],
                point_coords,
                valid_masks,
            )
            if cfg["postprocessing"].get("decoder_materialize_output", False):
                pred["pred_masks"] = pred["pred_masks"].clone(memory_format=torch.contiguous_format)
                pred["pred_logits"] = pred["pred_logits"].clone(memory_format=torch.contiguous_format)
            if cfg["postprocessing"].get("decoder_output_bf16", False) and pred["pred_masks"].is_cuda:
                pred["pred_masks"] = pred["pred_masks"].to(torch.bfloat16)
            raw_masks.append(
                (
                    pred["pred_masks"].detach(),
                    pred["pred_logits"].detach(),
                    pred.get("_serialized_point_order", None),
                )
            )
        else:
            for sub in sub_events:
                sub_feats = enc_out["encoded"][sub["all_idx"]]
                sub_coords = (
                    full_data.coords[sub["all_idx"]]
                    if getattr(model.mask3d_head, "needs_point_coords", True)
                    else None
                )
                _mark_cudagraph_step()
                pred = model.decode(sub_feats, sub_coords, sub["valid_mask"])
                if not cfg["postprocessing"].get("decoder_keep_padded", False):
                    pred = trim_pred_by_valid_mask(pred, sub["valid_mask"])
                if cfg["postprocessing"].get("decoder_output_bf16", False) and pred["pred_masks"].is_cuda:
                    pred["pred_masks"] = pred["pred_masks"].to(torch.bfloat16)
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
        full_clf_scores=getattr(full_data, "clf_scores", None),
        pre_kmeans_iters=cfg["postprocessing"].get("pre_kmeans_iters", 0),
        pre_kmeans_min_margin=cfg["postprocessing"].get("pre_kmeans_min_margin", 0.0),
        kmeans_iters=cfg["postprocessing"].get("kmeans_iters", 0),
        kmeans_min_margin=cfg["postprocessing"].get("kmeans_min_margin", 0.0),
        kmeans_use_fp16_sim=cfg["postprocessing"].get("kmeans_use_fp16_sim", False),
        kmeans_max_refine_cluster_size=cfg["postprocessing"].get("kmeans_max_refine_cluster_size", 0),
        kmeans_refine_max_score=cfg["postprocessing"].get("kmeans_refine_max_score", 0.0),
        clf_recall=cfg["postprocessing"].get("clf_recall", 0.995),
        decoder_score_threshold=cfg["postprocessing"].get("decoder_score_threshold", 0.0),
        decoder_active_prob_threshold=cfg["postprocessing"].get("decoder_active_prob_threshold", 0.0),
        decoder_active_prob_power=cfg["postprocessing"].get("decoder_active_prob_power", 1.0),
        decoder_active_prob_floor=cfg["postprocessing"].get("decoder_active_prob_floor", 0.0),
        decoder_active_query_topk=cfg["postprocessing"].get("decoder_active_query_topk", 0),
        decoder_use_logit_scores=cfg["postprocessing"].get("decoder_use_logit_scores", False),
        decoder_score_mode=cfg["postprocessing"].get("decoder_score_mode", "prob"),
        decoder_score_bf16=cfg["postprocessing"].get("decoder_score_bf16", False),
        decoder_compile_assign=cfg["postprocessing"].get("decoder_compile_assign", False),
        decoder_batch_assignments=cfg["postprocessing"].get("decoder_batch_assignments", False),
        decoder_primary_sub_events=cfg["postprocessing"].get("decoder_primary_sub_events", 0),
        embeddings_already_normalized=cfg["postprocessing"].get("embeddings_already_normalized", False),
        cluster_score_filter=cfg["postprocessing"].get("cluster_score_filter", False),
        cluster_score_filter_max_size=cfg["postprocessing"].get("cluster_score_filter_max_size", 3),
        cluster_score_filter_mean_threshold=cfg["postprocessing"].get("cluster_score_filter_mean_threshold", 0.0),
        cluster_score_filter_min_threshold=cfg["postprocessing"].get("cluster_score_filter_min_threshold", 0.0),
        stitch_overlap=cfg["postprocessing"].get("stitch_overlap", False),
        stitch_score_gate=cfg["postprocessing"].get("stitch_score_gate", 0.02),
        stitch_min_overlap=cfg["postprocessing"].get("stitch_min_overlap", 1),
        stitch_min_overlap_frac=cfg["postprocessing"].get("stitch_min_overlap_frac", 0.6),
        stitch_query_topk=cfg["postprocessing"].get("stitch_query_topk", 1),
        stitch_union_topk=cfg["postprocessing"].get("stitch_union_topk", True),
        stitch_topk_score_gate=cfg["postprocessing"].get("stitch_topk_score_gate", None),
        stitch_topk_assign=cfg["postprocessing"].get("stitch_topk_assign", False),
        stitch_topk_assign_score_gate=cfg["postprocessing"].get("stitch_topk_assign_score_gate", 0.0),
        stitch_source_matrix=cfg["postprocessing"].get("stitch_source_matrix", False),
        extra_window_bridge=cfg["postprocessing"].get("extra_window_bridge", False),
        extra_window_bridge_max_cluster_size=cfg["postprocessing"].get("extra_window_bridge_max_cluster_size", 2),
        extra_window_bridge_max_merged_size=cfg["postprocessing"].get("extra_window_bridge_max_merged_size", 6),
        extra_window_bridge_min_score=cfg["postprocessing"].get("extra_window_bridge_min_score", 0.003),
        extra_window_bridge_min_label_hits=cfg["postprocessing"].get("extra_window_bridge_min_label_hits", 1),
        extra_window_bridge_max_hits=cfg["postprocessing"].get("extra_window_bridge_max_hits", 0),
        extra_window_bridge_query_topk=cfg["postprocessing"].get("extra_window_bridge_query_topk", 0),
        restore_topk_components=cfg["postprocessing"].get("restore_topk_components", False),
        restore_topk_min_score=cfg["postprocessing"].get("restore_topk_min_score", 0.01),
        restore_topk_max_conflict_size=cfg["postprocessing"].get("restore_topk_max_conflict_size", 0),
        restore_topk_min_target_size=cfg["postprocessing"].get("restore_topk_min_target_size", 3),
        restore_topk_unique_layer=cfg["postprocessing"].get("restore_topk_unique_layer", True),
        restore_window_clusters=cfg["postprocessing"].get("restore_window_clusters", False),
        restore_window_min_size=cfg["postprocessing"].get("restore_window_min_size", 3),
        restore_window_min_new_hits=cfg["postprocessing"].get("restore_window_min_new_hits", 3),
        restore_window_min_score=cfg["postprocessing"].get("restore_window_min_score", 0.0),
        restore_window_max_conflict_size=cfg["postprocessing"].get("restore_window_max_conflict_size", 2),
        restore_window_min_new_frac=cfg["postprocessing"].get("restore_window_min_new_frac", 0.6),
        bridge_window_clusters=cfg["postprocessing"].get("bridge_window_clusters", False),
        bridge_window_min_score=cfg["postprocessing"].get("bridge_window_min_score", 0.005),
        bridge_window_min_candidate_size=cfg["postprocessing"].get("bridge_window_min_candidate_size", 3),
        bridge_window_min_label_overlap=cfg["postprocessing"].get("bridge_window_min_label_overlap", 1),
        bridge_window_max_cluster_size=cfg["postprocessing"].get("bridge_window_max_cluster_size", 3),
        bridge_window_max_merged_size=cfg["postprocessing"].get("bridge_window_max_merged_size", 8),
        pad_window_size2=cfg["postprocessing"].get("pad_window_size2", False),
        pad_window_min_score=cfg["postprocessing"].get("pad_window_min_score", 0.005),
        pad_window_extra_min_score=cfg["postprocessing"].get("pad_window_extra_min_score", 0.005),
        pad_window_min_overlap=cfg["postprocessing"].get("pad_window_min_overlap", 2),
        pad_window_unique_layer=cfg["postprocessing"].get("pad_window_unique_layer", True),
        pad_window_max_proposals_per_label=cfg["postprocessing"].get("pad_window_max_proposals_per_label", 0),
        split_window_clusters=cfg["postprocessing"].get("split_window_clusters", False),
        split_window_min_score=cfg["postprocessing"].get("split_window_min_score", 0.005),
        split_window_min_cluster_size=cfg["postprocessing"].get("split_window_min_cluster_size", 5),
        split_window_max_cluster_size=cfg["postprocessing"].get("split_window_max_cluster_size", 12),
        split_window_min_part_size=cfg["postprocessing"].get("split_window_min_part_size", 3),
        split_window_min_remaining_size=cfg["postprocessing"].get("split_window_min_remaining_size", 3),
        split_window_min_inside_frac=cfg["postprocessing"].get("split_window_min_inside_frac", 0.6),
        split_window_max_part_frac=cfg["postprocessing"].get("split_window_max_part_frac", 0.7),
        split_window_min_duplicate_layers=cfg["postprocessing"].get("split_window_min_duplicate_layers", 0),
        split_window_max_part_duplicate_layers=cfg["postprocessing"].get("split_window_max_part_duplicate_layers", -1),
        split_window_require_remainder_candidate=cfg["postprocessing"].get("split_window_require_remainder_candidate", False),
        split_window_min_remainder_candidate_hits=cfg["postprocessing"].get("split_window_min_remainder_candidate_hits", 3),
        split_window_max_splits=cfg["postprocessing"].get("split_window_max_splits", 128),
        pad_size2=cfg["postprocessing"].get("pad_size2", False),
        pad_size2_after_kmeans=cfg["postprocessing"].get("pad_size2_after_kmeans", False),
        pad_size2_min_sim=cfg["postprocessing"].get("pad_size2_min_sim", 0.0),
        pad_size2_clf_floor=cfg["postprocessing"].get("pad_size2_clf_floor", 0.0),
        pad_size2_clf_ceiling=cfg["postprocessing"].get("pad_size2_clf_ceiling", 0.0),
        pad_size2_unique_layer=cfg["postprocessing"].get("pad_size2_unique_layer", True),
        pad_size2_topk=cfg["postprocessing"].get("pad_size2_topk", 8),
        pad_size2_max_candidates=cfg["postprocessing"].get("pad_size2_max_candidates", 0),
        pad_size2_max_clusters=cfg["postprocessing"].get("pad_size2_max_clusters", 0),
        pad_size2_max_conflict_size=cfg["postprocessing"].get("pad_size2_max_conflict_size", 0),
        pad_size2_geom_z_max=cfg["postprocessing"].get("pad_size2_geom_z_max", 0.0),
        pad_size2_geom_phi_max=cfg["postprocessing"].get("pad_size2_geom_phi_max", 0.0),
        pad_size2_rank_clusters_by_clf=cfg["postprocessing"].get("pad_size2_rank_clusters_by_clf", True),
        pad_size2_rank_candidates_by_clf=cfg["postprocessing"].get("pad_size2_rank_candidates_by_clf", True),
        merge_similar=cfg["postprocessing"].get("merge_similar", False),
        merge_similar_direct=cfg["postprocessing"].get("merge_similar_direct", False),
        merge_similar_label_space=cfg["postprocessing"].get("merge_similar_label_space", False),
        merge_similar_fuse_assign_merge=cfg["postprocessing"].get("merge_similar_fuse_assign_merge", False),
        merge_similar_min_sim=cfg["postprocessing"].get("merge_similar_min_sim", 0.9),
        merge_similar_small_cluster_size=cfg["postprocessing"].get("merge_similar_small_cluster_size", 2),
        merge_similar_max_merged_size=cfg["postprocessing"].get("merge_similar_max_merged_size", 12),
        merge_similar_topk=cfg["postprocessing"].get("merge_similar_topk", 4),
        merge_similar_require_unique_layer=cfg["postprocessing"].get("merge_similar_require_unique_layer", False),
        merge_similar_direct_target_min_size=cfg["postprocessing"].get("merge_similar_direct_target_min_size", 0),
        merge_geometric_small=cfg["postprocessing"].get("merge_geometric_small", False),
        merge_geometric_small_min_size=cfg["postprocessing"].get("merge_geometric_small_min_size", 2),
        merge_geometric_small_max_size=cfg["postprocessing"].get("merge_geometric_small_max_size", 4),
        merge_geometric_small_z_max=cfg["postprocessing"].get("merge_geometric_small_z_max", 0.003),
        merge_geometric_small_phi_max=cfg["postprocessing"].get("merge_geometric_small_phi_max", 0.03),
        merge_geometric_small_target_min_size=cfg["postprocessing"].get("merge_geometric_small_target_min_size", 1),
        merge_geometric_small_require_unique_layer=cfg["postprocessing"].get("merge_geometric_small_require_unique_layer", True),
        merge_geometric_endpoint=cfg["postprocessing"].get("merge_geometric_endpoint", False),
        merge_geometric_endpoint_min_size=cfg["postprocessing"].get("merge_geometric_endpoint_min_size", 2),
        merge_geometric_endpoint_max_size=cfg["postprocessing"].get("merge_geometric_endpoint_max_size", 7),
        merge_geometric_endpoint_z_max=cfg["postprocessing"].get("merge_geometric_endpoint_z_max", 0.002),
        merge_geometric_endpoint_phi_max=cfg["postprocessing"].get("merge_geometric_endpoint_phi_max", 0.02),
        merge_geometric_endpoint_r_overlap_max=cfg["postprocessing"].get("merge_geometric_endpoint_r_overlap_max", 0.01),
        merge_geometric_endpoint_target_min_size=cfg["postprocessing"].get("merge_geometric_endpoint_target_min_size", 1),
        merge_geometric_endpoint_max_layer_overlap=cfg["postprocessing"].get("merge_geometric_endpoint_max_layer_overlap", 0),
        merge_same_query=cfg["postprocessing"].get("merge_same_query", False),
        split_overmerged=cfg["postprocessing"].get("split_overmerged", False),
        split_overmerged_min_size=cfg["postprocessing"].get("split_overmerged_min_size", 6),
        split_overmerged_max_size=cfg["postprocessing"].get("split_overmerged_max_size", 12),
        split_overmerged_min_part_size=cfg["postprocessing"].get("split_overmerged_min_part_size", 3),
        split_overmerged_min_duplicate_layers=cfg["postprocessing"].get("split_overmerged_min_duplicate_layers", 1),
        split_overmerged_max_part_duplicate_layers=cfg["postprocessing"].get("split_overmerged_max_part_duplicate_layers", 0),
        split_overmerged_min_improve=cfg["postprocessing"].get("split_overmerged_min_improve", 0.04),
        split_overmerged_max_centroid_sim=cfg["postprocessing"].get("split_overmerged_max_centroid_sim", 0.85),
        split_overmerged_iters=cfg["postprocessing"].get("split_overmerged_iters", 4),
        split_overmerged_max_clusters=cfg["postprocessing"].get("split_overmerged_max_clusters", 128),
        residual_regroup=cfg["postprocessing"].get("residual_regroup", False),
        residual_regroup_clf_threshold=cfg["postprocessing"].get("residual_regroup_clf_threshold", 0.015),
        residual_regroup_max_cluster_size=cfg["postprocessing"].get("residual_regroup_max_cluster_size", 3),
        residual_regroup_topk=cfg["postprocessing"].get("residual_regroup_topk", 32),
        residual_regroup_seed_sim=cfg["postprocessing"].get("residual_regroup_seed_sim", 0.5),
        residual_regroup_pair_sim=cfg["postprocessing"].get("residual_regroup_pair_sim", 0.1),
        residual_regroup_min_group_size=cfg["postprocessing"].get("residual_regroup_min_group_size", 3),
        residual_regroup_max_group_size=cfg["postprocessing"].get("residual_regroup_max_group_size", 4),
        residual_regroup_require_unique_layer=cfg["postprocessing"].get("residual_regroup_require_unique_layer", True),
        residual_regroup_clear_pool=cfg["postprocessing"].get("residual_regroup_clear_pool", True),
        residual_regroup_pair_repair=cfg["postprocessing"].get("residual_regroup_pair_repair", False),
        residual_regroup_pair_repair_sim=cfg["postprocessing"].get("residual_regroup_pair_repair_sim", 0.3),
        residual_regroup_pair_repair_topk=cfg["postprocessing"].get("residual_regroup_pair_repair_topk", 16),
        residual_regroup_max_hits=cfg["postprocessing"].get("residual_regroup_max_hits", 0),
        dedup_same_layer=cfg["postprocessing"].get("dedup_same_layer", False),
        dedup_same_layer_min_cluster_size=cfg["postprocessing"].get("dedup_same_layer_min_cluster_size", 4),
        dedup_same_layer_min_size_after=cfg["postprocessing"].get("dedup_same_layer_min_size_after", 3),
        dedup_same_layer_min_sim_gap=cfg["postprocessing"].get("dedup_same_layer_min_sim_gap", 0.0),
        dedup_same_layer_by_score=cfg["postprocessing"].get("dedup_same_layer_by_score", False),
        dedup_same_layer_score_min_cluster_size=cfg["postprocessing"].get("dedup_same_layer_score_min_cluster_size", 4),
        dedup_same_layer_score_min_size_after=cfg["postprocessing"].get("dedup_same_layer_score_min_size_after", 3),
        dedup_same_layer_score_min_gap=cfg["postprocessing"].get("dedup_same_layer_score_min_gap", 0.0),
        prune_small_clusters=cfg["postprocessing"].get("prune_small_clusters", False),
        prune_small_cluster_max_size=cfg["postprocessing"].get("prune_small_cluster_max_size", 5),
        prune_small_cluster_min_size_after=cfg["postprocessing"].get("prune_small_cluster_min_size_after", 3),
        prune_small_cluster_max_drop=cfg["postprocessing"].get("prune_small_cluster_max_drop", 1),
        prune_small_cluster_min_sim_gap=cfg["postprocessing"].get("prune_small_cluster_min_sim_gap", 0.0),
        prune_small_cluster_min_score_gap=cfg["postprocessing"].get("prune_small_cluster_min_score_gap", 0.0),
        prune_small_cluster_require_duplicate_layer=cfg["postprocessing"].get("prune_small_cluster_require_duplicate_layer", False),
        prune_small_cluster_duplicate_layer_only=cfg["postprocessing"].get("prune_small_cluster_duplicate_layer_only", False),
    )
    return compute_tracking_metrics_from_prediction(
        predicted,
        full_data,
        pt_thld=cfg["eval"]["pt_thld"],
        predicted_count_thld=cfg["eval"]["predicted_count_thld"],
        min_track_length=cfg["eval"]["min_track_length"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "infer.yaml"))
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)

    recompile_limit = int(os.environ.get("HEPTV2_DYNAMO_RECOMPILE_LIMIT", "0") or 0)
    if recompile_limit > 0:
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = recompile_limit
        dynamo_config.accumulated_recompile_limit = max(
            int(getattr(dynamo_config, "accumulated_recompile_limit", recompile_limit)),
            recompile_limit * 16,
        )
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(int(cfg.get("num_threads", 1)))
    try:
        torch.set_num_interop_threads(int(cfg.get("num_interop_threads", 1)))
    except RuntimeError:
        pass
    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)

    amp_enabled = bool(cfg.get("amp", {}).get("enabled", False))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg.get("amp", {}).get("dtype", "bfloat16")
    ]

    data_root = Path(cfg["data_root"])
    dataset = TrackmlLarge(root=data_root, transform=TrackingTransform())
    indices = dataset.idx_split[cfg.get("split", "test")]
    # Optional: restrict to a specific evtid range (useful when the dataset has
    # been extended past what the checkpoint was trained on).
    evtid_range = cfg["eval"].get("evtid_range")
    if evtid_range is not None:
        lo, hi = int(evtid_range[0]), int(evtid_range[1])
        evtids = dataset.data.evtid[indices]
        keep = (evtids >= lo) & (evtids <= hi)
        indices = indices[keep]
        print(f"[split] filtered to evtid in [{lo},{hi}] -> {len(indices)} events")
    subset = dataset[indices]
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=False,
    )

    model = Transformer(
        attn_type="hept",
        in_dim=dataset.x_dim,
        coords_dim=dataset.coords_dim,
        task=cfg["dataset_name"],
        **cfg["model_kwargs"],
    ).to(device)
    model.eval()

    _load_checkpoint(model, cfg["checkpoint_path"], device)
    model = _apply_torchao_quant(model)
    model = _apply_fp16_include(model)

    # Mirror src/utils/get_model.py: patch nvcc probe + compile encode/decode/model.
    _patch_dynamo_nvcc_probe()
    compile_mode = os.environ.get("HEPTV2_CODEX_COMPILE_MODE", "").strip()
    decode_compile_mode = os.environ.get("HEPTV2_CODEX_DECODE_COMPILE_MODE", compile_mode).strip()
    encode_compile_kwargs = _compile_kwargs_from_mode(compile_mode)
    decode_compile_kwargs = _compile_kwargs_from_mode(decode_compile_mode)
    compact_assignment = _compact_assignment_active(cfg, model)
    batched_decode = bool(cfg.get("postprocessing", {}).get("decoder_batched_decode", False))
    if hasattr(model, "encode_online"):
        model.encode_online = torch.compile(model.encode_online, **encode_compile_kwargs)
    else:
        model.encode = torch.compile(model.encode, **encode_compile_kwargs)
    compile_batched_decode = bool(cfg.get("postprocessing", {}).get("decoder_compile_batched_decode", False))
    if compact_assignment:
        if compile_batched_decode:
            model.decode_assignment_inputs = torch.compile(model.decode_assignment_inputs, **decode_compile_kwargs)
    elif batched_decode and not compile_batched_decode:
        print("[compile] decoder_batched_decode=true: keeping decoder eager to avoid slow batched graph compilation.")
    else:
        model.decode = torch.compile(model.decode, **decode_compile_kwargs)
        if not batched_decode:
            model = torch.compile(model, **decode_compile_kwargs)

    metric_sums = defaultdict(list)
    limit = cfg["eval"].get("limit_events")
    for i, data in enumerate(tqdm(loader)):
        if limit is not None and i >= limit:
            break
        metrics = run_event(model, data, cfg, device, amp_enabled, amp_dtype)
        for k, v in metrics.items():
            if isinstance(v, float) and np.isnan(v):
                continue
            metric_sums[k].append(v)

    print("\n=== aggregated metrics (mean over events) ===")
    for k in sorted(metric_sums):
        vals = np.asarray(metric_sums[k], dtype=np.float64)
        print(f"  {k:>24s}: {vals.mean():.8f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
