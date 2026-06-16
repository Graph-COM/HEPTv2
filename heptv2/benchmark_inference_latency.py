"""Benchmark the online inference path after an event is resident on the GPU.

This is intentionally separate from `run_inference.py`: metrics and data
loading are useful for evaluation, but they obscure the latency of the path we
would actually run online:

minimal preprocessing -> encoder/decoder -> shared post-processing assignment.
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.amp import autocast
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .data.dataset import TrackingTransform, TrackmlLarge
from .data.preprocess import process_data, process_inference_data
from .eval.postprocess import (
    _build_decoder_assignments,
    _compute_clf_keep_mask,
    _merge_similar_clusters_direct,
    decoder_kmeans_postprocess,
    trim_pred_by_valid_mask,
)
from .eval.compact_postprocess import postprocess_compact_assignment
from .model import Transformer
from .run_inference import (
    _apply_fp16_include,
    _apply_torchao_quant,
    _compact_assignment_active,
    _compile_kwargs_from_mode,
    _load_checkpoint,
    _mark_cudagraph_step,
    _needs_encoder_clf,
    _patch_dynamo_nvcc_probe,
)
from .utils.block_size_utils import resolve_block_sizes
from .utils.config_utils import apply_overrides


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _postprocess_fixed(raw_masks, sub_events, full_data, cfg):
    pp = cfg["postprocessing"]
    return decoder_kmeans_postprocess(
        raw_masks,
        sub_events,
        full_data,
        full_emb=full_data.emb_all,
        full_clf_scores=getattr(full_data, "clf_scores", None),
        pre_kmeans_iters=pp.get("pre_kmeans_iters", 0),
        pre_kmeans_min_margin=pp.get("pre_kmeans_min_margin", 0.0),
        kmeans_iters=pp.get("kmeans_iters", 0),
        kmeans_min_margin=pp.get("kmeans_min_margin", 0.0),
        kmeans_use_fp16_sim=pp.get("kmeans_use_fp16_sim", False),
        kmeans_max_refine_cluster_size=pp.get("kmeans_max_refine_cluster_size", 0),
        kmeans_refine_max_score=pp.get("kmeans_refine_max_score", 0.0),
        clf_recall=pp.get("clf_recall", 1.0),
        decoder_score_threshold=pp.get("decoder_score_threshold", 0.0),
        decoder_active_prob_threshold=pp.get("decoder_active_prob_threshold", 0.0),
        decoder_active_prob_power=pp.get("decoder_active_prob_power", 1.0),
        decoder_active_prob_floor=pp.get("decoder_active_prob_floor", 0.0),
        decoder_active_query_topk=pp.get("decoder_active_query_topk", 0),
        decoder_use_logit_scores=pp.get("decoder_use_logit_scores", False),
        decoder_score_mode=pp.get("decoder_score_mode", "prob"),
        decoder_score_bf16=pp.get("decoder_score_bf16", False),
        decoder_compile_assign=pp.get("decoder_compile_assign", False),
        decoder_batch_assignments=pp.get("decoder_batch_assignments", False),
        codex_fused_nomask_target=pp.get("codex_fused_nomask_target", False),
        decoder_primary_sub_events=pp.get("decoder_primary_sub_events", 0),
        embeddings_already_normalized=pp.get("embeddings_already_normalized", False),
        stitch_overlap=pp.get("stitch_overlap", False),
        stitch_score_gate=pp.get("stitch_score_gate", 0.02),
        stitch_min_overlap=pp.get("stitch_min_overlap", 1),
        stitch_min_overlap_frac=pp.get("stitch_min_overlap_frac", 0.6),
        stitch_query_topk=pp.get("stitch_query_topk", 1),
        stitch_union_topk=pp.get("stitch_union_topk", True),
        stitch_topk_score_gate=pp.get("stitch_topk_score_gate", None),
        stitch_topk_assign=pp.get("stitch_topk_assign", False),
        stitch_topk_assign_score_gate=pp.get("stitch_topk_assign_score_gate", 0.0),
        stitch_source_matrix=pp.get("stitch_source_matrix", False),
        pad_size2=pp.get("pad_size2", False),
        pad_size2_after_kmeans=pp.get("pad_size2_after_kmeans", False),
        pad_size2_min_sim=pp.get("pad_size2_min_sim", 0.0),
        pad_size2_clf_floor=pp.get("pad_size2_clf_floor", 0.0),
        pad_size2_clf_ceiling=pp.get("pad_size2_clf_ceiling", 0.0),
        pad_size2_unique_layer=pp.get("pad_size2_unique_layer", True),
        pad_size2_topk=pp.get("pad_size2_topk", 8),
        pad_size2_max_candidates=pp.get("pad_size2_max_candidates", 0),
        pad_size2_max_clusters=pp.get("pad_size2_max_clusters", 0),
        pad_size2_max_conflict_size=pp.get("pad_size2_max_conflict_size", 0),
        pad_size2_geom_z_max=pp.get("pad_size2_geom_z_max", 0.0),
        pad_size2_geom_phi_max=pp.get("pad_size2_geom_phi_max", 0.0),
        pad_size2_geom_only=pp.get("pad_size2_geom_only", False),
        pad_size2_geom_clf_weight=pp.get("pad_size2_geom_clf_weight", 0.0),
        pad_size2_rank_clusters_by_clf=pp.get("pad_size2_rank_clusters_by_clf", True),
        pad_size2_rank_candidates_by_clf=pp.get("pad_size2_rank_candidates_by_clf", True),
        merge_similar=pp.get("merge_similar", False),
        merge_similar_direct=pp.get("merge_similar_direct", False),
        merge_similar_label_space=pp.get("merge_similar_label_space", False),
        merge_similar_fuse_assign_merge=pp.get("merge_similar_fuse_assign_merge", False),
        merge_similar_min_sim=pp.get("merge_similar_min_sim", 0.9),
        merge_similar_small_cluster_size=pp.get("merge_similar_small_cluster_size", 2),
        merge_similar_max_merged_size=pp.get("merge_similar_max_merged_size", 12),
        merge_similar_topk=pp.get("merge_similar_topk", 4),
        merge_similar_require_unique_layer=pp.get("merge_similar_require_unique_layer", False),
        merge_similar_direct_target_min_size=pp.get("merge_similar_direct_target_min_size", 0),
        merge_similar_bf16_sim=pp.get("merge_similar_bf16_sim", False),
        merge_geometric_small=pp.get("merge_geometric_small", False),
        merge_geometric_small_min_size=pp.get("merge_geometric_small_min_size", 2),
        merge_geometric_small_max_size=pp.get("merge_geometric_small_max_size", 4),
        merge_geometric_small_z_max=pp.get("merge_geometric_small_z_max", 0.003),
        merge_geometric_small_phi_max=pp.get("merge_geometric_small_phi_max", 0.03),
        merge_geometric_small_target_min_size=pp.get("merge_geometric_small_target_min_size", 1),
        merge_geometric_small_require_unique_layer=pp.get("merge_geometric_small_require_unique_layer", True),
        merge_geometric_endpoint=pp.get("merge_geometric_endpoint", False),
        merge_geometric_endpoint_min_size=pp.get("merge_geometric_endpoint_min_size", 2),
        merge_geometric_endpoint_max_size=pp.get("merge_geometric_endpoint_max_size", 7),
        merge_geometric_endpoint_z_max=pp.get("merge_geometric_endpoint_z_max", 0.002),
        merge_geometric_endpoint_phi_max=pp.get("merge_geometric_endpoint_phi_max", 0.02),
        merge_geometric_endpoint_r_overlap_max=pp.get("merge_geometric_endpoint_r_overlap_max", 0.01),
        merge_geometric_endpoint_target_min_size=pp.get("merge_geometric_endpoint_target_min_size", 1),
        merge_geometric_endpoint_max_layer_overlap=pp.get("merge_geometric_endpoint_max_layer_overlap", 0),
        merge_same_query=pp.get("merge_same_query", False),
        dedup_same_layer=pp.get("dedup_same_layer", False),
        dedup_same_layer_min_cluster_size=pp.get("dedup_same_layer_min_cluster_size", 4),
        dedup_same_layer_min_size_after=pp.get("dedup_same_layer_min_size_after", 3),
        dedup_same_layer_min_sim_gap=pp.get("dedup_same_layer_min_sim_gap", 0.0),
    )



class _CompactPostprocessCUDAGraphCache:
    def __init__(self, cfg):
        self.cfg = cfg
        self.entries = {}

    @staticmethod
    def _key(compact_pred, full_data):
        point_order = compact_pred.get("_serialized_point_order", None)
        return (
            tuple(compact_pred["mask_embed"].shape),
            tuple(compact_pred["point_feats"].shape),
            tuple(compact_pred["pred_logits"].shape),
            None if point_order is None else tuple(point_order.shape),
            tuple(full_data.x.shape),
            tuple(full_data.decoder_sub_all_idx.shape),
            tuple(full_data.valid_mask.shape),
        )

    @staticmethod
    def _make_static_full_data(full_data):
        static = SimpleNamespace(
            x=torch.empty_like(full_data.x),
            coords=full_data.coords.clone(),
            valid_mask=full_data.valid_mask.clone(),
            decoder_sub_all_idx=full_data.decoder_sub_all_idx.clone(),
            decoder_sub_valid_mask=full_data.decoder_sub_valid_mask.clone(),
            decoder_overlap=getattr(full_data, "decoder_overlap", False),
            emb_all=full_data.emb_all.clone(),
            clf_scores=full_data.clf_scores.clone(),
        )
        if hasattr(full_data, "layer") and full_data.layer is not None:
            static.layer = full_data.layer.clone()
        return static

    @staticmethod
    def _copy_full_data(dst, src):
        dst.coords.copy_(src.coords)
        dst.valid_mask.copy_(src.valid_mask)
        dst.decoder_sub_all_idx.copy_(src.decoder_sub_all_idx)
        dst.decoder_sub_valid_mask.copy_(src.decoder_sub_valid_mask)
        dst.emb_all.copy_(src.emb_all)
        dst.clf_scores.copy_(src.clf_scores)
        if hasattr(dst, "layer") and hasattr(src, "layer") and src.layer is not None:
            dst.layer.copy_(src.layer)

    def _make_entry(self, compact_pred, full_data):
        static_pred = {
            "mask_embed": compact_pred["mask_embed"].clone(),
            "point_feats": compact_pred["point_feats"].clone(),
            "pred_logits": compact_pred["pred_logits"].clone(),
        }
        point_order = compact_pred.get("_serialized_point_order", None)
        if point_order is not None:
            static_pred["_serialized_point_order"] = point_order.clone()
        static_data = self._make_static_full_data(full_data)

        for _ in range(3):
            static_out = postprocess_compact_assignment(static_pred, static_data, self.cfg)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = postprocess_compact_assignment(static_pred, static_data, self.cfg)
        torch.cuda.synchronize()
        return {
            "graph": graph,
            "pred": static_pred,
            "data": static_data,
            "out": static_out,
        }

    def run(self, compact_pred, full_data):
        key = self._key(compact_pred, full_data)
        entry = self.entries.get(key)
        if entry is None:
            entry = self._make_entry(compact_pred, full_data)
            self.entries[key] = entry
        entry["pred"]["mask_embed"].copy_(compact_pred["mask_embed"])
        entry["pred"]["point_feats"].copy_(compact_pred["point_feats"])
        entry["pred"]["pred_logits"].copy_(compact_pred["pred_logits"])
        if "_serialized_point_order" in entry["pred"]:
            entry["pred"]["_serialized_point_order"].copy_(compact_pred["_serialized_point_order"])
        self._copy_full_data(entry["data"], full_data)
        entry["graph"].replay()
        return entry["out"]


def _postprocess_lazy_embedding(raw_masks, sub_events, full_data, cfg, model, emb_source):
    pp = cfg["postprocessing"]
    clf_keep = _compute_clf_keep_mask(
        full_data,
        getattr(full_data, "clf_scores", None),
        clf_recall=pp.get("clf_recall", 1.0),
    )
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
        batch_assignments=pp.get("decoder_batch_assignments", False),
        max_sub_events=pp.get("decoder_primary_sub_events", 0),
        return_scores=False,
    )
    if not pp.get("merge_similar", False):
        return predicted
    active = full_data.valid_mask.bool() & (predicted >= 0)
    active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
    if active_idx.numel() > 0:
        with autocast(device_type="cuda", enabled=emb_source.is_cuda, dtype=torch.bfloat16):
            emb_active = torch.nn.functional.normalize(
                model.mlp_out(emb_source.index_select(0, active_idx)),
                p=2,
                dim=-1,
            )
        emb = torch.empty(
            (full_data.x.shape[0], int(emb_active.shape[1])),
            device=emb_active.device,
            dtype=emb_active.dtype,
        )
        emb.index_copy_(0, active_idx, emb_active)
    else:
        emb = torch.empty(
            (full_data.x.shape[0], int(emb_source.shape[1])),
            device=emb_source.device,
            dtype=emb_source.dtype,
        )
    full_data.emb_all = emb
    if pp.get("merge_similar", False):
        predicted = _merge_similar_clusters_direct(
            predicted,
            full_data,
            full_data.emb_all,
            min_sim=pp.get("merge_similar_min_sim", 0.9),
            small_cluster_size=pp.get("merge_similar_small_cluster_size", 2),
            max_merged_size=pp.get("merge_similar_max_merged_size", 12),
            target_min_size=pp.get("merge_similar_direct_target_min_size", 0),
            normalized_emb=full_data.emb_all if pp.get("embeddings_already_normalized", False) else None,
            use_bf16_sim=pp.get("merge_similar_bf16_sim", False),
        )
    return predicted


def _event_ms(start, end):
    return float(start.elapsed_time(end))


@torch.inference_mode()
def run_timed_event(
    model,
    data_gpu,
    cfg,
    amp_enabled,
    amp_dtype,
    mode,
    materialize_decode_output=False,
    compact_assignment=False,
    dynamic_pad=False,
    pad_bucket_lengths=None,
    lazy_embedding=False,
    post_graph_cache=None,
):
    if data_gpu.x.is_cuda and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
    enc_cfg = cfg["model_kwargs"]
    enc_block, dec_block = resolve_block_sizes(enc_cfg)
    pp = cfg["postprocessing"]

    events = {name: torch.cuda.Event(enable_timing=True) for name in (
        "start",
        "pre",
        "encode",
        "decode",
        "post",
    )}

    torch.cuda.synchronize()
    events["start"].record()
    if mode == "legacy":
        full_data, sub_events = process_data(
            data_gpu,
            fixed_len=enc_cfg["fixed_seq_len"],
            encoder_block_size=enc_block,
            decoder_block_size=dec_block,
            regions=model.helper_funcs["regions"],
            num_sub_events=cfg["eval"].get("num_sub_events", 3),
            decoder_overlap=enc_cfg.get("decoder_overlap", True),
            eta_abs_max=cfg["eval"]["eta_abs_max"],
        )
    elif mode == "minimal":
        full_data, sub_events = process_inference_data(
            data_gpu,
            fixed_len=enc_cfg["fixed_seq_len"],
            encoder_block_size=enc_block,
            decoder_block_size=dec_block,
            regions=model.helper_funcs["regions"],
            num_sub_events=cfg["eval"].get("num_sub_events", 3),
            decoder_overlap=enc_cfg.get("decoder_overlap", True),
            eta_abs_max=cfg["eval"]["eta_abs_max"],
            return_sub_events=False,
            dynamic_pad=dynamic_pad,
            pad_bucket_lengths=pad_bucket_lengths,
            match_fixed_region_bins=bool(cfg.get("inference_path", {}).get("match_fixed_region_bins", False)),
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    events["pre"].record()

    raw_masks = []
    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
        need_clf = _needs_encoder_clf(cfg)
        if hasattr(model, "encode_online"):
            _mark_cudagraph_step()
            enc_out = model.encode_online(
                full_data,
                need_clf=need_clf,
                need_emb=not bool(lazy_embedding),
                return_emb_source=bool(lazy_embedding),
            )
        else:
            _mark_cudagraph_step()
            enc_out = model.encode(full_data)
        if not bool(lazy_embedding):
            full_data.emb_all = enc_out["emb_all"].detach()
            if pp.get("embedding_output_bf16", False) and full_data.emb_all.is_cuda:
                full_data.emb_all = full_data.emb_all.to(torch.bfloat16)
        clf_probs = enc_out.get("clf_probs")
        full_data.clf_scores = None if clf_probs is None else clf_probs.detach()
    events["encode"].record()

    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
        if pp.get("decoder_batched_decode", False):
            all_idx = full_data.decoder_sub_all_idx
            valid_masks = full_data.decoder_sub_valid_mask
            point_coords = (
                full_data.coords[all_idx]
                if getattr(model.mask3d_head, "needs_point_coords", True)
                else None
            )
            if bool(compact_assignment):
                _mark_cudagraph_step()
                compact_pred = model.decode_assignment_inputs(
                    enc_out["encoded"][all_idx],
                    point_coords,
                    valid_masks,
                )
            else:
                _mark_cudagraph_step()
                pred = model.decode(
                    enc_out["encoded"][all_idx],
                    point_coords,
                    valid_masks,
                )
                if bool(materialize_decode_output):
                    pred["pred_masks"] = pred["pred_masks"].clone(memory_format=torch.contiguous_format)
                    pred["pred_logits"] = pred["pred_logits"].clone(memory_format=torch.contiguous_format)
                if pp.get("decoder_output_bf16", False) and pred["pred_masks"].is_cuda:
                    pred["pred_masks"] = pred["pred_masks"].to(torch.bfloat16)
                raw_masks.append(
                    (
                        pred["pred_masks"].detach(),
                        pred["pred_logits"].detach(),
                        pred.get("_serialized_point_order", None),
                    )
                )
        else:
            if sub_events is None:
                raise RuntimeError("non-batched decode requires sub_events")
            for sub in sub_events:
                point_coords = (
                    full_data.coords[sub["all_idx"]]
                    if getattr(model.mask3d_head, "needs_point_coords", True)
                    else None
                )
                _mark_cudagraph_step()
                pred = model.decode(
                    enc_out["encoded"][sub["all_idx"]],
                    point_coords,
                    sub["valid_mask"],
                )
                if bool(materialize_decode_output):
                    pred["pred_masks"] = pred["pred_masks"].clone(memory_format=torch.contiguous_format)
                    pred["pred_logits"] = pred["pred_logits"].clone(memory_format=torch.contiguous_format)
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
    events["decode"].record()

    if bool(compact_assignment):
        if bool(lazy_embedding):
            with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                full_data.emb_all = torch.nn.functional.normalize(
                    model.mlp_out(enc_out["emb_source"]),
                    p=2,
                    dim=-1,
                ).detach()
            if pp.get("embedding_output_bf16", False) and full_data.emb_all.is_cuda:
                full_data.emb_all = full_data.emb_all.to(torch.bfloat16)
        if post_graph_cache is not None:
            predicted = post_graph_cache.run(compact_pred, full_data)
        else:
            _mark_cudagraph_step()
            predicted = postprocess_compact_assignment(compact_pred, full_data, cfg)
    elif bool(lazy_embedding):
        predicted = _postprocess_lazy_embedding(
            raw_masks,
            sub_events,
            full_data,
            cfg,
            model,
            enc_out["emb_source"].detach(),
        )
    else:
        predicted = _postprocess_fixed(raw_masks, sub_events, full_data, cfg)
    events["post"].record()
    torch.cuda.synchronize()

    # Touch the result so accidental dead-code-style changes cannot hide work.
    assigned = int((predicted >= 0).sum().item())
    return {
        "pre_ms": _event_ms(events["start"], events["pre"]),
        "encode_ms": _event_ms(events["pre"], events["encode"]),
        "decode_ms": _event_ms(events["encode"], events["decode"]),
        "model_ms": _event_ms(events["pre"], events["decode"]),
        "post_ms": _event_ms(events["decode"], events["post"]),
        "total_ms": _event_ms(events["start"], events["post"]),
        "assigned": assigned,
        "n_total": int(full_data.x.shape[0]),
    }


def _summarize(rows):
    out = {}
    for key in ("pre_ms", "encode_ms", "decode_ms", "model_ms", "post_ms", "total_ms"):
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
        }
    out["assigned_mean"] = float(np.asarray([row["assigned"] for row in rows]).mean())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "infer.yaml"))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--mode", choices=("legacy", "minimal"), default="minimal")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default=os.environ.get("HEPTV2_CODEX_COMPILE_MODE", "default"),
    )
    parser.add_argument("--compile-batched-decode", action="store_true")
    parser.add_argument("--materialize-decode-output", action="store_true")
    parser.add_argument("--model-bf16", action="store_true")
    parser.add_argument("--model-fp16", action="store_true")
    parser.add_argument("--compact-assignment", action="store_true")
    parser.add_argument("--dynamic-pad", action="store_true")
    parser.add_argument("--pad-buckets", type=str, default="")
    parser.add_argument("--skip-first-per-shape", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--lazy-embedding", action="store_true")
    parser.add_argument("--warmup-events", type=int, default=8)
    parser.add_argument("--limit-events", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    path_cfg = cfg.get("inference_path", {})
    if path_cfg.get("compact_assignment", False):
        args.compact_assignment = True
    if path_cfg.get("dynamic_pad", False):
        args.dynamic_pad = True
    if not args.pad_buckets and path_cfg.get("pad_buckets"):
        buckets = path_cfg["pad_buckets"]
        if isinstance(buckets, (list, tuple)):
            args.pad_buckets = ",".join(str(v) for v in buckets)
        else:
            args.pad_buckets = str(buckets)
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

    dataset = TrackmlLarge(root=Path(cfg["data_root"]), transform=TrackingTransform())
    indices = dataset.idx_split[cfg.get("split", "test")]
    subset = dataset[indices]
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
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
    if args.model_fp16:
        model = model.to(torch.float16)
    elif args.model_bf16:
        model = model.to(torch.bfloat16)
    _patch_dynamo_nvcc_probe()
    decode_compile_mode = os.environ.get("HEPTV2_CODEX_DECODE_COMPILE_MODE", args.compile_mode).strip()
    compile_kwargs = _compile_kwargs_from_mode(args.compile_mode)
    decode_compile_kwargs = _compile_kwargs_from_mode(decode_compile_mode)
    compile_kwargs["dynamic"] = bool(args.compile_dynamic)
    decode_compile_kwargs["dynamic"] = bool(args.compile_dynamic)
    if args.compile_fullgraph:
        compile_kwargs["fullgraph"] = True
        decode_compile_kwargs["fullgraph"] = True
    if args.compile_mode != "default":
        compile_kwargs["mode"] = args.compile_mode
    if decode_compile_mode and decode_compile_mode != "default":
        decode_compile_kwargs["mode"] = decode_compile_mode
    if hasattr(model, "encode_online"):
        model.encode_online = torch.compile(model.encode_online, **compile_kwargs)
    else:
        model.encode = torch.compile(model.encode, **compile_kwargs)
    pp_cfg = cfg.get("postprocessing", {})
    compact_assignment = _compact_assignment_active(cfg, model) and bool(args.compact_assignment)
    compile_batched_decode = bool(args.compile_batched_decode or pp_cfg.get("decoder_compile_batched_decode", False))
    materialize_decode_output = bool(args.materialize_decode_output or pp_cfg.get("decoder_materialize_output", False))
    if compact_assignment and compile_batched_decode:
        model.decode_assignment_inputs = torch.compile(model.decode_assignment_inputs, **decode_compile_kwargs)
    elif compile_batched_decode or not pp_cfg.get("decoder_batched_decode", False):
        model.decode = torch.compile(model.decode, **decode_compile_kwargs)

    measured = []
    seen_shapes = set()
    post_graph_cache = (
        _CompactPostprocessCUDAGraphCache(cfg)
        if compact_assignment and bool(int(os.environ.get("HEPTV2_POST_CUDAGRAPH", "0")))
        else None
    )
    pad_bucket_lengths = [
        int(part) for part in args.pad_buckets.split(",") if part.strip()
    ] or None
    warmup = int(args.warmup_events)
    limit = int(args.limit_events)
    max_events = len(indices) if limit <= 0 else min(limit, len(indices))
    for i, data in enumerate(tqdm(loader, total=max_events)):
        if i >= max_events:
            break
        data_gpu = data.to(device)
        row = run_timed_event(
            model,
            data_gpu,
            cfg,
            amp_enabled,
            amp_dtype,
            args.mode,
            materialize_decode_output=materialize_decode_output,
            compact_assignment=compact_assignment,
            dynamic_pad=args.dynamic_pad,
            pad_bucket_lengths=pad_bucket_lengths,
            lazy_embedding=args.lazy_embedding,
            post_graph_cache=post_graph_cache,
        )
        shape_key = int(row["n_total"])
        first_shape = shape_key not in seen_shapes
        seen_shapes.add(shape_key)
        if i >= warmup and not (args.skip_first_per_shape and first_shape):
            measured.append(row)

    summary = _summarize(measured)
    print(f"\n=== latency summary: mode={args.mode}, measured_events={len(measured)} ===")
    for key in ("pre_ms", "encode_ms", "decode_ms", "model_ms", "post_ms", "total_ms"):
        vals = summary[key]
        print(f"  {key:>10s}: mean={vals['mean']:.3f}  p50={vals['p50']:.3f}  p90={vals['p90']:.3f}")
    print(f"  assigned_mean: {summary['assigned_mean']:.1f}")


if __name__ == "__main__":
    main()
