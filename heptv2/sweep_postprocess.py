"""Run one forward pass per event and sweep post-processing variants."""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
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
from .utils.block_size_utils import resolve_block_sizes
from .utils.config_utils import apply_overrides


PP_KEYS = [
    "pre_kmeans_iters",
    "pre_kmeans_min_margin",
    "kmeans_iters",
    "kmeans_min_margin",
    "kmeans_use_fp16_sim",
    "kmeans_max_refine_cluster_size",
    "kmeans_refine_max_score",
    "clf_recall",
    "decoder_score_threshold",
    "decoder_active_prob_threshold",
    "decoder_active_prob_power",
    "decoder_active_prob_floor",
    "decoder_active_query_topk",
    "decoder_use_logit_scores",
    "decoder_score_mode",
    "decoder_score_bf16",
    "decoder_compile_assign",
    "decoder_batch_assignments",
    "decoder_primary_sub_events",
    "embeddings_already_normalized",
    "cluster_score_filter",
    "cluster_score_filter_max_size",
    "cluster_score_filter_mean_threshold",
    "cluster_score_filter_min_threshold",
    "stitch_overlap",
    "stitch_score_gate",
    "stitch_min_overlap",
    "stitch_min_overlap_frac",
    "stitch_query_topk",
    "stitch_union_topk",
    "stitch_topk_score_gate",
    "stitch_topk_assign",
    "stitch_topk_assign_score_gate",
    "stitch_source_matrix",
    "extra_window_bridge",
    "extra_window_bridge_max_cluster_size",
    "extra_window_bridge_max_merged_size",
    "extra_window_bridge_min_score",
    "extra_window_bridge_min_label_hits",
    "extra_window_bridge_max_hits",
    "extra_window_bridge_query_topk",
    "restore_topk_components",
    "restore_topk_min_score",
    "restore_topk_max_conflict_size",
    "restore_topk_min_target_size",
    "restore_topk_unique_layer",
    "restore_window_clusters",
    "restore_window_min_size",
    "restore_window_min_new_hits",
    "restore_window_min_score",
    "restore_window_max_conflict_size",
    "restore_window_min_new_frac",
    "bridge_window_clusters",
    "bridge_window_min_score",
    "bridge_window_min_candidate_size",
    "bridge_window_min_label_overlap",
    "bridge_window_max_cluster_size",
    "bridge_window_max_merged_size",
    "pad_window_size2",
    "pad_window_min_score",
    "pad_window_extra_min_score",
    "pad_window_min_overlap",
    "pad_window_unique_layer",
    "pad_window_max_proposals_per_label",
    "split_window_clusters",
    "split_window_min_score",
    "split_window_min_cluster_size",
    "split_window_max_cluster_size",
    "split_window_min_part_size",
    "split_window_min_remaining_size",
    "split_window_min_inside_frac",
    "split_window_max_part_frac",
    "split_window_min_duplicate_layers",
    "split_window_max_part_duplicate_layers",
    "split_window_require_remainder_candidate",
    "split_window_min_remainder_candidate_hits",
    "split_window_max_splits",
    "pad_size2",
    "pad_size2_after_kmeans",
    "pad_size2_min_sim",
    "pad_size2_clf_floor",
    "pad_size2_clf_ceiling",
    "pad_size2_unique_layer",
    "pad_size2_topk",
    "pad_size2_max_candidates",
    "pad_size2_max_clusters",
    "pad_size2_max_conflict_size",
    "pad_size2_geom_z_max",
    "pad_size2_geom_phi_max",
    "pad_size2_rank_clusters_by_clf",
    "pad_size2_rank_candidates_by_clf",
    "merge_similar",
    "merge_similar_direct",
    "merge_similar_label_space",
    "merge_similar_min_sim",
    "merge_similar_small_cluster_size",
    "merge_similar_max_merged_size",
    "merge_similar_topk",
    "merge_similar_require_unique_layer",
    "merge_similar_direct_target_min_size",
    "merge_similar_bf16_sim",
    "merge_same_query",
    "split_overmerged",
    "split_overmerged_min_size",
    "split_overmerged_max_size",
    "split_overmerged_min_part_size",
    "split_overmerged_min_duplicate_layers",
    "split_overmerged_max_part_duplicate_layers",
    "split_overmerged_min_improve",
    "split_overmerged_max_centroid_sim",
    "split_overmerged_iters",
    "split_overmerged_max_clusters",
    "residual_regroup",
    "residual_regroup_clf_threshold",
    "residual_regroup_max_cluster_size",
    "residual_regroup_topk",
    "residual_regroup_seed_sim",
    "residual_regroup_pair_sim",
    "residual_regroup_min_group_size",
    "residual_regroup_max_group_size",
    "residual_regroup_require_unique_layer",
    "residual_regroup_clear_pool",
    "residual_regroup_pair_repair",
    "residual_regroup_pair_repair_sim",
    "residual_regroup_pair_repair_topk",
    "residual_regroup_max_hits",
    "dedup_same_layer",
    "dedup_same_layer_min_cluster_size",
    "dedup_same_layer_min_size_after",
    "dedup_same_layer_min_sim_gap",
    "dedup_same_layer_by_score",
    "dedup_same_layer_score_min_cluster_size",
    "dedup_same_layer_score_min_size_after",
    "dedup_same_layer_score_min_gap",
    "prune_small_clusters",
    "prune_small_cluster_max_size",
    "prune_small_cluster_min_size_after",
    "prune_small_cluster_max_drop",
    "prune_small_cluster_min_sim_gap",
    "prune_small_cluster_min_score_gap",
    "prune_small_cluster_require_duplicate_layer",
    "prune_small_cluster_duplicate_layer_only",
]


def _variant_presets(name, base_pp):
    def pp(**kwargs):
        out = dict(base_pp)
        out.update(kwargs)
        return out

    current = pp(residual_regroup=False)
    if name == "residual_smoke":
        return [
            ("current", current),
            (
                "rg015_clear_layer",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=32,
                    residual_regroup_seed_sim=0.50,
                    residual_regroup_pair_sim=0.10,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=True,
                ),
            ),
            (
                "rg015_keep_layer",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=32,
                    residual_regroup_seed_sim=0.50,
                    residual_regroup_pair_sim=0.10,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                ),
            ),
            (
                "rg015_keep_nolayer",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=32,
                    residual_regroup_seed_sim=0.50,
                    residual_regroup_pair_sim=0.10,
                    residual_regroup_require_unique_layer=False,
                    residual_regroup_clear_pool=False,
                ),
            ),
            (
                "rg020_keep_layer",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.020,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=32,
                    residual_regroup_seed_sim=0.50,
                    residual_regroup_pair_sim=0.10,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                ),
            ),
            (
                "rg015_cons_s2",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=2,
                    residual_regroup_topk=24,
                    residual_regroup_seed_sim=0.60,
                    residual_regroup_pair_sim=0.20,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                ),
            ),
        ]
    if name == "residual_pair":
        return [
            ("current", current),
            (
                "rg015_pair",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=32,
                    residual_regroup_seed_sim=0.50,
                    residual_regroup_pair_sim=0.10,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_pair_repair=True,
                    residual_regroup_pair_repair_sim=0.30,
                    residual_regroup_pair_repair_topk=16,
                ),
            ),
            (
                "rg015_pair_cap4k",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=32,
                    residual_regroup_seed_sim=0.50,
                    residual_regroup_pair_sim=0.10,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_pair_repair=True,
                    residual_regroup_pair_repair_sim=0.30,
                    residual_regroup_pair_repair_topk=16,
                    residual_regroup_max_hits=4096,
                ),
            ),
            (
                "rg010_pair_cap8k",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.010,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=32,
                    residual_regroup_seed_sim=0.55,
                    residual_regroup_pair_sim=0.15,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_pair_repair=True,
                    residual_regroup_pair_repair_sim=0.35,
                    residual_regroup_pair_repair_topk=16,
                    residual_regroup_max_hits=8192,
                ),
            ),
            (
                "rg015_pair_pad04",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=32,
                    residual_regroup_seed_sim=0.50,
                    residual_regroup_pair_sim=0.10,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_pair_repair=True,
                    residual_regroup_pair_repair_sim=0.30,
                    residual_regroup_pair_repair_topk=16,
                    pad_size2=True,
                    pad_size2_min_sim=0.4,
                ),
            ),
        ]
    if name == "residual_fast":
        return [
            ("current", current),
            (
                "rg015_cap2k_t16",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=16,
                    residual_regroup_seed_sim=0.55,
                    residual_regroup_pair_sim=0.15,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                ),
            ),
            (
                "rg015_cap4k_t24",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=24,
                    residual_regroup_seed_sim=0.55,
                    residual_regroup_pair_sim=0.15,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=4096,
                ),
            ),
            (
                "rg020_cap2k_t16",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.020,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=16,
                    residual_regroup_seed_sim=0.55,
                    residual_regroup_pair_sim=0.15,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                ),
            ),
            (
                "rg015_s2_cap2k",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=2,
                    residual_regroup_topk=16,
                    residual_regroup_seed_sim=0.55,
                    residual_regroup_pair_sim=0.15,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                ),
            ),
        ]
    if name == "residual_tight":
        return [
            ("current", current),
            (
                "rg015_cap2k_t16_pad04",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=16,
                    residual_regroup_seed_sim=0.55,
                    residual_regroup_pair_sim=0.15,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                    pad_size2=True,
                    pad_size2_min_sim=0.4,
                ),
            ),
            (
                "rg015_cap2k_s60_pad04",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=16,
                    residual_regroup_seed_sim=0.60,
                    residual_regroup_pair_sim=0.20,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                    pad_size2=True,
                    pad_size2_min_sim=0.4,
                ),
            ),
            (
                "rg020_cap2k_s60_pad04",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.020,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=16,
                    residual_regroup_seed_sim=0.60,
                    residual_regroup_pair_sim=0.20,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                    pad_size2=True,
                    pad_size2_min_sim=0.4,
                ),
            ),
            (
                "rg015_s2_cap2k_pad04",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=2,
                    residual_regroup_topk=16,
                    residual_regroup_seed_sim=0.55,
                    residual_regroup_pair_sim=0.15,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                    pad_size2=True,
                    pad_size2_min_sim=0.4,
                ),
            ),
            (
                "rg015_cap2k_no_pad",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=3,
                    residual_regroup_topk=16,
                    residual_regroup_seed_sim=0.55,
                    residual_regroup_pair_sim=0.15,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                    pad_size2=False,
                ),
            ),
        ]
    if name == "padding_ceiling":
        return [
            ("current", current),
            (
                "pad02_ce005",
                pp(
                    pad_size2=True,
                    pad_size2_min_sim=0.2,
                    pad_size2_clf_floor=0.0,
                    pad_size2_clf_ceiling=0.005,
                ),
            ),
            (
                "pad02_ce010",
                pp(
                    pad_size2=True,
                    pad_size2_min_sim=0.2,
                    pad_size2_clf_floor=0.0,
                    pad_size2_clf_ceiling=0.010,
                ),
            ),
            (
                "pad02_ce015",
                pp(
                    pad_size2=True,
                    pad_size2_min_sim=0.2,
                    pad_size2_clf_floor=0.0,
                    pad_size2_clf_ceiling=0.015,
                ),
            ),
            (
                "pad025_ce010",
                pp(
                    pad_size2=True,
                    pad_size2_min_sim=0.25,
                    pad_size2_clf_floor=0.0,
                    pad_size2_clf_ceiling=0.010,
                ),
            ),
            (
                "pad03_ce010",
                pp(
                    pad_size2=True,
                    pad_size2_min_sim=0.3,
                    pad_size2_clf_floor=0.0,
                    pad_size2_clf_ceiling=0.010,
                ),
            ),
            (
                "rg_s2_pad02_ce010",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=2,
                    residual_regroup_topk=16,
                    residual_regroup_seed_sim=0.55,
                    residual_regroup_pair_sim=0.15,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                    pad_size2=True,
                    pad_size2_min_sim=0.2,
                    pad_size2_clf_floor=0.0,
                    pad_size2_clf_ceiling=0.010,
                ),
            ),
            (
                "rg_s2_pad025_ce010",
                pp(
                    residual_regroup=True,
                    residual_regroup_clf_threshold=0.015,
                    residual_regroup_max_cluster_size=2,
                    residual_regroup_topk=16,
                    residual_regroup_seed_sim=0.55,
                    residual_regroup_pair_sim=0.15,
                    residual_regroup_require_unique_layer=True,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                    pad_size2=True,
                    pad_size2_min_sim=0.25,
                    pad_size2_clf_floor=0.0,
                    pad_size2_clf_ceiling=0.010,
                ),
            ),
        ]
    if name == "stitch_clf":
        return [
            ("current", current),
            ("rec9990", pp(clf_recall=0.9990)),
            ("rec9998", pp(clf_recall=0.9998)),
            ("rec1000", pp(clf_recall=1.0)),
            (
                "gate000_rec9995",
                pp(
                    stitch_overlap=True,
                    stitch_score_gate=0.0,
                    clf_recall=0.9995,
                ),
            ),
            (
                "gate002_rec9998",
                pp(
                    stitch_overlap=True,
                    stitch_score_gate=0.002,
                    clf_recall=0.9998,
                ),
            ),
            (
                "gate000_rec9998",
                pp(
                    stitch_overlap=True,
                    stitch_score_gate=0.0,
                    clf_recall=0.9998,
                ),
            ),
            (
                "merge070",
                pp(
                    merge_similar=True,
                    merge_similar_min_sim=0.70,
                    merge_similar_small_cluster_size=2,
                    merge_similar_max_merged_size=12,
                    merge_similar_topk=4,
                    merge_similar_require_unique_layer=False,
                ),
            ),
            (
                "merge080",
                pp(
                    merge_similar=True,
                    merge_similar_min_sim=0.80,
                    merge_similar_small_cluster_size=2,
                    merge_similar_max_merged_size=12,
                    merge_similar_topk=4,
                    merge_similar_require_unique_layer=False,
                ),
            ),
        ]
    if name == "split_overmerge":
        split_base = dict(
            split_overmerged=True,
            split_overmerged_max_size=12,
            split_overmerged_min_part_size=3,
            split_overmerged_iters=4,
            split_overmerged_max_clusters=64,
        )
        rg_s2 = dict(
            residual_regroup=True,
            residual_regroup_clf_threshold=0.015,
            residual_regroup_max_cluster_size=2,
            residual_regroup_topk=16,
            residual_regroup_seed_sim=0.55,
            residual_regroup_pair_sim=0.15,
            residual_regroup_require_unique_layer=True,
            residual_regroup_clear_pool=False,
            residual_regroup_max_hits=2048,
            pad_size2=True,
            pad_size2_min_sim=0.4,
        )
        return [
            ("current", current),
            (
                "split_dup_i04_s85",
                pp(
                    **split_base,
                    split_overmerged_min_size=6,
                    split_overmerged_min_duplicate_layers=1,
                    split_overmerged_max_part_duplicate_layers=0,
                    split_overmerged_min_improve=0.04,
                    split_overmerged_max_centroid_sim=0.85,
                ),
            ),
            (
                "split_dup_i03_s90",
                pp(
                    **split_base,
                    split_overmerged_min_size=6,
                    split_overmerged_min_duplicate_layers=1,
                    split_overmerged_max_part_duplicate_layers=0,
                    split_overmerged_min_improve=0.03,
                    split_overmerged_max_centroid_sim=0.90,
                ),
            ),
            (
                "split_dup_i06_s80",
                pp(
                    **split_base,
                    split_overmerged_min_size=6,
                    split_overmerged_min_duplicate_layers=1,
                    split_overmerged_max_part_duplicate_layers=0,
                    split_overmerged_min_improve=0.06,
                    split_overmerged_max_centroid_sim=0.80,
                ),
            ),
            (
                "split_nodup_i06_s80",
                pp(
                    **split_base,
                    split_overmerged_min_size=6,
                    split_overmerged_min_duplicate_layers=0,
                    split_overmerged_max_part_duplicate_layers=0,
                    split_overmerged_min_improve=0.06,
                    split_overmerged_max_centroid_sim=0.80,
                ),
            ),
            (
                "split_force64",
                pp(
                    **split_base,
                    split_overmerged_min_size=6,
                    split_overmerged_min_duplicate_layers=0,
                    split_overmerged_max_part_duplicate_layers=99,
                    split_overmerged_min_improve=0.0,
                    split_overmerged_max_centroid_sim=1.0,
                ),
            ),
            (
                "split_force64b",
                pp(
                    **split_base,
                    split_overmerged_min_size=6,
                    split_overmerged_min_duplicate_layers=0,
                    split_overmerged_max_part_duplicate_layers=99,
                    split_overmerged_min_improve=0.0,
                    split_overmerged_max_centroid_sim=0.95,
                ),
            ),
            (
                "split_size5_dup",
                pp(
                    **split_base,
                    split_overmerged_min_size=5,
                    split_overmerged_min_duplicate_layers=1,
                    split_overmerged_max_part_duplicate_layers=0,
                    split_overmerged_min_improve=0.04,
                    split_overmerged_max_centroid_sim=0.85,
                ),
            ),
            (
                "split_rg_s2",
                pp(
                    **split_base,
                    **rg_s2,
                    split_overmerged_min_size=6,
                    split_overmerged_min_duplicate_layers=1,
                    split_overmerged_max_part_duplicate_layers=0,
                    split_overmerged_min_improve=0.04,
                    split_overmerged_max_centroid_sim=0.85,
                ),
            ),
            (
                "split_rg_s2_loose",
                pp(
                    **split_base,
                    **rg_s2,
                    split_overmerged_min_size=6,
                    split_overmerged_min_duplicate_layers=0,
                    split_overmerged_max_part_duplicate_layers=0,
                    split_overmerged_min_improve=0.06,
                    split_overmerged_max_centroid_sim=0.80,
                ),
            ),
        ]
    if name == "dedup_layer":
        rg_s2 = dict(
            residual_regroup=True,
            residual_regroup_clf_threshold=0.015,
            residual_regroup_max_cluster_size=2,
            residual_regroup_topk=16,
            residual_regroup_seed_sim=0.55,
            residual_regroup_pair_sim=0.15,
            residual_regroup_require_unique_layer=True,
            residual_regroup_clear_pool=False,
            residual_regroup_max_hits=2048,
            pad_size2=True,
            pad_size2_min_sim=0.4,
        )
        return [
            ("current", current),
            (
                "dedup4_gap0",
                pp(
                    dedup_same_layer=True,
                    dedup_same_layer_min_cluster_size=4,
                    dedup_same_layer_min_size_after=3,
                    dedup_same_layer_min_sim_gap=0.0,
                ),
            ),
            (
                "dedup4_gap02",
                pp(
                    dedup_same_layer=True,
                    dedup_same_layer_min_cluster_size=4,
                    dedup_same_layer_min_size_after=3,
                    dedup_same_layer_min_sim_gap=0.02,
                ),
            ),
            (
                "dedup5_gap0",
                pp(
                    dedup_same_layer=True,
                    dedup_same_layer_min_cluster_size=5,
                    dedup_same_layer_min_size_after=3,
                    dedup_same_layer_min_sim_gap=0.0,
                ),
            ),
            (
                "dedup6_gap0",
                pp(
                    dedup_same_layer=True,
                    dedup_same_layer_min_cluster_size=6,
                    dedup_same_layer_min_size_after=3,
                    dedup_same_layer_min_sim_gap=0.0,
                ),
            ),
            (
                "dedup_rg_s2",
                pp(
                    **rg_s2,
                    dedup_same_layer=True,
                    dedup_same_layer_min_cluster_size=4,
                    dedup_same_layer_min_size_after=3,
                    dedup_same_layer_min_sim_gap=0.0,
                ),
            ),
            (
                "dedup_rg_s2_gap02",
                pp(
                    **rg_s2,
                    dedup_same_layer=True,
                    dedup_same_layer_min_cluster_size=4,
                    dedup_same_layer_min_size_after=3,
                    dedup_same_layer_min_sim_gap=0.02,
                ),
            ),
        ]
    if name == "merge_residual":
        rg_s2 = dict(
            residual_regroup=True,
            residual_regroup_clf_threshold=0.015,
            residual_regroup_max_cluster_size=2,
            residual_regroup_topk=16,
            residual_regroup_seed_sim=0.55,
            residual_regroup_pair_sim=0.15,
            residual_regroup_require_unique_layer=True,
            residual_regroup_clear_pool=False,
            residual_regroup_max_hits=2048,
            pad_size2=True,
            pad_size2_min_sim=0.4,
        )
        return [
            ("current", current),
            ("merge_off", pp(merge_similar=False)),
            ("merge080", pp(merge_similar=True, merge_similar_min_sim=0.80)),
            ("merge085", pp(merge_similar=True, merge_similar_min_sim=0.85)),
            ("merge090", pp(merge_similar=True, merge_similar_min_sim=0.90)),
            ("rg_s2_base", pp(**rg_s2)),
            ("rg_s2_merge080", pp(**rg_s2, merge_similar=True, merge_similar_min_sim=0.80)),
            ("rg_s2_merge085", pp(**rg_s2, merge_similar=True, merge_similar_min_sim=0.85)),
            ("rg_s2_merge090", pp(**rg_s2, merge_similar=True, merge_similar_min_sim=0.90)),
            ("rg_s2_merge_off", pp(**rg_s2, merge_similar=False)),
        ]
    if name == "claude_regroup":
        rg = dict(
            residual_regroup=True,
            residual_regroup_clf_threshold=0.015,
            residual_regroup_max_cluster_size=3,
            residual_regroup_topk=32,
            residual_regroup_seed_sim=0.50,
            residual_regroup_pair_sim=0.10,
            residual_regroup_min_group_size=3,
            residual_regroup_max_group_size=4,
            residual_regroup_require_unique_layer=True,
            residual_regroup_max_hits=0,
            pad_size2=True,
            pad_size2_clf_floor=0.0,
            pad_size2_clf_ceiling=0.010,
        )
        return [
            ("current", current),
            (
                "clear_pad03_ce10",
                pp(
                    **rg,
                    residual_regroup_clear_pool=True,
                    residual_regroup_pair_repair=False,
                    pad_size2_min_sim=0.3,
                ),
            ),
            (
                "keep_pad03_ce10",
                pp(
                    **rg,
                    residual_regroup_clear_pool=False,
                    residual_regroup_pair_repair=False,
                    pad_size2_min_sim=0.3,
                ),
            ),
            (
                "clear_pad02_ce10",
                pp(
                    **rg,
                    residual_regroup_clear_pool=True,
                    residual_regroup_pair_repair=False,
                    pad_size2_min_sim=0.2,
                ),
            ),
            (
                "keep_pad02_ce10",
                pp(
                    **rg,
                    residual_regroup_clear_pool=False,
                    residual_regroup_pair_repair=False,
                    pad_size2_min_sim=0.2,
                ),
            ),
            (
                "clear_pair_pad03_ce10",
                pp(
                    **rg,
                    residual_regroup_clear_pool=True,
                    residual_regroup_pair_repair=True,
                    residual_regroup_pair_repair_sim=0.30,
                    residual_regroup_pair_repair_topk=16,
                    pad_size2_min_sim=0.3,
                ),
            ),
            (
                "keep_pair_pad03_ce10",
                pp(
                    **rg,
                    residual_regroup_clear_pool=False,
                    residual_regroup_pair_repair=True,
                    residual_regroup_pair_repair_sim=0.30,
                    residual_regroup_pair_repair_topk=16,
                    pad_size2_min_sim=0.3,
                ),
            ),
        ]
    if name == "claude_fast":
        rg = dict(
            residual_regroup=True,
            residual_regroup_clf_threshold=0.015,
            residual_regroup_max_cluster_size=3,
            residual_regroup_topk=32,
            residual_regroup_seed_sim=0.50,
            residual_regroup_pair_sim=0.10,
            residual_regroup_min_group_size=3,
            residual_regroup_max_group_size=4,
            residual_regroup_require_unique_layer=True,
            residual_regroup_pair_repair=False,
            pad_size2=True,
            pad_size2_clf_floor=0.0,
            pad_size2_clf_ceiling=0.010,
        )
        return [
            ("current", current),
            (
                "clear_cap2k_pad03",
                pp(
                    **rg,
                    residual_regroup_clear_pool=True,
                    residual_regroup_max_hits=2048,
                    pad_size2_min_sim=0.3,
                ),
            ),
            (
                "clear_cap4k_pad03",
                pp(
                    **rg,
                    residual_regroup_clear_pool=True,
                    residual_regroup_max_hits=4096,
                    pad_size2_min_sim=0.3,
                ),
            ),
            (
                "keep_cap2k_pad03",
                pp(
                    **rg,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                    pad_size2_min_sim=0.3,
                ),
            ),
            (
                "keep_cap4k_pad03",
                pp(
                    **rg,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=4096,
                    pad_size2_min_sim=0.3,
                ),
            ),
            (
                "clear_cap2k_pad02",
                pp(
                    **rg,
                    residual_regroup_clear_pool=True,
                    residual_regroup_max_hits=2048,
                    pad_size2_min_sim=0.2,
                ),
            ),
            (
                "keep_cap2k_pad02",
                pp(
                    **rg,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                    pad_size2_min_sim=0.2,
                ),
            ),
            (
                "keep_cap2k_pad04",
                pp(
                    **rg,
                    residual_regroup_clear_pool=False,
                    residual_regroup_max_hits=2048,
                    pad_size2_min_sim=0.4,
                ),
            ),
        ]
    if name == "budget20":
        return [
            (
                "base_9801_like",
                pp(
                    stitch_overlap=False,
                    merge_similar=False,
                    pad_size2=False,
                    residual_regroup=False,
                    dedup_same_layer=False,
                    split_overmerged=False,
                ),
            ),
            (
                "merge_only",
                pp(
                    stitch_overlap=False,
                    merge_similar=True,
                    merge_similar_min_sim=0.75,
                    merge_similar_small_cluster_size=2,
                    merge_similar_max_merged_size=12,
                    merge_similar_topk=4,
                    merge_similar_require_unique_layer=False,
                    pad_size2=False,
                    residual_regroup=False,
                ),
            ),
            (
                "stitch_only",
                pp(
                    stitch_overlap=True,
                    stitch_score_gate=0.005,
                    stitch_min_overlap=1,
                    stitch_min_overlap_frac=0.6,
                    merge_similar=False,
                    pad_size2=False,
                    residual_regroup=False,
                ),
            ),
            (
                "stitch_merge075",
                pp(
                    stitch_overlap=True,
                    stitch_score_gate=0.005,
                    stitch_min_overlap=1,
                    stitch_min_overlap_frac=0.6,
                    merge_similar=True,
                    merge_similar_min_sim=0.75,
                    merge_similar_small_cluster_size=2,
                    merge_similar_max_merged_size=12,
                    merge_similar_topk=4,
                    merge_similar_require_unique_layer=False,
                    pad_size2=False,
                    residual_regroup=False,
                ),
            ),
            (
                "stitch_merge080",
                pp(
                    stitch_overlap=True,
                    stitch_score_gate=0.005,
                    stitch_min_overlap=1,
                    stitch_min_overlap_frac=0.6,
                    merge_similar=True,
                    merge_similar_min_sim=0.80,
                    merge_similar_small_cluster_size=2,
                    merge_similar_max_merged_size=12,
                    merge_similar_topk=4,
                    merge_similar_require_unique_layer=False,
                    pad_size2=False,
                    residual_regroup=False,
                ),
            ),
        ]
    if name == "stitch_merge_grid":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            split_overmerged=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        return [
            sm("decoder_only", stitch_overlap=False, merge_similar=False),
            sm("merge_m075", stitch_overlap=False),
            sm("stitch_only", merge_similar=False),
            sm("sm075_base"),
            sm("sg002_f06_m075", stitch_score_gate=0.002),
            sm("sg010_f06_m075", stitch_score_gate=0.010),
            sm("sg005_f05_m075", stitch_min_overlap_frac=0.5),
            sm("sg005_f07_m075", stitch_min_overlap_frac=0.7),
            sm("sg002_f05_m075", stitch_score_gate=0.002, stitch_min_overlap_frac=0.5),
            sm("sg010_f07_m075", stitch_score_gate=0.010, stitch_min_overlap_frac=0.7),
            sm("minov2_m075", stitch_min_overlap=2),
            sm("sm070", merge_similar_min_sim=0.70),
            sm("sm072", merge_similar_min_sim=0.72),
            sm("sm078", merge_similar_min_sim=0.78),
            sm("sm080", merge_similar_min_sim=0.80),
            sm("small1_m075", merge_similar_small_cluster_size=1),
            sm("small3_m075", merge_similar_small_cluster_size=3),
            sm("max8_m075", merge_similar_max_merged_size=8),
            sm("max10_m075", merge_similar_max_merged_size=10),
            sm("max16_m075", merge_similar_max_merged_size=16),
            sm("top2_m075", merge_similar_topk=2),
            sm("top8_m075", merge_similar_topk=8),
            sm("unique_layer", merge_similar_require_unique_layer=True),
        ]
    if name == "active_score_grid":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.0,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=1.0,
            decoder_active_prob_floor=0.0,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            split_overmerged=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        return [
            sm("base_s0_a0"),
            sm("score001", decoder_score_threshold=0.001),
            sm("score002", decoder_score_threshold=0.002),
            sm("score005", decoder_score_threshold=0.005),
            sm("score010", decoder_score_threshold=0.010),
            sm("score020", decoder_score_threshold=0.020),
            sm("score050", decoder_score_threshold=0.050),
            sm("score100", decoder_score_threshold=0.100),
            sm("active005", decoder_active_prob_threshold=0.05),
            sm("active010", decoder_active_prob_threshold=0.10),
            sm("active020", decoder_active_prob_threshold=0.20),
            sm("active030", decoder_active_prob_threshold=0.30),
            sm("active050", decoder_active_prob_threshold=0.50),
            sm("active070", decoder_active_prob_threshold=0.70),
            sm("s002_a010", decoder_score_threshold=0.002, decoder_active_prob_threshold=0.10),
            sm("s005_a010", decoder_score_threshold=0.005, decoder_active_prob_threshold=0.10),
            sm("s010_a010", decoder_score_threshold=0.010, decoder_active_prob_threshold=0.10),
            sm("s005_a020", decoder_score_threshold=0.005, decoder_active_prob_threshold=0.20),
            sm("s010_a020", decoder_score_threshold=0.010, decoder_active_prob_threshold=0.20),
        ]
    if name == "active_maxprob_grid":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.0,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=1.0,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            cluster_score_filter_max_size=3,
            cluster_score_filter_mean_threshold=0.0,
            cluster_score_filter_min_threshold=0.0,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            split_overmerged=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        return [
            sm("base_ap1_s0"),
            sm("ap0_s0", decoder_active_prob_power=0.0),
            sm("ap025_s0", decoder_active_prob_power=0.25),
            sm("ap05_s0", decoder_active_prob_power=0.5),
            sm("ap075_s0", decoder_active_prob_power=0.75),
            sm("ap1_s0005", decoder_score_threshold=0.0005),
            sm("ap1_s001", decoder_score_threshold=0.001),
            sm("ap025_s001", decoder_active_prob_power=0.25, decoder_score_threshold=0.001),
            sm("ap05_s001", decoder_active_prob_power=0.5, decoder_score_threshold=0.001),
            sm("ap075_s001", decoder_active_prob_power=0.75, decoder_score_threshold=0.001),
            sm("ap0_s001", decoder_active_prob_power=0.0, decoder_score_threshold=0.001),
            sm("ap1_floor025_s001", decoder_active_prob_floor=0.25, decoder_score_threshold=0.001),
            sm("ap1_floor050_s001", decoder_active_prob_floor=0.50, decoder_score_threshold=0.001),
            sm("ap1_floor075_s001", decoder_active_prob_floor=0.75, decoder_score_threshold=0.001),
            sm(
                "cf3_mean0005",
                cluster_score_filter=True,
                cluster_score_filter_max_size=3,
                cluster_score_filter_mean_threshold=0.0005,
            ),
            sm(
                "cf3_mean001",
                cluster_score_filter=True,
                cluster_score_filter_max_size=3,
                cluster_score_filter_mean_threshold=0.001,
            ),
            sm(
                "cf4_mean001",
                cluster_score_filter=True,
                cluster_score_filter_max_size=4,
                cluster_score_filter_mean_threshold=0.001,
            ),
            sm(
                "s001_cf3_mean0005",
                decoder_score_threshold=0.001,
                cluster_score_filter=True,
                cluster_score_filter_max_size=3,
                cluster_score_filter_mean_threshold=0.0005,
            ),
            sm(
                "s001_cf3_mean001",
                decoder_score_threshold=0.001,
                cluster_score_filter=True,
                cluster_score_filter_max_size=3,
                cluster_score_filter_mean_threshold=0.001,
            ),
            sm(
                "s001_cf4_mean001",
                decoder_score_threshold=0.001,
                cluster_score_filter=True,
                cluster_score_filter_max_size=4,
                cluster_score_filter_mean_threshold=0.001,
            ),
            sm(
                "s001_dedup_score",
                decoder_score_threshold=0.001,
                dedup_same_layer_by_score=True,
                dedup_same_layer_score_min_cluster_size=4,
                dedup_same_layer_score_min_size_after=3,
                dedup_same_layer_score_min_gap=0.0,
            ),
            sm(
                "s001_dedup_score_gap0005",
                decoder_score_threshold=0.001,
                dedup_same_layer_by_score=True,
                dedup_same_layer_score_min_cluster_size=4,
                dedup_same_layer_score_min_size_after=3,
                dedup_same_layer_score_min_gap=0.0005,
            ),
            sm(
                "s001_cf3_dedup",
                decoder_score_threshold=0.001,
                cluster_score_filter=True,
                cluster_score_filter_max_size=3,
                cluster_score_filter_mean_threshold=0.0005,
                dedup_same_layer_by_score=True,
                dedup_same_layer_score_min_cluster_size=4,
                dedup_same_layer_score_min_size_after=3,
                dedup_same_layer_score_min_gap=0.0,
            ),
        ]
    if name == "active_power_fine":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.0,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=1.0,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            split_overmerged=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        variants = [
            sm("base_ap1_s0"),
            sm("base_ap1_s001", decoder_score_threshold=0.001),
        ]
        for power in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.10]:
            tag = str(power).replace(".", "")
            variants.append(sm(f"ap{tag}_s0", decoder_active_prob_power=power))
            variants.append(
                sm(
                    f"ap{tag}_s002",
                    decoder_active_prob_power=power,
                    decoder_score_threshold=0.002,
                )
            )
            variants.append(
                sm(
                    f"ap{tag}_s005",
                    decoder_active_prob_power=power,
                    decoder_score_threshold=0.005,
                )
            )
        for floor in [0.10, 0.20, 0.30, 0.40, 0.60]:
            tag = str(floor).replace(".", "")
            variants.append(
                sm(
                    f"floor{tag}_s002",
                    decoder_active_prob_floor=floor,
                    decoder_score_threshold=0.002,
                )
            )
        return variants
    if name == "shifted_topk":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.001,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            decoder_active_query_topk=0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_topk_score_gate=None,
            stitch_topk_assign=False,
            stitch_topk_assign_score_gate=0.0,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            split_overmerged=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        return [
            sm("base_ap075_s001"),
            sm("top2_g005", stitch_query_topk=2, stitch_topk_score_gate=0.005),
            sm("top2_g010", stitch_query_topk=2, stitch_topk_score_gate=0.010),
            sm("top2_g020", stitch_query_topk=2, stitch_topk_score_gate=0.020),
            sm("top2_g050", stitch_query_topk=2, stitch_topk_score_gate=0.050),
            sm("top2_minov2_g005", stitch_query_topk=2, stitch_topk_score_gate=0.005, stitch_min_overlap=2),
            sm("top2_frac07_g010", stitch_query_topk=2, stitch_topk_score_gate=0.010, stitch_min_overlap_frac=0.7),
            sm("top3_g010", stitch_query_topk=3, stitch_topk_score_gate=0.010),
            sm("top3_g020", stitch_query_topk=3, stitch_topk_score_gate=0.020),
            sm("top2_assign_g010", stitch_query_topk=2, stitch_topk_score_gate=0.010, stitch_topk_assign=True),
            sm(
                "top2_assign_g010_a001",
                stitch_query_topk=2,
                stitch_topk_score_gate=0.010,
                stitch_topk_assign=True,
                stitch_topk_assign_score_gate=0.001,
            ),
            sm(
                "top2_assign_g020_a005",
                stitch_query_topk=2,
                stitch_topk_score_gate=0.020,
                stitch_topk_assign=True,
                stitch_topk_assign_score_gate=0.005,
            ),
            sm(
                "ap025_top2_g005",
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                stitch_query_topk=2,
                stitch_topk_score_gate=0.005,
            ),
            sm(
                "ap025_top2_g010",
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                stitch_query_topk=2,
                stitch_topk_score_gate=0.010,
            ),
            sm(
                "ap025_top2_max8",
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                stitch_query_topk=2,
                stitch_topk_score_gate=0.010,
                merge_similar_max_merged_size=8,
            ),
        ]
    if name == "topk_restore":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.001,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            stitch_topk_score_gate=None,
            stitch_topk_assign=False,
            stitch_topk_assign_score_gate=0.0,
            restore_topk_components=False,
            restore_topk_min_score=0.01,
            restore_topk_max_conflict_size=0,
            restore_topk_min_target_size=3,
            restore_topk_unique_layer=True,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            split_overmerged=False,
            split_window_clusters=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        restore = dict(
            stitch_query_topk=2,
            stitch_union_topk=False,
            stitch_topk_score_gate=0.005,
            restore_topk_components=True,
        )
        return [
            sm("base_ap075_s001"),
            sm("top2_restore_s005_u0", **restore, restore_topk_min_score=0.005, restore_topk_max_conflict_size=0),
            sm("top2_restore_s010_u0", **restore, restore_topk_min_score=0.010, restore_topk_max_conflict_size=0),
            sm("top2_restore_s020_u0", **restore, restore_topk_min_score=0.020, restore_topk_max_conflict_size=0),
            sm("top2_restore_s005_u2", **restore, restore_topk_min_score=0.005, restore_topk_max_conflict_size=2),
            sm("top2_restore_s010_u2", **restore, restore_topk_min_score=0.010, restore_topk_max_conflict_size=2),
            sm("top2_restore_t2_s010", **restore, restore_topk_min_score=0.010, restore_topk_min_target_size=2),
            sm("top2_restore_nolayer", **restore, restore_topk_min_score=0.010, restore_topk_unique_layer=False),
            sm(
                "ap025_top2_restore",
                **restore,
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                restore_topk_min_score=0.010,
            ),
        ]
    if name == "kmeans_grid":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.001,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            split_window_clusters=False,
            split_overmerged=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        return [
            sm("base_k0"),
            sm("kmeans1", kmeans_iters=1),
            sm("kmeans1_m001", kmeans_iters=1, kmeans_min_margin=0.001),
            sm("kmeans1_m005", kmeans_iters=1, kmeans_min_margin=0.005),
            sm("kmeans1_m010", kmeans_iters=1, kmeans_min_margin=0.010),
            sm("kmeans2", kmeans_iters=2),
            sm("kmeans3", kmeans_iters=3),
            sm("ap025_k1", kmeans_iters=1, decoder_active_prob_power=0.25, decoder_score_threshold=0.0),
        ]
    if name == "kmeans_tune":
        base = dict(
            kmeans_iters=1,
            clf_recall=1.0,
            decoder_score_threshold=0.001,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            restore_window_clusters=False,
            restore_window_min_score=0.005,
            restore_window_max_conflict_size=2,
            restore_window_min_new_frac=0.6,
            bridge_window_clusters=False,
            bridge_window_min_score=0.005,
            bridge_window_min_candidate_size=3,
            bridge_window_min_label_overlap=1,
            bridge_window_max_cluster_size=2,
            bridge_window_max_merged_size=8,
            pad_window_size2=False,
            split_window_clusters=False,
            split_overmerged=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        restore = dict(restore_window_clusters=True)
        bridge = dict(bridge_window_clusters=True)
        return [
            sm("k1_base"),
            sm("k1_s0", decoder_score_threshold=0.0),
            sm("k1_s002", decoder_score_threshold=0.002),
            sm("k1_s002_m001", decoder_score_threshold=0.002, kmeans_min_margin=0.001),
            sm("k1_s002_m005", decoder_score_threshold=0.002, kmeans_min_margin=0.005),
            sm("k1_s002_m010", decoder_score_threshold=0.002, kmeans_min_margin=0.010),
            sm("k1_s002_m020", decoder_score_threshold=0.002, kmeans_min_margin=0.020),
            sm("k1_s005", decoder_score_threshold=0.005),
            sm("k1_ap05_s001", decoder_active_prob_power=0.5),
            sm("k1_ap1_s001", decoder_active_prob_power=1.0),
            sm("k1_merge070", merge_similar_min_sim=0.70),
            sm("k1_merge080", merge_similar_min_sim=0.80),
            sm("k1_merge_max8", merge_similar_max_merged_size=8),
            sm("k1_restore_s005", **restore, restore_window_min_score=0.005),
            sm("k1_restore_s010", **restore, restore_window_min_score=0.010),
            sm("k1_bridge_s005", **bridge, bridge_window_min_score=0.005),
            sm("k1_bridge_s010", **bridge, bridge_window_min_score=0.010),
            sm("k1_bridge_restore", **bridge, **restore, bridge_window_min_score=0.010, restore_window_min_score=0.005),
        ]
    if name == "kmeans_merge_stitch_fine":
        base = dict(
            pre_kmeans_iters=0,
            pre_kmeans_min_margin=0.0,
            kmeans_iters=1,
            kmeans_min_margin=0.005,
            clf_recall=1.0,
            decoder_score_threshold=0.002,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            decoder_use_logit_scores=False,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            restore_topk_components=False,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            split_window_clusters=False,
            split_overmerged=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            prune_small_clusters=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        return [
            sm("base"),
            sm("m068", merge_similar_min_sim=0.68),
            sm("m070", merge_similar_min_sim=0.70),
            sm("m072", merge_similar_min_sim=0.72),
            sm("m074", merge_similar_min_sim=0.74),
            sm("m076", merge_similar_min_sim=0.76),
            sm("m078", merge_similar_min_sim=0.78),
            sm("m080", merge_similar_min_sim=0.80),
            sm("small3_m075", merge_similar_small_cluster_size=3),
            sm("small3_m080", merge_similar_small_cluster_size=3, merge_similar_min_sim=0.80),
            sm("small3_m085", merge_similar_small_cluster_size=3, merge_similar_min_sim=0.85),
            sm("small4_m085", merge_similar_small_cluster_size=4, merge_similar_min_sim=0.85),
            sm("max8", merge_similar_max_merged_size=8),
            sm("max10", merge_similar_max_merged_size=10),
            sm("max16", merge_similar_max_merged_size=16),
            sm("top2", merge_similar_topk=2),
            sm("top8", merge_similar_topk=8),
            sm("uniq_m070", merge_similar_min_sim=0.70, merge_similar_require_unique_layer=True),
            sm("uniq_m075", merge_similar_require_unique_layer=True),
            sm("uniq_small3_m075", merge_similar_small_cluster_size=3, merge_similar_require_unique_layer=True),
            sm("sg002", stitch_score_gate=0.002),
            sm("sg010", stitch_score_gate=0.010),
            sm("frac05", stitch_min_overlap_frac=0.5),
            sm("frac07", stitch_min_overlap_frac=0.7),
            sm("minov2", stitch_min_overlap=2),
            sm("sg002_m072", stitch_score_gate=0.002, merge_similar_min_sim=0.72),
            sm("sg002_m078", stitch_score_gate=0.002, merge_similar_min_sim=0.78),
            sm("sg010_m072", stitch_score_gate=0.010, merge_similar_min_sim=0.72),
        ]
    if name == "kmeans_order":
        base = dict(
            pre_kmeans_iters=0,
            pre_kmeans_min_margin=0.0,
            kmeans_iters=1,
            kmeans_min_margin=0.005,
            clf_recall=1.0,
            decoder_score_threshold=0.002,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            restore_topk_components=False,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            split_window_clusters=False,
            split_overmerged=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            prune_small_clusters=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        return [
            sm("post_k1"),
            sm("post_k1_sg010", stitch_score_gate=0.010),
            sm("pre_k1_post0", pre_kmeans_iters=1, pre_kmeans_min_margin=0.005, kmeans_iters=0),
            sm("pre_k1_post0_m0", pre_kmeans_iters=1, pre_kmeans_min_margin=0.0, kmeans_iters=0),
            sm("pre_k1_post1", pre_kmeans_iters=1, pre_kmeans_min_margin=0.005),
            sm("pre_k1_post1_sg010", pre_kmeans_iters=1, pre_kmeans_min_margin=0.005, stitch_score_gate=0.010),
            sm("pre_k1_post1_max8", pre_kmeans_iters=1, pre_kmeans_min_margin=0.005, merge_similar_max_merged_size=8),
            sm("pre_k1_post1_m078", pre_kmeans_iters=1, pre_kmeans_min_margin=0.005, merge_similar_min_sim=0.78),
            sm("pre_k1_post1_m072", pre_kmeans_iters=1, pre_kmeans_min_margin=0.005, merge_similar_min_sim=0.72),
            sm("pre_k1_post1_s002", pre_kmeans_iters=1, pre_kmeans_min_margin=0.005, decoder_score_threshold=0.001),
            sm("pre_k2_post0", pre_kmeans_iters=2, pre_kmeans_min_margin=0.005, kmeans_iters=0),
            sm("pre_k2_post1", pre_kmeans_iters=2, pre_kmeans_min_margin=0.005),
        ]
    if name == "kmeans_residual_tiny":
        base = dict(
            pre_kmeans_iters=0,
            pre_kmeans_min_margin=0.0,
            kmeans_iters=1,
            clf_recall=1.0,
            decoder_score_threshold=0.002,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            residual_regroup_clf_threshold=0.015,
            residual_regroup_max_cluster_size=3,
            residual_regroup_topk=8,
            residual_regroup_seed_sim=0.55,
            residual_regroup_pair_sim=0.15,
            residual_regroup_min_group_size=3,
            residual_regroup_max_group_size=4,
            residual_regroup_require_unique_layer=True,
            residual_regroup_clear_pool=False,
            residual_regroup_pair_repair=False,
            residual_regroup_max_hits=512,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            split_window_clusters=False,
            split_overmerged=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        rg = dict(residual_regroup=True)
        return [
            sm("k1_s002_base"),
            sm("rg256_t8_s55", **rg, residual_regroup_max_hits=256, residual_regroup_topk=8, residual_regroup_seed_sim=0.55),
            sm("rg512_t8_s55", **rg, residual_regroup_max_hits=512, residual_regroup_topk=8, residual_regroup_seed_sim=0.55),
            sm("rg512_t8_s60", **rg, residual_regroup_max_hits=512, residual_regroup_topk=8, residual_regroup_seed_sim=0.60, residual_regroup_pair_sim=0.20),
            sm("rg1024_t8_s55", **rg, residual_regroup_max_hits=1024, residual_regroup_topk=8, residual_regroup_seed_sim=0.55),
            sm("rg64_s2_t8", **rg, residual_regroup_max_hits=64, residual_regroup_max_cluster_size=2, residual_regroup_topk=8),
            sm("rg128_s2_t8", **rg, residual_regroup_max_hits=128, residual_regroup_max_cluster_size=2, residual_regroup_topk=8),
            sm("rg256_s2_t8", **rg, residual_regroup_max_hits=256, residual_regroup_max_cluster_size=2, residual_regroup_topk=8),
            sm("rg512_s2_t8", **rg, residual_regroup_max_hits=512, residual_regroup_max_cluster_size=2, residual_regroup_topk=8),
            sm("rg1024_s2_t8", **rg, residual_regroup_max_hits=1024, residual_regroup_max_cluster_size=2, residual_regroup_topk=8),
            sm("rg512_t12", **rg, residual_regroup_max_hits=512, residual_regroup_topk=12),
        ]
    if name == "latency30":
        base = dict(
            pre_kmeans_iters=0,
            pre_kmeans_min_margin=0.0,
            kmeans_iters=1,
            kmeans_min_margin=0.005,
            kmeans_use_fp16_sim=False,
            kmeans_max_refine_cluster_size=0,
            kmeans_refine_max_score=0.0,
            clf_recall=1.0,
            decoder_score_threshold=0.002,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.010,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            stitch_topk_assign=False,
            merge_similar=True,
            merge_similar_direct=False,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            merge_similar_direct_target_min_size=0,
            pad_size2=False,
            residual_regroup=False,
            restore_topk_components=False,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            split_window_clusters=False,
            split_overmerged=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            prune_small_clusters=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        return [
            sm("k1_merge"),
            sm("k1_nomerge", merge_similar=False),
            sm("k1_direct_merge", merge_similar_direct=True),
            sm("k1_direct_merge_m070", merge_similar_direct=True, merge_similar_min_sim=0.70),
            sm("k1_direct_merge_m080", merge_similar_direct=True, merge_similar_min_sim=0.80),
            sm("k1_direct_merge_s1", merge_similar_direct=True, merge_similar_small_cluster_size=1),
            sm("k1_direct_merge_s3", merge_similar_direct=True, merge_similar_small_cluster_size=3, merge_similar_direct_target_min_size=4),
            sm(
                "k1_logit_nomerge",
                merge_similar=False,
                decoder_use_logit_scores=True,
                decoder_active_prob_power=0.0,
                decoder_score_threshold=-6.2,
                stitch_score_gate=-4.6,
            ),
            sm(
                "k1_logit_merge_top2",
                decoder_use_logit_scores=True,
                decoder_active_prob_power=0.0,
                decoder_score_threshold=-6.2,
                stitch_score_gate=-4.6,
                merge_similar_topk=2,
            ),
            sm(
                "k1_logit_direct",
                merge_similar_direct=True,
                decoder_use_logit_scores=True,
                decoder_active_prob_power=0.0,
                decoder_score_threshold=-6.2,
                stitch_score_gate=-4.6,
            ),
            sm("k1_ref2_merge", kmeans_max_refine_cluster_size=2),
            sm("k1_ref3_merge", kmeans_max_refine_cluster_size=3),
            sm("k1_ref5_merge", kmeans_max_refine_cluster_size=5),
            sm("k1_ref8_merge", kmeans_max_refine_cluster_size=8),
            sm("k1_ref3_nomerge", merge_similar=False, kmeans_max_refine_cluster_size=3),
            sm("k1_ref5_nomerge", merge_similar=False, kmeans_max_refine_cluster_size=5),
            sm("k1_ref3_nm_s000", merge_similar=False, kmeans_max_refine_cluster_size=3, decoder_score_threshold=0.0),
            sm("k1_ref3_nm_s001", merge_similar=False, kmeans_max_refine_cluster_size=3, decoder_score_threshold=0.001),
            sm("k1_ref3_nm_s003", merge_similar=False, kmeans_max_refine_cluster_size=3, decoder_score_threshold=0.003),
            sm("k1_ref3_nm_sg002", merge_similar=False, kmeans_max_refine_cluster_size=3, stitch_score_gate=0.002),
            sm("k1_ref3_nm_sg005", merge_similar=False, kmeans_max_refine_cluster_size=3, stitch_score_gate=0.005),
            sm("k1_ref3_nm_sg015", merge_similar=False, kmeans_max_refine_cluster_size=3, stitch_score_gate=0.015),
            sm("k1_ref3_nm_ap050", merge_similar=False, kmeans_max_refine_cluster_size=3, decoder_active_prob_power=0.50),
            sm("k1_ref3_nm_ap100", merge_similar=False, kmeans_max_refine_cluster_size=3, decoder_active_prob_power=1.00),
            sm("k1_ref3_nm_ap125", merge_similar=False, kmeans_max_refine_cluster_size=3, decoder_active_prob_power=1.25),
            sm("k1_ref3_nm_m000", merge_similar=False, kmeans_max_refine_cluster_size=3, kmeans_min_margin=0.0),
            sm("k1_ref3_nm_m010", merge_similar=False, kmeans_max_refine_cluster_size=3, kmeans_min_margin=0.010),
            sm("k1_merge_top2", merge_similar_topk=2),
            sm("k1_merge_top1", merge_similar_topk=1),
            sm("k1_merge_m080", merge_similar_min_sim=0.80),
            sm("k1_merge_max8", merge_similar_max_merged_size=8),
            sm("k1_direct_merge_t1", merge_similar_direct=True, merge_similar_direct_target_min_size=1),
            sm("k1_direct_merge_t2", merge_similar_direct=True, merge_similar_direct_target_min_size=2),
            sm("k1_direct_merge_t1_m070", merge_similar_direct=True, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.70),
            sm("k1_direct_merge_t2_m070", merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70),
            sm("k1_direct_t1_score005", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.005),
            sm("k1_direct_t1_score010", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.010),
            sm("k1_direct_t1_score020", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.020),
            sm("k1_direct_t1_score050", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.050),
            sm("k1_direct_t1_score100", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.100),
            sm("k1_direct_t1_score150", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.150),
            sm("k1_direct_t1_score200", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.200),
            sm("k1_direct_t1_score250", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.250),
            sm("k1_direct_t1_score300", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.300),
            sm("k1_direct_t1_score200_m000", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.200, kmeans_min_margin=0.0),
            sm("k1_direct_t1_score200_m010", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_refine_max_score=0.200, kmeans_min_margin=0.010),
            sm("k1_direct_t1_m070_score200", merge_similar_direct=True, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.70, kmeans_refine_max_score=0.200),
            sm("k1_direct_s1_t1_m070", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.70),
            sm("k1_direct_s1_t2_m070", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70),
            sm("k1_direct_s1_t2_m070_ap085", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=0.85),
            sm("k1_direct_s1_t2_m070_ap090", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=0.90),
            sm("k1_direct_s1_t2_m070_ap095", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=0.95),
            sm("k1_direct_s1_t2_m070_ap060", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=0.60),
            sm("k1_direct_s1_t2_m070_ap065", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=0.65),
            sm("k1_direct_s1_t2_m070_ap070", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=0.70),
            sm("k1_direct_s1_t2_m070_ap080", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=0.80),
            sm("k1_direct_s1_t2_m065", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65),
            sm("k1_direct_s1_t2_m066", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.66),
            sm("k1_direct_s1_t2_m067", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.67),
            sm("k1_direct_s1_t2_m068", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68),
            sm("k1_direct_s1_t2_m069", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.69),
            sm("k1_direct_s1_t2_m065_s003", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003),
            sm("k1_direct_s1_t2_m065_s003_sg020", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, stitch_score_gate=0.020),
            sm("k1_direct_s1_t2_m065_s003_sg030", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, stitch_score_gate=0.030),
            sm("k1_direct_s1_t2_m065_s003_sg050", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, stitch_score_gate=0.050),
            sm("k1_direct_s2_t2_m068_s003", merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003),
            sm("k1_direct_s2_t2_m068_s003_sg020", merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003, stitch_score_gate=0.020),
            sm("k1_direct_s2_t2_m068_s003_sg030", merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003, stitch_score_gate=0.030),
            sm("k1_direct_s2_t2_m068_s003_sg050", merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003, stitch_score_gate=0.050),
            sm("nostitch_prob_k1_direct_s1_t2_m065_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s1_t2_m060_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.60, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s1_t2_m055_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.55, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s1_t2_m050_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.50, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s1_t2_m045_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.45, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s1_t2_m060_s001", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.60, decoder_score_threshold=0.001),
            sm("nostitch_prob_k1_direct_s1_t2_m055_s001", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.55, decoder_score_threshold=0.001),
            sm("nostitch_prob_k1_direct_s1_t1_m060_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.60, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s1_t1_m055_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.55, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s2_t2_m060_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.60, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s2_t2_m065_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s2_t2_m066_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.66, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s2_t2_m067_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.67, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s2_t2_m068_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s2_t2_m070_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.003),
            sm("fast5_assign_only_s003", stitch_overlap=False, merge_similar=False, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_assign_k1_s003", stitch_overlap=False, merge_similar=False, kmeans_iters=1, decoder_score_threshold=0.003),
            sm("fast5_assign_k0_pad32_s003", stitch_overlap=False, merge_similar=False, kmeans_iters=0, decoder_score_threshold=0.003, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_assign_k0_pad64_s003", stitch_overlap=False, merge_similar=False, kmeans_iters=0, decoder_score_threshold=0.003, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_assign_logprob_pad64_l581_ap060", stitch_overlap=False, merge_similar=False, kmeans_iters=0, decoder_score_mode="logprob", decoder_score_threshold=-5.809143, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_assign_logprob_pad128_l581_ap060", stitch_overlap=False, merge_similar=False, kmeans_iters=0, decoder_score_mode="logprob", decoder_score_threshold=-5.809143, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=128, pad_size2_topk=1, pad_size2_max_candidates=1024),
            sm("fast5_direct_k0_s2_t2_m065_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_direct_k0_s2_t2_m068_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_direct_k0_s2_t2_m070_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_label_k0_s2_t2_m065_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_label_k0_s2_t2_m068_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_label_k0_s2_t2_m070_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_label_k0_s2_t2_m068_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_logprob_k0_s2_t2_m068_l581_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="logprob", decoder_score_threshold=-5.809143, decoder_active_prob_power=0.60),
            sm("fast5_logprob_direct_k0_s2_t2_m068_l581_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="logprob", decoder_score_threshold=-5.809143, decoder_active_prob_power=0.60),
            sm("fast5_logprob_k0_s2_t2_m068_l581_ap075", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="logprob", decoder_score_threshold=-5.809143, decoder_active_prob_power=0.75),
            sm("fast5_logprob_k0_s2_t2_m070_l581_ap075", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_mode="logprob", decoder_score_threshold=-5.809143, decoder_active_prob_power=0.75),
            sm("fast5_clamp_k0_s2_t2_m068_s58_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.60),
            sm("fast5_clamp_k0_s2_t2_m068_s58_ap075", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.75),
            sm("fast5_clamp_k0_s2_t2_m068_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t2_m065_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t2_m066_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.66, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t2_m067_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.67, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t2_m070_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s56_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.6, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s58_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s62_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-6.2, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap070", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.70),
            sm("fast5_clamp_direct_k0_s2_t2_m060_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.60, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m063_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.63, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m064_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.64, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m065_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m066_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.66, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m067_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.67, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m069_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.69, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s58_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m067_s58_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.67, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m066_s58_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.66, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s52_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.2, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s50_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.0, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap050_pad32", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap060_pad32", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_clamp_direct_k0_s2_t2_m070_s54_ap060_pad32", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap050_pad16_u", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50, pad_size2=True, pad_size2_max_conflict_size=0, pad_size2_min_sim=0.4, pad_size2_max_clusters=16, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap050_pad32_u", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50, pad_size2=True, pad_size2_max_conflict_size=0, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap060_pad16_u", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=0, pad_size2_min_sim=0.4, pad_size2_max_clusters=16, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap060_pad32_u", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=0, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_clamp_direct_k0_s2_t2_m070_s54_ap060_pad16_u", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=0, pad_size2_min_sim=0.4, pad_size2_max_clusters=16, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_clamp_direct_k0_s2_t2_m070_s54_ap060_pad32_u", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=0, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap060_pad16_rc0", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=16, pad_size2_topk=1, pad_size2_max_candidates=512, pad_size2_rank_clusters_by_clf=False),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap060_pad32_rc0", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512, pad_size2_rank_clusters_by_clf=False),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap060_pad32_rc0_rh0", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512, pad_size2_rank_clusters_by_clf=False, pad_size2_rank_candidates_by_clf=False),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap060_pad16_rc0_rh0", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=16, pad_size2_topk=1, pad_size2_max_candidates=512, pad_size2_rank_clusters_by_clf=False, pad_size2_rank_candidates_by_clf=False),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap050_geom64", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.3, pad_size2_max_clusters=64, pad_size2_topk=1, pad_size2_max_candidates=1024, pad_size2_geom_z_max=0.005, pad_size2_geom_phi_max=0.04),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap050_max10", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, merge_similar_max_merged_size=10, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap050_max16", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, merge_similar_max_merged_size=16, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap050_max20", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, merge_similar_max_merged_size=20, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t2_m068_s54_ap060_max16", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, merge_similar_max_merged_size=16, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t2_m070_s54_ap060_max16", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, merge_similar_max_merged_size=16, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m066_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.66, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m067_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.67, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m069_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.69, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m070_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m071_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.71, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s50_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.0, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s52_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.2, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s56_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.6, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s58_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s54_ap045", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.45),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s54_ap055", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.55),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s54_ap065", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.65),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s54_ap050_max10", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, merge_similar_max_merged_size=10, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_stitch_direct_k0_s2_t1_m068_s54_ap050_max10", stitch_overlap=True, stitch_score_gate=0.005, stitch_min_overlap=1, stitch_min_overlap_frac=0.6, stitch_query_topk=1, stitch_union_topk=True, stitch_topk_assign=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, merge_similar_max_merged_size=10, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_stitch_top2assign_direct_k0_s2_t1_m068_s54_ap050_max10", stitch_overlap=True, stitch_score_gate=0.005, stitch_min_overlap=1, stitch_min_overlap_frac=0.6, stitch_query_topk=2, stitch_union_topk=True, stitch_topk_assign=True, stitch_topk_assign_score_gate=-5.4, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, merge_similar_max_merged_size=10, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_stitch_top4assign_direct_k0_s2_t1_m068_s54_ap050_max10", stitch_overlap=True, stitch_score_gate=0.005, stitch_min_overlap=1, stitch_min_overlap_frac=0.6, stitch_query_topk=4, stitch_union_topk=True, stitch_topk_assign=True, stitch_topk_assign_score_gate=-5.4, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, merge_similar_max_merged_size=10, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_stitch_top2restore_direct_k0_s2_t1_m068_s54_ap050_max10", stitch_overlap=True, stitch_score_gate=0.005, stitch_min_overlap=1, stitch_min_overlap_frac=0.6, stitch_query_topk=2, stitch_union_topk=True, stitch_topk_assign=False, restore_topk_components=True, restore_topk_min_score=-5.4, restore_topk_max_conflict_size=0, restore_topk_min_target_size=3, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, merge_similar_max_merged_size=10, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m068_s54_ap050_bfsim", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, merge_similar_bf16_sim=True, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s2_t1_m069_s54_ap050_bfsim", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.69, merge_similar_bf16_sim=True, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_s2_t1_m068_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_s2_t1_m068_s54_ap050_km002", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_min_margin=0.002, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_s2_t1_m068_s54_ap050_km005", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_min_margin=0.005, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_ref2_s2_t1_m068_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_max_refine_cluster_size=2, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_ref3_s2_t1_m068_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_max_refine_cluster_size=3, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_ref4_s2_t1_m068_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_max_refine_cluster_size=4, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_ref5_s2_t1_m068_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_max_refine_cluster_size=5, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_ref6_s2_t1_m068_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_max_refine_cluster_size=6, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_ref4_s2_t1_m069_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.69, kmeans_iters=1, kmeans_max_refine_cluster_size=4, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_ref4_s2_t1_m070_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.70, kmeans_iters=1, kmeans_max_refine_cluster_size=4, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_ref4_s2_t2_m069_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.69, kmeans_iters=1, kmeans_max_refine_cluster_size=4, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_s2_t1_m069_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.69, kmeans_iters=1, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_s2_t1_m070_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.70, kmeans_iters=1, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_s2_t2_m068_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k1_s2_t2_m069_s54_ap050_km000", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.69, kmeans_iters=1, kmeans_min_margin=0.0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s1_t1_m068_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s1_t2_m068_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_direct_k0_s3_t3_m068_s54_ap050", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=3, merge_similar_direct_target_min_size=3, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.50),
            sm("fast5_clamp_k0_s2_t2_m068_s62_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-6.2, decoder_active_prob_power=0.60),
            sm("fast5_clamp_k0_s2_t2_m070_s58_ap075", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_mode="clamp_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.75),
            sm("fast5_rawtop2_k0_s2_t2_m068_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="rawtop2_clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_rawtop4_k0_s2_t2_m068_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="rawtop4_clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_rawtop8_k0_s2_t2_m068_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="rawtop8_clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_rawtop4_k0_s2_t2_m068_s58_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="rawtop4_clamp_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.60),
            sm("fast5_rawtop4_k0_s2_t2_m070_s54_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_mode="rawtop4_clamp_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.60),
            sm("fast5_lp_k0_s2_t2_m068_s58_ap075", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="logit_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.75),
            sm("fast5_lp_k0_s2_t2_m068_s62_ap075", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="logit_logprob", decoder_score_threshold=-6.2, decoder_active_prob_power=0.75),
            sm("fast5_lp_k0_s2_t2_m068_s54_ap075", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="logit_logprob", decoder_score_threshold=-5.4, decoder_active_prob_power=0.75),
            sm("fast5_lp_k0_s2_t2_m068_s58_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="logit_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.60),
            sm("fast5_lp_k0_s2_t2_m070_s58_ap075", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_score_mode="logit_logprob", decoder_score_threshold=-5.8, decoder_active_prob_power=0.75),
            sm("fast5_logit_k0_s2_t2_m068_s58", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8),
            sm("fast5_logit_k0_s2_t2_m068_s62", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-6.2),
            sm("fast5_logitgate_k0_s2_t2_m068_s58_a001", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_use_logit_scores=True, decoder_active_prob_threshold=0.01, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8),
            sm("fast5_logitgate_k0_s2_t2_m068_s58_a005", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_use_logit_scores=True, decoder_active_prob_threshold=0.05, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8),
            sm("fast5_logitgate_k0_s2_t2_m068_s58_a010", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_use_logit_scores=True, decoder_active_prob_threshold=0.10, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8),
            sm("fast5_logitgate_k0_s2_t2_m070_s58_a005", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_iters=0, decoder_use_logit_scores=True, decoder_active_prob_threshold=0.05, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8),
            sm("fast5_lo_k0_s2_t2_m068_s40_ap075", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="logit_logodds", decoder_score_threshold=-4.0, decoder_active_prob_power=0.75),
            sm("fast5_label_k0_s2_t3_m060_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=3, merge_similar_min_sim=0.60, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_label_k0_s2_t3_m065_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=3, merge_similar_min_sim=0.65, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_label_k0_s2_t3_m068_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=3, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_label_k0_s2_t4_m060_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=4, merge_similar_min_sim=0.60, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_label_k0_s2_t3_m060_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=3, merge_similar_min_sim=0.60, kmeans_iters=0, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_direct_k0_s1_t2_m060_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.60, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_direct_k0_s1_t2_m065_s003", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, kmeans_iters=0, decoder_score_threshold=0.003),
            sm("fast5_direct_k0_s2_t2_m060_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.60, kmeans_iters=0, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_direct_k0_s2_t2_m068_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_direct_k0_s2_t3_m060_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=3, merge_similar_min_sim=0.60, kmeans_iters=0, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_direct_k0_s2_t3_m065_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=3, merge_similar_min_sim=0.65, kmeans_iters=0, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_direct_k0_s2_t3_m068_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=3, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_direct_k0_s2_t4_m060_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=4, merge_similar_min_sim=0.60, kmeans_iters=0, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_direct_k0_s2_t2_m068_s002_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_threshold=0.002, decoder_active_prob_power=0.60),
            sm("fast5_direct_k0_s2_t2_m068_s001_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_threshold=0.001, decoder_active_prob_power=0.60),
            sm("fast5_direct_k1_ref2_s2_t2_m068_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_max_refine_cluster_size=2, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_direct_k1_ref3_s2_t2_m068_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_max_refine_cluster_size=3, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_direct_k1_ref5_s2_t2_m068_s003_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=1, kmeans_max_refine_cluster_size=5, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("fast5_logprob_direct_k0_s2_t2_m068_l621_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="logprob", decoder_score_threshold=-6.214608, decoder_active_prob_power=0.60),
            sm("fast5_logprob_direct_k0_s2_t2_m068_l690_ap060", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_label_space=False, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_mode="logprob", decoder_score_threshold=-6.907755, decoder_active_prob_power=0.60),
            sm("fast5_direct_k0_s2_t2_m068_s003_ap085", stitch_overlap=False, merge_similar=True, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, kmeans_iters=0, decoder_score_threshold=0.003, decoder_active_prob_power=0.85),
            sm("nostitch_prob_k1_direct_s2_t2_m070_s003_ap050", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.003, decoder_active_prob_power=0.50),
            sm("nostitch_prob_k1_direct_s2_t2_m070_s003_ap060", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("nostitch_prob_k1_direct_s2_t2_m070_s003_ap085", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.003, decoder_active_prob_power=0.85),
            sm("nostitch_prob_k1_direct_s2_t2_m070_s003_ap100", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.003, decoder_active_prob_power=1.00),
            sm("nostitch_prob_k1_direct_s2_t2_m068_s003_ap060", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003, decoder_active_prob_power=0.60),
            sm("nostitch_prob_k1_direct_s2_t2_m068_s003_ap085", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003, decoder_active_prob_power=0.85),
            sm("nostitch_prob_k1_direct_s2_t2_m068_s003_max8", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, merge_similar_max_merged_size=8, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s2_t2_m068_s003_max16", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, merge_similar_max_merged_size=16, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s2_t2_m070_s003_max8", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, merge_similar_max_merged_size=8, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s2_t2_m070_s003_max16", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, merge_similar_max_merged_size=16, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s2_t2_m068_s003_km000", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003, kmeans_min_margin=0.0),
            sm("nostitch_prob_k1_direct_s2_t2_m068_s003_km002", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003, kmeans_min_margin=0.002),
            sm("nostitch_prob_k1_direct_s2_t2_m068_s003_km010", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003, kmeans_min_margin=0.010),
            sm("nostitch_prob_k1_direct_s2_t2_m068_s003_s005", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.005),
            sm("nostitch_prob_k1_direct_s2_t2_m068_s003_floor025", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003, decoder_active_prob_floor=0.25),
            sm("nostitch_prob_k1_direct_s2_t2_m065_s002", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.002),
            sm("nostitch_prob_k1_direct_s2_t2_m065_s004", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.004),
            sm("nostitch_prob_k1_direct_s2_t2_m065_s003_km010", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, kmeans_min_margin=0.010),
            sm("nostitch_prob_k1_direct_s2_t2_m065_s003_pad64_after", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, pad_size2=True, pad_size2_after_kmeans=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("nostitch_prob_k1_direct_s3_t2_m075_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=3, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.75, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s3_t2_m080_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=3, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.80, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s3_t3_m080_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=3, merge_similar_direct_target_min_size=3, merge_similar_min_sim=0.80, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s4_t3_m085_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=4, merge_similar_direct_target_min_size=3, merge_similar_min_sim=0.85, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s4_t4_m085_s003", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=4, merge_similar_direct_target_min_size=4, merge_similar_min_sim=0.85, decoder_score_threshold=0.003),
            sm("primary3_bridge_s2_t2_m068_s003_b2", stitch_overlap=False, decoder_primary_sub_events=3, extra_window_bridge=True, extra_window_bridge_max_cluster_size=2, extra_window_bridge_max_merged_size=6, extra_window_bridge_min_score=0.003, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003),
            sm("primary3_bridge_s2_t2_m070_s003_b2", stitch_overlap=False, decoder_primary_sub_events=3, extra_window_bridge=True, extra_window_bridge_max_cluster_size=2, extra_window_bridge_max_merged_size=6, extra_window_bridge_min_score=0.003, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.003),
            sm("primary3_bridge_s2_t2_m068_s003_b3", stitch_overlap=False, decoder_primary_sub_events=3, extra_window_bridge=True, extra_window_bridge_max_cluster_size=3, extra_window_bridge_max_merged_size=8, extra_window_bridge_min_score=0.003, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003),
            sm("primary3_bridge_s2_t2_m068_s005_b2", stitch_overlap=False, decoder_primary_sub_events=3, extra_window_bridge=True, extra_window_bridge_max_cluster_size=2, extra_window_bridge_max_merged_size=6, extra_window_bridge_min_score=0.005, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003),
            sm("primary3_bridge_s2_t2_m068_s003_b2_h4k", stitch_overlap=False, decoder_primary_sub_events=3, extra_window_bridge=True, extra_window_bridge_max_cluster_size=2, extra_window_bridge_max_merged_size=6, extra_window_bridge_min_score=0.003, extra_window_bridge_max_hits=4096, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003),
            sm("primary3_bridge_s2_t2_m070_s003_b2_q2048", stitch_overlap=False, decoder_primary_sub_events=3, extra_window_bridge=True, extra_window_bridge_max_cluster_size=2, extra_window_bridge_max_merged_size=6, extra_window_bridge_min_score=0.003, extra_window_bridge_query_topk=2048, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.003),
            sm("primary3_bridge_s2_t2_m070_s003_b2_q1024", stitch_overlap=False, decoder_primary_sub_events=3, extra_window_bridge=True, extra_window_bridge_max_cluster_size=2, extra_window_bridge_max_merged_size=6, extra_window_bridge_min_score=0.003, extra_window_bridge_query_topk=1024, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.003),
            sm("primary3_bridge_s2_t2_m070_s003_b2_q512", stitch_overlap=False, decoder_primary_sub_events=3, extra_window_bridge=True, extra_window_bridge_max_cluster_size=2, extra_window_bridge_max_merged_size=6, extra_window_bridge_min_score=0.003, extra_window_bridge_query_topk=512, merge_similar_direct=True, merge_similar_small_cluster_size=2, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.003),
            sm("nostitch_prob_k1_direct_s1_t2_m065_s003_pad64", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("nostitch_prob_k1_direct_s1_t2_m060_s003_pad64", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.60, decoder_score_threshold=0.003, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("nostitch_prob_k1_direct_s1_t2_m065_s003_pad64_after", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, pad_size2=True, pad_size2_after_kmeans=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("nostitch_prob_k1_direct_s1_t2_m060_s003_pad64_after", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.60, decoder_score_threshold=0.003, pad_size2=True, pad_size2_after_kmeans=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("nostitch_prob_k1_direct_s1_t2_m065_s003_km0", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, kmeans_min_margin=0.0),
            sm("nostitch_prob_k1_direct_s1_t2_m065_s003_km2", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, kmeans_min_margin=0.002),
            sm("nostitch_logit_k1_direct_s1_t2_m065_s58", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8),
            sm("nostitch_logit_k1_direct_s1_t2_m065_s62", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-6.2),
            sm("nostitch_logit_k1_direct_s1_t2_m065_s66", stitch_overlap=False, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-6.6),
            sm("stitch_logit_k1_direct_s1_t2_m065_s58_g46", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8, stitch_score_gate=-4.6),
            sm("stitch_logit_k1_direct_s1_t2_m065_s62_g50", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-6.2, stitch_score_gate=-5.0),
            sm("k1_direct_s1_t2_m065_s003_q2048", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, decoder_active_query_topk=2048),
            sm("k1_direct_s1_t2_m065_s003_q1536", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, decoder_active_query_topk=1536),
            sm("k1_direct_s1_t2_m065_s003_q1280", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, decoder_active_query_topk=1280),
            sm("k1_direct_s1_t2_m065_s003_q1024", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, decoder_active_query_topk=1024),
            sm("k1_direct_s1_t2_m065_s003_q768", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_score_threshold=0.003, decoder_active_query_topk=768),
            sm("k1_direct_s1_t2_m065_logit_q2048", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8, stitch_score_gate=-4.6, decoder_active_query_topk=2048),
            sm("k1_direct_s1_t2_m065_logit_q1536", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8, stitch_score_gate=-4.6, decoder_active_query_topk=1536),
            sm("k1_direct_s1_t2_m065_logit_q1280", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8, stitch_score_gate=-4.6, decoder_active_query_topk=1280),
            sm("k1_direct_s1_t2_m065_logit_q1024", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8, stitch_score_gate=-4.6, decoder_active_query_topk=1024),
            sm("k1_direct_s1_t2_m065_logit_q768", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_use_logit_scores=True, decoder_active_prob_power=0.0, decoder_score_threshold=-5.8, stitch_score_gate=-4.6, decoder_active_query_topk=768),
            sm("k1_direct_s1_t2_m065_ap080", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65, decoder_active_prob_power=0.80),
            sm("k1_direct_s1_t2_m068_s003", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_score_threshold=0.003),
            sm("k1_direct_s1_t2_m068_ap080", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.68, decoder_active_prob_power=0.80),
            sm("k1_direct_s1_t2_m075", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.75),
            sm("k1_direct_s1_t2_m080", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.80),
            sm("k1_direct_s1_t2_m070_s000", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.0),
            sm("k1_direct_s1_t2_m070_s001", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.001),
            sm("k1_direct_s1_t2_m070_s003", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_score_threshold=0.003),
            sm("k1_direct_s1_t2_m070_sg005", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, stitch_score_gate=0.005),
            sm("k1_direct_s1_t2_m070_sg015", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, stitch_score_gate=0.015),
            sm("k1_direct_s1_t2_m070_km000", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_min_margin=0.0),
            sm("k1_direct_s1_t2_m070_km002", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_min_margin=0.002),
            sm("k1_direct_s1_t2_m070_km010", merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, kmeans_min_margin=0.010),
            sm("k1_ref2_direct_t1", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_max_refine_cluster_size=2),
            sm("k1_ref3_direct_t1", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_max_refine_cluster_size=3),
            sm("k1_ref5_direct_t1", merge_similar_direct=True, merge_similar_direct_target_min_size=1, kmeans_max_refine_cluster_size=5),
            sm("k1_ref3_direct_t2", merge_similar_direct=True, merge_similar_direct_target_min_size=2, kmeans_max_refine_cluster_size=3),
            sm("k0_merge_top2", kmeans_iters=0, merge_similar_topk=2),
            sm("k0_direct_merge", kmeans_iters=0, merge_similar_direct=True),
            sm("k0_direct_merge_t1", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=1),
            sm("k0_direct_merge_t2", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2),
            sm("k0_direct_merge_t1_m070", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.70),
            sm("k0_direct_merge_t2_m070", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70),
            sm("k0_direct_s1_t1_m070", kmeans_iters=0, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=1, merge_similar_min_sim=0.70),
            sm("k0_direct_s1_t2_m070", kmeans_iters=0, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70),
            sm("k0_direct_merge_t2_m065", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.65),
            sm("k0_direct_merge_t2_m060", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.60),
            sm("k0_direct_merge_t2_m055", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.55),
            sm("k0_direct_merge_t2_m070_ap085", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=0.85),
            sm("k0_direct_merge_t2_m070_ap090", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=0.90),
            sm("k0_direct_merge_t2_m070_ap095", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=0.95),
            sm("k0_direct_merge_t2_m070_ap100", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, decoder_active_prob_power=1.0),
            sm("k0_direct_merge_t2_m080", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.80),
            sm("k0_direct_t2_m070_pad32_c512", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("k0_direct_t2_m070_pad32_c1024", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=1024),
            sm("k0_direct_t2_m070_pad64_c512", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("k0_direct_t2_m070_pad32_ce010", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_clf_ceiling=0.010, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("k0_direct_t2_m070_geom32_c512", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.3, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512, pad_size2_geom_z_max=0.005, pad_size2_geom_phi_max=0.04),
            sm("k0_direct_s1_t2_m070_pad32_c512", kmeans_iters=0, merge_similar_direct=True, merge_similar_small_cluster_size=1, merge_similar_direct_target_min_size=2, merge_similar_min_sim=0.70, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("k1_direct_t1_pad32_c512", merge_similar_direct=True, merge_similar_direct_target_min_size=1, pad_size2=True, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, pad_size2_topk=1, pad_size2_max_candidates=512),
            sm("k0_direct_merge_t2_sg005", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, stitch_score_gate=0.005),
            sm("k0_direct_merge_t2_sg002", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, stitch_score_gate=0.002),
            sm("k0_direct_merge_t2_ap100", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, decoder_active_prob_power=1.0),
            sm("k0_direct_merge_t2_ap050", kmeans_iters=0, merge_similar_direct=True, merge_similar_direct_target_min_size=2, decoder_active_prob_power=0.5),
            sm("k0_direct_merge_m070", kmeans_iters=0, merge_similar_direct=True, merge_similar_min_sim=0.70),
            sm("k0_merge", kmeans_iters=0),
            sm("k0_nomerge", kmeans_iters=0, merge_similar=False),
            sm("nostitch_k1_merge", stitch_overlap=False),
            sm("nostitch_k1_nomerge", stitch_overlap=False, merge_similar=False),
        ]
    if name == "kmeans_pad_fast":
        base = dict(
            kmeans_iters=1,
            kmeans_min_margin=0.005,
            clf_recall=1.0,
            decoder_score_threshold=0.002,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            pad_size2_min_sim=0.3,
            pad_size2_clf_floor=0.0,
            pad_size2_clf_ceiling=0.010,
            pad_size2_unique_layer=True,
            pad_size2_topk=4,
            pad_size2_max_candidates=0,
            pad_size2_max_clusters=0,
            residual_regroup=False,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            split_window_clusters=False,
            split_overmerged=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        pad = dict(pad_size2=True)
        return [
            sm("k1_s002_m005_base"),
            sm("pad_ce010_all", **pad, pad_size2_max_candidates=0),
            sm("pad_ce010_c512", **pad, pad_size2_max_candidates=512),
            sm("pad_ce010_c1024", **pad, pad_size2_max_candidates=1024),
            sm("pad_ce010_c2048", **pad, pad_size2_max_candidates=2048),
            sm("pad_ce005_c1024", **pad, pad_size2_clf_ceiling=0.005, pad_size2_max_candidates=1024),
            sm("pad_ce020_c1024", **pad, pad_size2_clf_ceiling=0.020, pad_size2_max_candidates=1024),
            sm("pad_sim04_c1024", **pad, pad_size2_min_sim=0.4, pad_size2_max_candidates=1024),
            sm("pad_top8_c1024", **pad, pad_size2_topk=8, pad_size2_max_candidates=1024),
            sm("pad_c1024_rg128", **pad, pad_size2_max_candidates=1024, residual_regroup=True, residual_regroup_max_cluster_size=2, residual_regroup_max_hits=128, residual_regroup_topk=8, residual_regroup_seed_sim=0.55, residual_regroup_pair_sim=0.15, residual_regroup_require_unique_layer=True, residual_regroup_clear_pool=False),
        ]
    if name == "kmeans_pad_steal":
        base = dict(
            pre_kmeans_iters=0,
            pre_kmeans_min_margin=0.0,
            kmeans_iters=1,
            kmeans_min_margin=0.005,
            clf_recall=1.0,
            decoder_score_threshold=0.002,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.010,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            pad_size2_min_sim=0.3,
            pad_size2_clf_floor=0.0,
            pad_size2_clf_ceiling=0.0,
            pad_size2_unique_layer=True,
            pad_size2_topk=4,
            pad_size2_max_candidates=0,
            pad_size2_max_clusters=0,
            pad_size2_max_conflict_size=0,
            pad_size2_geom_z_max=0.0,
            pad_size2_geom_phi_max=0.0,
            residual_regroup=False,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            split_window_clusters=False,
            split_overmerged=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            prune_small_clusters=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        pad = dict(pad_size2=True)
        return [
            sm("base"),
            sm("pad_unassigned", **pad),
            sm("pad_unassigned_sim04", **pad, pad_size2_min_sim=0.4),
            sm("pad_unassigned_ce050", **pad, pad_size2_clf_ceiling=0.050),
            sm("steal1", **pad, pad_size2_max_conflict_size=1),
            sm("steal2", **pad, pad_size2_max_conflict_size=2),
            sm("steal2_sim04", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4),
            sm("steal2_sim05", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.5),
            sm("steal2_top8", **pad, pad_size2_max_conflict_size=2, pad_size2_topk=8),
            sm("steal2_c2048", **pad, pad_size2_max_conflict_size=2, pad_size2_max_candidates=2048),
            sm("steal2_c4096", **pad, pad_size2_max_conflict_size=2, pad_size2_max_candidates=4096),
            sm("steal2_ce050", **pad, pad_size2_max_conflict_size=2, pad_size2_clf_ceiling=0.050),
            sm("steal2_floor005", **pad, pad_size2_max_conflict_size=2, pad_size2_clf_floor=0.005),
            sm("steal2_mc32_s04", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32),
            sm("steal2_mc48_s04", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=48),
            sm("steal2_mc64_s04", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64),
            sm("steal2_mc64_s04_t1", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64, pad_size2_topk=1),
            sm("steal2_mc64_s04_t2", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=64, pad_size2_topk=2),
            sm("steal2_mc128_s04", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=128),
            sm("steal2_mc96_s04_t1", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=96, pad_size2_topk=1),
            sm("steal2_mc256_s04", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=256),
            sm("steal2_mc512_s04", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=512),
            sm("steal2_mc128_s05", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.5, pad_size2_max_clusters=128),
            sm("steal2_mc64_s05_t1", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.5, pad_size2_max_clusters=64, pad_size2_topk=1),
            sm("steal2_mc64_s05_t2", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.5, pad_size2_max_clusters=64, pad_size2_topk=2),
            sm("steal2_mc256_s05", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.5, pad_size2_max_clusters=256),
            sm("steal2_mc512_s05", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.5, pad_size2_max_clusters=512),
            sm("steal2_mc256_c2048", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=256, pad_size2_max_candidates=2048),
            sm("rg64_pad_mc32", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, residual_regroup=True, residual_regroup_max_cluster_size=2, residual_regroup_max_hits=64, residual_regroup_topk=8, residual_regroup_seed_sim=0.55, residual_regroup_pair_sim=0.15, residual_regroup_require_unique_layer=True, residual_regroup_clear_pool=False),
            sm("rg128_pad_mc32", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=32, residual_regroup=True, residual_regroup_max_cluster_size=2, residual_regroup_max_hits=128, residual_regroup_topk=8, residual_regroup_seed_sim=0.55, residual_regroup_pair_sim=0.15, residual_regroup_require_unique_layer=True, residual_regroup_clear_pool=False),
            sm("rg64_pad_mc16", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=16, residual_regroup=True, residual_regroup_max_cluster_size=2, residual_regroup_max_hits=64, residual_regroup_topk=8, residual_regroup_seed_sim=0.55, residual_regroup_pair_sim=0.15, residual_regroup_require_unique_layer=True, residual_regroup_clear_pool=False),
            sm("geom_mc64", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.3, pad_size2_max_clusters=64, pad_size2_geom_z_max=0.005, pad_size2_geom_phi_max=0.04),
            sm("geom_mc128", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.3, pad_size2_max_clusters=128, pad_size2_geom_z_max=0.005, pad_size2_geom_phi_max=0.04),
            sm("geom_mc256", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.3, pad_size2_max_clusters=256, pad_size2_geom_z_max=0.005, pad_size2_geom_phi_max=0.04),
            sm("geom_mc512", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.3, pad_size2_max_clusters=512, pad_size2_geom_z_max=0.005, pad_size2_geom_phi_max=0.04),
            sm("geom_tight_mc256", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.3, pad_size2_max_clusters=256, pad_size2_geom_z_max=0.002, pad_size2_geom_phi_max=0.03),
            sm("geom_s04_mc256", **pad, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.4, pad_size2_max_clusters=256, pad_size2_geom_z_max=0.005, pad_size2_geom_phi_max=0.04),
            sm("nok_base", kmeans_iters=0),
            sm("nok_geom256", **pad, kmeans_iters=0, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.3, pad_size2_max_clusters=256, pad_size2_geom_z_max=0.005, pad_size2_geom_phi_max=0.04),
            sm("nok_geom512", **pad, kmeans_iters=0, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.3, pad_size2_max_clusters=512, pad_size2_geom_z_max=0.005, pad_size2_geom_phi_max=0.04),
            sm("nok_geom1024", **pad, kmeans_iters=0, pad_size2_max_conflict_size=2, pad_size2_min_sim=0.3, pad_size2_max_clusters=1024, pad_size2_geom_z_max=0.005, pad_size2_geom_phi_max=0.04),
        ]
    if name == "kmeans_split_pair":
        base = dict(
            pre_kmeans_iters=0,
            pre_kmeans_min_margin=0.0,
            kmeans_iters=1,
            kmeans_min_margin=0.005,
            clf_recall=1.0,
            decoder_score_threshold=0.002,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.010,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            split_overmerged=False,
            split_overmerged_min_size=4,
            split_overmerged_max_size=8,
            split_overmerged_min_part_size=2,
            split_overmerged_min_duplicate_layers=1,
            split_overmerged_max_part_duplicate_layers=0,
            split_overmerged_min_improve=0.04,
            split_overmerged_max_centroid_sim=0.85,
            split_overmerged_iters=4,
            split_overmerged_max_clusters=128,
            pad_size2=False,
            pad_size2_min_sim=0.4,
            pad_size2_clf_floor=0.0,
            pad_size2_clf_ceiling=0.0,
            pad_size2_unique_layer=True,
            pad_size2_topk=4,
            pad_size2_max_candidates=0,
            pad_size2_max_clusters=32,
            pad_size2_max_conflict_size=2,
            residual_regroup=False,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            split_window_clusters=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            prune_small_clusters=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        split = dict(split_overmerged=True)
        pad = dict(pad_size2=True)
        return [
            sm("base"),
            sm("pad_mc32", **pad),
            sm("split4p2_dup_i04_s85", **split),
            sm("split4p2_dup_i02_s90", **split, split_overmerged_min_improve=0.02, split_overmerged_max_centroid_sim=0.90),
            sm("split4p2_dup_i06_s80", **split, split_overmerged_min_improve=0.06, split_overmerged_max_centroid_sim=0.80),
            sm("split5p2_dup_i04", **split, split_overmerged_min_size=5),
            sm("split6p3_dup_i04", **split, split_overmerged_min_size=6, split_overmerged_min_part_size=3),
            sm("split4p2_dup_pad", **split, **pad),
            sm("split4p2_loose_pad", **split, **pad, split_overmerged_min_improve=0.02, split_overmerged_max_centroid_sim=0.90),
            sm("split5p2_dup_pad", **split, **pad, split_overmerged_min_size=5),
            sm("split4p2_max64_pad", **split, **pad, split_overmerged_max_clusters=64),
            sm("split4p2_max32_pad", **split, **pad, split_overmerged_max_clusters=32),
        ]
    if name == "prune_grid":
        base = dict(
            kmeans_iters=1,
            kmeans_min_margin=0.005,
            clf_recall=1.0,
            decoder_score_threshold=0.002,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_union_topk=True,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            split_window_clusters=False,
            split_overmerged=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            prune_small_clusters=False,
            prune_small_cluster_max_size=5,
            prune_small_cluster_min_size_after=3,
            prune_small_cluster_max_drop=1,
            prune_small_cluster_min_sim_gap=0.0,
            prune_small_cluster_min_score_gap=0.0,
            prune_small_cluster_require_duplicate_layer=False,
            prune_small_cluster_duplicate_layer_only=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        prune = dict(prune_small_clusters=True)
        return [
            sm("k1_s002_m005_base"),
            sm("prune_sim02_m5", **prune, prune_small_cluster_min_sim_gap=0.02),
            sm("prune_sim04_m5", **prune, prune_small_cluster_min_sim_gap=0.04),
            sm("prune_sim06_m5", **prune, prune_small_cluster_min_sim_gap=0.06),
            sm("prune_sim08_m5", **prune, prune_small_cluster_min_sim_gap=0.08),
            sm("prune_score001_m5", **prune, prune_small_cluster_min_score_gap=0.001),
            sm("prune_score005_m5", **prune, prune_small_cluster_min_score_gap=0.005),
            sm("prune_sim04_score001", **prune, prune_small_cluster_min_sim_gap=0.04, prune_small_cluster_min_score_gap=0.001),
            sm("prune_sim04_m6", **prune, prune_small_cluster_max_size=6, prune_small_cluster_min_sim_gap=0.04),
            sm("prune_sim04_drop2", **prune, prune_small_cluster_max_drop=2, prune_small_cluster_min_sim_gap=0.04),
            sm("prune_dup_sim02", **prune, prune_small_cluster_require_duplicate_layer=True, prune_small_cluster_min_sim_gap=0.02),
            sm("prune_dup_only_score001", **prune, prune_small_cluster_require_duplicate_layer=True, prune_small_cluster_duplicate_layer_only=True, prune_small_cluster_min_score_gap=0.001),
            sm("prune_rg128", **prune, prune_small_cluster_min_sim_gap=0.04, residual_regroup=True, residual_regroup_max_cluster_size=2, residual_regroup_max_hits=128, residual_regroup_topk=8, residual_regroup_seed_sim=0.55, residual_regroup_pair_sim=0.15, residual_regroup_require_unique_layer=True, residual_regroup_clear_pool=False),
        ]
    if name == "window_restore":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.001,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_topk_score_gate=None,
            stitch_topk_assign=False,
            restore_window_clusters=False,
            restore_window_min_size=3,
            restore_window_min_new_hits=3,
            restore_window_min_score=0.0,
            restore_window_max_conflict_size=2,
            restore_window_min_new_frac=0.6,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            split_overmerged=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        restore = dict(restore_window_clusters=True)
        return [
            sm("base_ap075_s001"),
            sm("restore_s0_frac06", **restore, restore_window_min_score=0.0, restore_window_min_new_frac=0.6),
            sm("restore_s001_frac06", **restore, restore_window_min_score=0.001, restore_window_min_new_frac=0.6),
            sm("restore_s005_frac06", **restore, restore_window_min_score=0.005, restore_window_min_new_frac=0.6),
            sm("restore_s010_frac06", **restore, restore_window_min_score=0.010, restore_window_min_new_frac=0.6),
            sm("restore_s001_frac08", **restore, restore_window_min_score=0.001, restore_window_min_new_frac=0.8),
            sm("restore_s005_frac08", **restore, restore_window_min_score=0.005, restore_window_min_new_frac=0.8),
            sm("restore_conflict1", **restore, restore_window_min_score=0.001, restore_window_max_conflict_size=1),
            sm("restore_new4", **restore, restore_window_min_score=0.001, restore_window_min_new_hits=4),
            sm("restore_size4", **restore, restore_window_min_score=0.001, restore_window_min_size=4),
            sm(
                "ap025_restore_s001",
                **restore,
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                restore_window_min_score=0.001,
            ),
            sm(
                "ap025_restore_s005",
                **restore,
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                restore_window_min_score=0.005,
            ),
        ]
    if name == "window_bridge":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.001,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            restore_window_clusters=False,
            restore_window_min_score=0.005,
            restore_window_max_conflict_size=2,
            restore_window_min_new_frac=0.6,
            bridge_window_clusters=False,
            bridge_window_min_score=0.005,
            bridge_window_min_candidate_size=3,
            bridge_window_min_label_overlap=1,
            bridge_window_max_cluster_size=3,
            bridge_window_max_merged_size=8,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            split_overmerged=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        bridge = dict(bridge_window_clusters=True)
        restore = dict(restore_window_clusters=True, restore_window_min_score=0.005)
        return [
            sm("base_ap075_s001"),
            sm("bridge_s005_c3_m8", **bridge, bridge_window_min_score=0.005, bridge_window_max_cluster_size=3, bridge_window_max_merged_size=8),
            sm("bridge_s010_c3_m8", **bridge, bridge_window_min_score=0.010, bridge_window_max_cluster_size=3, bridge_window_max_merged_size=8),
            sm("bridge_s020_c3_m8", **bridge, bridge_window_min_score=0.020, bridge_window_max_cluster_size=3, bridge_window_max_merged_size=8),
            sm("bridge_s005_c2_m8", **bridge, bridge_window_min_score=0.005, bridge_window_max_cluster_size=2, bridge_window_max_merged_size=8),
            sm("bridge_s005_c3_m6", **bridge, bridge_window_min_score=0.005, bridge_window_max_cluster_size=3, bridge_window_max_merged_size=6),
            sm("bridge_s005_ov2", **bridge, bridge_window_min_score=0.005, bridge_window_min_label_overlap=2),
            sm("bridge_restore_s005", **bridge, **restore, bridge_window_min_score=0.005),
            sm("bridge_restore_s010", **bridge, **restore, bridge_window_min_score=0.010),
            sm(
                "ap025_bridge_s010",
                **bridge,
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                bridge_window_min_score=0.010,
            ),
            sm(
                "ap025_bridge_restore_s010",
                **bridge,
                **restore,
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                bridge_window_min_score=0.010,
            ),
        ]
    if name == "window_pad":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.001,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            pad_window_min_score=0.005,
            pad_window_extra_min_score=0.005,
            pad_window_min_overlap=2,
            pad_window_unique_layer=True,
            pad_window_max_proposals_per_label=0,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            split_overmerged=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        pad = dict(pad_window_size2=True)
        return [
            sm("base_ap075_s001"),
            sm("wpad_s001_e001", **pad, pad_window_min_score=0.001, pad_window_extra_min_score=0.001),
            sm("wpad_s005_e005", **pad, pad_window_min_score=0.005, pad_window_extra_min_score=0.005),
            sm("wpad_s010_e010", **pad, pad_window_min_score=0.010, pad_window_extra_min_score=0.010),
            sm("wpad_s020_e020", **pad, pad_window_min_score=0.020, pad_window_extra_min_score=0.020),
            sm("wpad_s005_e010", **pad, pad_window_min_score=0.005, pad_window_extra_min_score=0.010),
            sm("wpad_s010_e005", **pad, pad_window_min_score=0.010, pad_window_extra_min_score=0.005),
            sm("wpad_no_layer", **pad, pad_window_min_score=0.010, pad_window_extra_min_score=0.010, pad_window_unique_layer=False),
            sm("wpad_bridge", **pad, bridge_window_clusters=True, bridge_window_min_score=0.020),
            sm("wpad_restore", **pad, restore_window_clusters=True, restore_window_min_score=0.010),
            sm("wpad_top1_s005", **pad, pad_window_min_score=0.005, pad_window_extra_min_score=0.005, pad_window_max_proposals_per_label=1),
            sm("wpad_top1_s010", **pad, pad_window_min_score=0.010, pad_window_extra_min_score=0.010, pad_window_max_proposals_per_label=1),
            sm("wpad_top2_s010", **pad, pad_window_min_score=0.010, pad_window_extra_min_score=0.010, pad_window_max_proposals_per_label=2),
            sm(
                "ap025_wpad_s010",
                **pad,
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                pad_window_min_score=0.010,
                pad_window_extra_min_score=0.010,
            ),
            sm(
                "ap025_wpad_s020",
                **pad,
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                pad_window_min_score=0.020,
                pad_window_extra_min_score=0.020,
            ),
        ]
    if name == "window_pad_fast":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.001,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            restore_window_clusters=False,
            bridge_window_clusters=False,
            pad_window_size2=False,
            pad_window_min_score=0.005,
            pad_window_extra_min_score=0.005,
            pad_window_min_overlap=2,
            pad_window_unique_layer=True,
            pad_window_max_proposals_per_label=1,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            split_overmerged=False,
            split_window_clusters=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        pad = dict(pad_window_size2=True)
        return [
            sm("base_ap075_s001"),
            sm("wpad_top1_s001", **pad, pad_window_min_score=0.001, pad_window_extra_min_score=0.001, pad_window_max_proposals_per_label=1),
            sm("wpad_top1_s005", **pad, pad_window_min_score=0.005, pad_window_extra_min_score=0.005, pad_window_max_proposals_per_label=1),
            sm("wpad_top1_s010", **pad, pad_window_min_score=0.010, pad_window_extra_min_score=0.010, pad_window_max_proposals_per_label=1),
            sm("wpad_top1_s020", **pad, pad_window_min_score=0.020, pad_window_extra_min_score=0.020, pad_window_max_proposals_per_label=1),
            sm("wpad_top2_s010", **pad, pad_window_min_score=0.010, pad_window_extra_min_score=0.010, pad_window_max_proposals_per_label=2),
            sm("wpad_top1_ov1_s010", **pad, pad_window_min_score=0.010, pad_window_extra_min_score=0.010, pad_window_min_overlap=1, pad_window_max_proposals_per_label=1),
            sm("wpad_top1_nolayer", **pad, pad_window_min_score=0.010, pad_window_extra_min_score=0.010, pad_window_unique_layer=False, pad_window_max_proposals_per_label=1),
            sm("ap025_wpad_top1_s010", **pad, decoder_active_prob_power=0.25, decoder_score_threshold=0.0, pad_window_min_score=0.010, pad_window_extra_min_score=0.010, pad_window_max_proposals_per_label=1),
        ]
    if name == "window_split":
        base = dict(
            kmeans_iters=0,
            clf_recall=1.0,
            decoder_score_threshold=0.001,
            decoder_active_prob_threshold=0.0,
            decoder_active_prob_power=0.75,
            decoder_active_prob_floor=0.0,
            cluster_score_filter=False,
            stitch_overlap=True,
            stitch_score_gate=0.005,
            stitch_min_overlap=1,
            stitch_min_overlap_frac=0.6,
            stitch_query_topk=1,
            stitch_topk_score_gate=None,
            stitch_topk_assign=False,
            restore_window_clusters=False,
            restore_window_min_score=0.005,
            restore_window_max_conflict_size=2,
            restore_window_min_new_frac=0.6,
            bridge_window_clusters=False,
            bridge_window_min_score=0.005,
            bridge_window_min_candidate_size=3,
            bridge_window_min_label_overlap=1,
            bridge_window_max_cluster_size=2,
            bridge_window_max_merged_size=8,
            pad_window_size2=False,
            split_window_clusters=False,
            split_window_min_score=0.005,
            split_window_min_cluster_size=5,
            split_window_max_cluster_size=12,
            split_window_min_part_size=3,
            split_window_min_remaining_size=3,
            split_window_min_inside_frac=0.6,
            split_window_max_part_frac=0.7,
            split_window_min_duplicate_layers=0,
            split_window_max_part_duplicate_layers=-1,
            split_window_require_remainder_candidate=False,
            split_window_min_remainder_candidate_hits=3,
            split_window_max_splits=128,
            merge_similar=True,
            merge_similar_min_sim=0.75,
            merge_similar_small_cluster_size=2,
            merge_similar_max_merged_size=12,
            merge_similar_topk=4,
            merge_similar_require_unique_layer=False,
            pad_size2=False,
            residual_regroup=False,
            dedup_same_layer=False,
            dedup_same_layer_by_score=False,
            split_overmerged=False,
        )

        def sm(name_, **kwargs):
            out = dict(base)
            out.update(kwargs)
            return (name_, pp(**out))

        split = dict(split_window_clusters=True)
        bridge = dict(bridge_window_clusters=True, bridge_window_min_score=0.020)
        restore = dict(restore_window_clusters=True, restore_window_min_score=0.005)
        return [
            sm("base_ap075_s001"),
            sm("wsplit_s005_c5_12_if06_pf07", **split, split_window_min_score=0.005),
            sm("wsplit_s010_c5_12_if06_pf07", **split, split_window_min_score=0.010),
            sm("wsplit_s020_c5_12_if06_pf07", **split, split_window_min_score=0.020),
            sm(
                "wsplit_s005_if08",
                **split,
                split_window_min_score=0.005,
                split_window_min_inside_frac=0.8,
            ),
            sm(
                "wsplit_s005_pf06",
                **split,
                split_window_min_score=0.005,
                split_window_max_part_frac=0.6,
            ),
            sm(
                "wsplit_s005_c4_10",
                **split,
                split_window_min_score=0.005,
                split_window_min_cluster_size=4,
                split_window_max_cluster_size=10,
            ),
            sm(
                "wsplit_s010_c4_10",
                **split,
                split_window_min_score=0.010,
                split_window_min_cluster_size=4,
                split_window_max_cluster_size=10,
            ),
            sm(
                "wsplit_s005_part4",
                **split,
                split_window_min_score=0.005,
                split_window_min_part_size=4,
                split_window_min_remaining_size=3,
            ),
            sm(
                "wsplit_s010_part4",
                **split,
                split_window_min_score=0.010,
                split_window_min_part_size=4,
                split_window_min_remaining_size=3,
            ),
            sm("wsplit_bridge", **split, **bridge, split_window_min_score=0.010),
            sm("wsplit_restore", **split, **restore, split_window_min_score=0.010),
            sm(
                "wsplit_dup_s005",
                **split,
                split_window_min_score=0.005,
                split_window_min_duplicate_layers=1,
                split_window_max_part_duplicate_layers=0,
            ),
            sm(
                "wsplit_dup_s010",
                **split,
                split_window_min_score=0.010,
                split_window_min_duplicate_layers=1,
                split_window_max_part_duplicate_layers=0,
            ),
            sm(
                "wsplit_dup_rem_s005",
                **split,
                split_window_min_score=0.005,
                split_window_min_duplicate_layers=1,
                split_window_max_part_duplicate_layers=0,
                split_window_require_remainder_candidate=True,
            ),
            sm(
                "wsplit_dup_rem_s010",
                **split,
                split_window_min_score=0.010,
                split_window_min_duplicate_layers=1,
                split_window_max_part_duplicate_layers=0,
                split_window_require_remainder_candidate=True,
            ),
            sm(
                "wsplit_dup_rem_if08",
                **split,
                split_window_min_score=0.005,
                split_window_min_duplicate_layers=1,
                split_window_max_part_duplicate_layers=0,
                split_window_require_remainder_candidate=True,
                split_window_min_inside_frac=0.8,
            ),
            sm(
                "wsplit_dup_rem_part4",
                **split,
                split_window_min_score=0.005,
                split_window_min_duplicate_layers=1,
                split_window_max_part_duplicate_layers=0,
                split_window_require_remainder_candidate=True,
                split_window_min_part_size=4,
            ),
            sm(
                "ap025_wsplit_s010",
                **split,
                decoder_active_prob_power=0.25,
                decoder_score_threshold=0.0,
                split_window_min_score=0.010,
            ),
        ]
    raise ValueError(f"Unknown preset: {name}")


def _pp_kwargs(pp_cfg):
    return {key: pp_cfg[key] for key in PP_KEYS if key in pp_cfg}


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
        if cfg["postprocessing"].get("embedding_output_bf16", False) and full_data.emb_all.is_cuda:
            full_data.emb_all = full_data.emb_all.to(torch.bfloat16)
        full_data.clf_scores = enc_out["clf_probs"].detach()
        if cfg["postprocessing"].get("decoder_batched_decode", False):
            all_idx = getattr(full_data, "decoder_sub_all_idx", None)
            valid_masks = getattr(full_data, "decoder_sub_valid_mask", None)
            if all_idx is None or valid_masks is None:
                all_idx = torch.stack([sub["all_idx"] for sub in sub_events], dim=0)
                valid_masks = torch.stack([sub["valid_mask"] for sub in sub_events], dim=0)
            pred = model.decode(
                enc_out["encoded"][all_idx],
                full_data.coords[all_idx],
                valid_masks,
            )
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
                sub_coords = full_data.coords[sub["all_idx"]]
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

    out = {}
    timings = {}
    for name, pp_cfg in variants:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        predicted = decoder_kmeans_postprocess(
            raw_masks,
            sub_events,
            full_data,
            full_emb=full_data.emb_all,
            full_clf_scores=full_data.clf_scores,
            **_pp_kwargs(pp_cfg),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings[name] = time.perf_counter() - start
        out[name] = compute_tracking_metrics_from_prediction(
            predicted,
            full_data,
            pt_thld=cfg["eval"]["pt_thld"],
            predicted_count_thld=cfg["eval"]["predicted_count_thld"],
            min_track_length=cfg["eval"]["min_track_length"],
        )
    return out, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "infer.yaml"))
    parser.add_argument(
        "--preset",
        choices=[
            "residual_smoke",
            "residual_pair",
            "residual_fast",
            "residual_tight",
            "padding_ceiling",
            "stitch_clf",
            "split_overmerge",
            "dedup_layer",
            "merge_residual",
            "claude_regroup",
            "claude_fast",
            "budget20",
            "stitch_merge_grid",
            "active_score_grid",
            "active_maxprob_grid",
            "active_power_fine",
            "shifted_topk",
            "topk_restore",
            "kmeans_grid",
            "kmeans_tune",
            "kmeans_merge_stitch_fine",
            "kmeans_order",
            "kmeans_residual_tiny",
            "kmeans_pad_fast",
            "kmeans_pad_steal",
            "kmeans_split_pair",
            "latency30",
            "prune_grid",
            "window_restore",
            "window_bridge",
            "window_pad",
            "window_pad_fast",
            "window_split",
        ],
        default="residual_smoke",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, args.override)

    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(int(cfg.get("num_threads", 1)))
    try:
        torch.set_num_interop_threads(int(cfg.get("num_interop_threads", 1)))
    except RuntimeError:
        pass
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
    batched_decode = bool(cfg.get("postprocessing", {}).get("decoder_batched_decode", False))
    model.encode = torch.compile(model.encode, dynamic=False)
    if batched_decode:
        print("[compile] decoder_batched_decode=true: keeping decoder eager to avoid slow batched graph compilation.")
    else:
        model.decode = torch.compile(model.decode, dynamic=False)
        model = torch.compile(model, dynamic=False)

    variants = _variant_presets(args.preset, cfg["postprocessing"])
    if args.variant:
        keep = set(args.variant)
        variants = [(name, pp) for name, pp in variants if name in keep]
    sums = {name: defaultdict(list) for name, _ in variants}
    limit = cfg["eval"].get("limit_events")
    timing_skip = int(cfg["eval"].get("postprocess_timing_skip_events", 0) or 0)
    for i, data in enumerate(tqdm(loader, desc=f"sweep:{args.preset}")):
        if limit is not None and i >= int(limit):
            break
        per_variant, timings = _run_event(model, data, cfg, device, amp_enabled, amp_dtype, variants)
        for name, metrics in per_variant.items():
            for k, v in metrics.items():
                if isinstance(v, float) and np.isnan(v):
                    continue
                sums[name][k].append(v)
            if i >= timing_skip:
                sums[name]["postprocess_sec"].append(timings[name])

    print("\n=== sweep results ===")
    for name, _ in variants:
        row = {k: float(np.asarray(vals, dtype=np.float64).mean()) for k, vals in sums[name].items()}
        print(
            f"{name:20s} dm={row.get('dm', float('nan')):.5f} "
            f"fake={row.get('fake_rate', float('nan')):.5f} "
            f"eff={row.get('technical_efficiency', float('nan')):.5f} "
            f"dup={row.get('dup_rate', float('nan')):.5f} "
            f"clusters={row.get('n_cleaned_clusters', float('nan')):.1f} "
            f"pp_ms={row.get('postprocess_sec', float('nan')) * 1000.0:.2f}"
        )


if __name__ == "__main__":
    main()
