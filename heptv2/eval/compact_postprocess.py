import os

import torch

from .postprocess import (
    _build_decoder_assignments_from_mask_embed_batched_top1,
    _codex_nomask_target_core_compiled,
    _compute_clf_keep_mask,
)


def postprocess_compact_assignment(compact_pred, full_data, cfg):
    pp = cfg["postprocessing"]
    if pp.get("decoder_score_mode", "prob") != "clamp_logprob":
        raise RuntimeError("compact assignment currently implements clamp_logprob mode only")
    clf_keep = _compute_clf_keep_mask(
        full_data,
        getattr(full_data, "clf_scores", None),
        clf_recall=pp.get("clf_recall", 1.0),
    )
    need_hit_scores = (
        float(os.environ.get("HEPTV2_CODEX_REFINE_SCORE_MAX", "1.0e9")) < 1.0e8
        or bool(int(os.environ.get("HEPTV2_CODEX_DEDUP_BY_HIT_SCORE", "0")))
    )
    assignment_result = _build_decoder_assignments_from_mask_embed_batched_top1(
        compact_pred["mask_embed"],
        compact_pred["point_feats"],
        compact_pred["pred_logits"],
        compact_pred.get("_serialized_point_order", None),
        full_data,
        clf_keep,
        mask_thres=pp.get("decoder_score_threshold", 0.0),
        active_prob_power=pp.get("decoder_active_prob_power", 1.0),
        query_chunk_size=pp.get("decoder_assignment_chunk_size", 512),
        active_query_topk=pp.get("decoder_active_query_topk", 0),
        return_scores=need_hit_scores,
    )
    if need_hit_scores and assignment_result is not None:
        predicted, hit_scores = assignment_result
    else:
        predicted = assignment_result
        hit_scores = None
    if predicted is None:
        raise RuntimeError("compact assignment path could not build assignments")

    if not pp.get("codex_fused_nomask_target", False):
        raise RuntimeError("compact assignment requires codex_fused_nomask_target=true")
    fixed_num_labels = int(pp.get("codex_fused_num_labels", 0) or 0)
    if fixed_num_labels <= 0:
        mask_embed = compact_pred["mask_embed"]
        if mask_embed.dim() >= 3:
            fixed_num_labels = int(mask_embed.shape[0]) * int(mask_embed.shape[1])
        else:
            fixed_num_labels = int(mask_embed.shape[0])

    normalized_emb = full_data.emb_all if pp.get("embeddings_already_normalized", False) else None
    if fixed_num_labels <= 1 or not hasattr(full_data, "layer") or full_data.layer is None:
        return predicted
    if getattr(full_data, "clf_scores", None) is None or normalized_emb is None:
        return predicted

    raw_len = int(predicted.shape[0])
    pad_len = int(pp.get("codex_fused_pad_length", 74040) or 0)
    valid_mask = full_data.valid_mask.bool()
    clf_scores = full_data.clf_scores.squeeze(-1).float()
    layers = full_data.layer.long()
    coords = full_data.coords
    if pad_len > raw_len:
        extra = pad_len - raw_len
        predicted_in = torch.cat(
            [
                predicted,
                torch.full((extra,), -1, device=predicted.device, dtype=predicted.dtype),
            ],
            dim=0,
        )
        valid_in = torch.cat(
            [valid_mask, torch.zeros((extra,), device=valid_mask.device, dtype=torch.bool)],
            dim=0,
        )
        emb_in = torch.cat(
            [
                normalized_emb,
                torch.zeros(
                    (extra, int(normalized_emb.shape[1])),
                    device=normalized_emb.device,
                    dtype=normalized_emb.dtype,
                ),
            ],
            dim=0,
        )
        clf_in = torch.cat(
            [clf_scores, torch.full((extra,), float("-inf"), device=clf_scores.device, dtype=clf_scores.dtype)],
            dim=0,
        )
        if hit_scores is None:
            hit_score_in = torch.full((pad_len,), float("inf"), device=clf_scores.device, dtype=torch.float32)
        else:
            hit_score_in = torch.cat(
                [
                    hit_scores.float(),
                    torch.full((extra,), float("inf"), device=hit_scores.device, dtype=torch.float32),
                ],
                dim=0,
            )
        layer_in = torch.cat(
            [layers, torch.zeros((extra,), device=layers.device, dtype=layers.dtype)],
            dim=0,
        )
        coord_in = torch.cat(
            [
                coords,
                torch.zeros((extra, int(coords.shape[1])), device=coords.device, dtype=coords.dtype),
            ],
            dim=0,
        )
    else:
        predicted_in = predicted
        valid_in = valid_mask
        emb_in = normalized_emb
        clf_in = clf_scores
        hit_score_in = hit_scores.float() if hit_scores is not None else torch.full_like(
            clf_scores,
            float("inf"),
            dtype=torch.float32,
        )
        layer_in = layers
        coord_in = coords
    out = _codex_nomask_target_core_compiled(
        predicted_in,
        valid_in,
        emb_in,
        clf_in,
        layer_in,
        coord_in,
        hit_score_in,
        fixed_num_labels,
    )
    return out[:raw_len]
