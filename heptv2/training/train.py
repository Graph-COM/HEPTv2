"""Per-batch train/eval steps + single-GPU epoch loop.

Ported from `src/tracking_trainer.py`. Drops DDP reductions, wandb,
efficiency plots, GNN-reference eval, encoder_kmeans/encoder_dbscan/
decoder_assignments postprocessing modes, freeze/reinit branches.
"""
from collections import defaultdict
from copy import deepcopy
from inspect import signature
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch_scatter import scatter_mean
from tqdm import tqdm

from ..eval.metrics import compute_tracking_metrics_from_prediction
from ..eval.postprocess import decoder_kmeans_postprocess

_POSTPROCESS_PARAM_NAMES = {
    name
    for name in signature(decoder_kmeans_postprocess).parameters
    if name not in {"pred_list", "sub_events", "full_data", "full_emb", "full_clf_scores"}
}


def _as_list(obj):
    if obj is None:
        return []
    return list(obj) if isinstance(obj, (list, tuple)) else [obj]


def trim_pred(pred, raw_size):
    """Trim per-point fields (`pred_masks`, `clf_*`, `emb_*`) to `raw_size`
    along the point dimension. Also trims aux_outputs for deep supervision.
    """
    if "pred_masks" in pred:
        pm = pred["pred_masks"]
        pred["pred_masks"] = pm[:, :, :raw_size, :] if pm.dim() == 4 else pm[:, :, :raw_size]
    for aux in pred.get("aux_outputs", []):
        if "pred_masks" in aux:
            pm = aux["pred_masks"]
            aux["pred_masks"] = pm[:, :, :raw_size, :] if pm.dim() == 4 else pm[:, :, :raw_size]
        if "_serialized_point_order" in aux and aux["_serialized_point_order"] is not None:
            aux["_serialized_point_order"] = aux["_serialized_point_order"][:raw_size]
    for key in ("_serialized_point_order", "clf_probs", "clf_logits", "emb_all", "emb_all_labels"):
        if key in pred and pred[key] is not None:
            pred[key] = pred[key][:raw_size]
    return pred


def trim_pred_by_valid_mask(pred, valid_mask):
    """If the decoder returned a serialized point order, trim all per-point
    fields to the valid-only length; else index by mask.
    """
    valid_mask = valid_mask.bool()
    if pred.get("_serialized_point_order") is not None:
        raw_size = int(valid_mask.sum().item())
        return trim_pred(pred, raw_size)
    if "pred_masks" in pred:
        pm = pred["pred_masks"]
        pred["pred_masks"] = pm[:, :, valid_mask, :] if pm.dim() == 4 else pm[:, :, valid_mask]
    for aux in pred.get("aux_outputs", []):
        if "pred_masks" in aux:
            pm = aux["pred_masks"]
            aux["pred_masks"] = pm[:, :, valid_mask, :] if pm.dim() == 4 else pm[:, :, valid_mask]
    return pred


def trim_targets_by_valid_mask(targets, valid_mask):
    valid_mask = valid_mask.bool()
    for tgt in targets:
        if "masks" in tgt:
            tgt["masks"] = tgt["masks"][:, valid_mask]
    return targets


def weighted_loss(losses, criterion):
    for k in list(losses.keys()):
        if k in criterion.weight_dict:
            losses[k] = losses[k] * criterion.weight_dict[k]
        else:
            losses.pop(k)
    total = sum(losses.values())
    loss_items = {k: v.item() for k, v in losses.items()}
    return total, loss_items


def _float_tensors_for_loss(obj):
    if torch.is_tensor(obj):
        return obj.float() if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: _float_tensors_for_loss(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_float_tensors_for_loss(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_float_tensors_for_loss(v) for v in obj)
    return obj


def _grad_norm(parameters):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norms = [torch.linalg.vector_norm(g, 2) for g in grads]
    return torch.linalg.vector_norm(torch.stack(norms), 2)


def _tracking_metrics_cfg_from_config(config):
    eval_cfg = config.get("eval", {}) or {}
    pp_cfg = config.get("postprocessing", {}) or {}
    out = {
        "pt_thld": eval_cfg.get("pt_thld", 0.9),
        "predicted_count_thld": eval_cfg.get("predicted_count_thld", 3),
        "min_track_length": eval_cfg.get("min_track_length", 0),
    }
    out.update(pp_cfg)
    return out


def _postprocess_kwargs(metrics_cfg):
    return {k: v for k, v in metrics_cfg.items() if k in _POSTPROCESS_PARAM_NAMES}


@torch.no_grad()
def _metrics_from_raw_masks(raw_masks, full_data, sub_events, metrics_cfg):
    predicted = decoder_kmeans_postprocess(
        raw_masks,
        sub_events,
        full_data,
        full_emb=full_data.emb_all,
        full_clf_scores=full_data.clf_scores,
        **_postprocess_kwargs(metrics_cfg),
    )
    return compute_tracking_metrics_from_prediction(
        predicted,
        full_data,
        pt_thld=metrics_cfg.get("pt_thld", 0.9),
        predicted_count_thld=metrics_cfg.get("predicted_count_thld", 3),
        min_track_length=metrics_cfg.get("min_track_length", 0),
    )


def calc_clf_loss(pred, data, criterion):
    """Pos-weighted BCE on the encoder classifier head + AUROC/precision/recall."""
    y_pred = pred["clf_logits"].squeeze(-1)
    y_true = data.targets[0][0]["y_true"].float()
    pos_weight = y_true * (criterion.weight_dict["clf_pos_weight"] - 1.0) + 1.0
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=pos_weight)
    try:
        auroc = roc_auc_score(y_true.detach().cpu().numpy(), y_pred.sigmoid().detach().float().cpu().numpy())
    except ValueError:
        auroc = 0.0
    preds_bin = (y_pred.sigmoid() > 0.1).detach().float().cpu().numpy()
    y_np = y_true.detach().cpu().numpy()
    recall = recall_score(y_np, preds_bin, zero_division=0)
    precision = precision_score(y_np, preds_bin, zero_division=0)
    return loss * criterion.weight_dict["clf_loss"], float(auroc), float(recall), float(precision)


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


def calc_infonce_loss(embeddings, emb_all, labels, labels_all, y_true, criterion,
                     k=256, chunk_size=5000, temperature=0.07):
    """Chunked InfoNCE against all same-label positives + top-k hardest negatives,
    averaged per cluster and then across clusters. Mirrors src exactly.
    """
    device = embeddings.device
    N = embeddings.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=device)

    _, inv = torch.unique(labels, return_inverse=True)
    counts = torch.bincount(inv)
    cluster_size = counts[inv]
    valid_anchor = cluster_size > 1

    loss_per_anchor = torch.empty(N, device=device)
    pos_map = torch.nonzero(y_true, as_tuple=False).squeeze(1)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        M = end - start
        chunk_emb = embeddings[start:end]
        sim_chunk = torch.einsum("md,nd->mn", chunk_emb, emb_all) / temperature

        self_indices = pos_map[start:end]
        rows = torch.arange(M, device=device)
        sim_chunk[rows, self_indices] = float("-inf")

        pos_mask = labels[start:end].unsqueeze(1) == labels_all.unsqueeze(0)
        sim_pos = sim_chunk.masked_fill(~pos_mask, float("-inf"))
        log_pos = torch.logsumexp(sim_pos, dim=1)

        sim_neg = sim_chunk.masked_fill(pos_mask, float("-inf"))
        topk_neg, _ = sim_neg.topk(k, dim=1, largest=True)
        log_neg = torch.logsumexp(topk_neg, dim=1)

        log_den = torch.logsumexp(torch.stack([log_pos, log_neg], dim=1), dim=1)
        loss_per_anchor[start:end] = -(log_pos - log_den)

    if valid_anchor.sum() == 0:
        return torch.tensor(0.0, device=device)
    losses = loss_per_anchor[valid_anchor]
    lbls = labels[valid_anchor]
    _, inv_lbls = torch.unique(lbls, return_inverse=True)
    cluster_losses = scatter_mean(losses, inv_lbls, dim=0)
    return cluster_losses.mean() * criterion.weight_dict["nce_weight"]


def _forward_all_losses(model, data, criterion, amp_enabled, amp_dtype, train: bool,
                       collect_raw: bool = False):
    full_data, sub_events = data
    total_loss = 0.0
    loss_accum = defaultdict(list)
    raw_masks = [] if collect_raw else None

    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
        enc_out = model.encode(full_data)
        full_data.emb_all = enc_out["emb_all"].detach()
        full_data.encoder_features = enc_out["encoded"].detach()
        full_data.clf_scores = enc_out["clf_probs"].detach()

        for sub in sub_events:
            sub_feats = enc_out["encoded"][sub["all_idx"]]
            sub_coords = full_data.coords[sub["all_idx"]]
            pred = model.decode(sub_feats, sub_coords, sub["valid_mask"])
            pred = trim_pred_by_valid_mask(pred, sub["valid_mask"])
            sub_targets = trim_targets_by_valid_mask(deepcopy(sub["targets"][0]), sub["valid_mask"])

            with autocast(device_type="cuda", enabled=False):
                loss = criterion(_float_tensors_for_loss(pred), sub_targets)
                loss, loss_items = weighted_loss(loss, criterion)
            total_loss = total_loss + loss
            for k, v in loss_items.items():
                loss_accum[k].append(v)

            if collect_raw:
                raw_masks.append((
                    pred["pred_masks"].detach(),
                    pred["pred_logits"].detach(),
                    pred.get("_serialized_point_order", None),
                ))

        pred_full = {
            "clf_probs": enc_out["clf_probs"],
            "clf_logits": enc_out["clf_logits"],
            "emb_all": enc_out["emb_all"],
            "emb_all_labels": enc_out["emb_all_labels"],
        }
        y_true = full_data.targets[0][0]["y_true"]
        pred_full = trim_pred(pred_full, y_true.shape[0])
        pred_full = _float_tensors_for_loss(pred_full)

        if "clf_logits" in pred_full and criterion.weight_dict.get("clf_loss", 0.0) > 0:
            with autocast(device_type="cuda", enabled=False):
                clf_loss, auroc, recall, prec = calc_clf_loss(pred_full, full_data, criterion)
            total_loss = total_loss + clf_loss
            loss_accum["clf_loss"].append(clf_loss.item())
            loss_accum["roc"].append(auroc)
            loss_accum["rec"].append(recall)
            loss_accum["prec"].append(prec)

        infonce_inputs = build_infonce_inputs(pred_full, full_data) if criterion.weight_dict.get("nce_weight", 0.0) > 0 else None
        if infonce_inputs is not None:
            emb, emb_all, emb_labels, labels_all, y_true_nce = infonce_inputs
            with autocast(device_type="cuda", enabled=False):
                infonce_loss = calc_infonce_loss(emb, emb_all, emb_labels, labels_all, y_true_nce, criterion)
            total_loss = total_loss + infonce_loss
            loss_accum["infonce"].append(infonce_loss.item())

    if collect_raw:
        return total_loss, loss_accum, raw_masks
    return total_loss, loss_accum


def _add_l2sp_loss(model, total_loss, loss_accum, l2sp_refs, l2sp_weight):
    if not l2sp_refs or l2sp_weight <= 0:
        return total_loss
    terms = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ref = l2sp_refs.get(name)
        if ref is None and name.startswith("_orig_mod."):
            ref = l2sp_refs.get(name[len("_orig_mod."):])
        if ref is None:
            continue
        terms.append((p.float() - ref.float()).pow(2).mean())
    if not terms:
        return total_loss
    l2sp_loss = torch.stack(terms).mean() * float(l2sp_weight)
    loss_accum["l2sp_loss"].append(float(l2sp_loss.detach().item()))
    return total_loss + l2sp_loss


def train_one_batch(model, optimizer, criterion, data, lr_s, amp_enabled=False,
                   amp_dtype=torch.float16, scaler=None, metrics_cfg=None,
                   l2sp_refs=None, l2sp_weight=0.0):
    model.train()
    optimizers = _as_list(optimizer)
    for opt in optimizers:
        opt.zero_grad()

    collect_raw = metrics_cfg is not None
    result = _forward_all_losses(
        model, data, criterion, amp_enabled, amp_dtype, train=True,
        collect_raw=collect_raw,
    )
    if collect_raw:
        total_loss, loss_accum, raw_masks = result
    else:
        total_loss, loss_accum = result
    total_loss = _add_l2sp_loss(model, total_loss, loss_accum, l2sp_refs, l2sp_weight)

    if scaler is not None and scaler.is_enabled():
        scaler.scale(total_loss).backward()
        for opt in optimizers:
            scaler.unscale_(opt)
        params = list(model.parameters())
        max_norm = _grad_norm(params)
        if torch.isfinite(max_norm):
            max_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
    else:
        total_loss.backward()
        max_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

    if lr_s is not None:
        for sched in _as_list(lr_s):
            if isinstance(sched, LambdaLR):
                sched.step()

    loss_out = {k: float(np.mean(v)) for k, v in loss_accum.items()} if loss_accum else {}
    loss_out["total_loss"] = float(total_loss.item()) if hasattr(total_loss, "item") else float(total_loss)
    loss_out["max_norm"] = float(max_norm.item())
    if collect_raw:
        full_data, sub_events = data
        metrics = _metrics_from_raw_masks(raw_masks, full_data, sub_events, metrics_cfg)
        for k, v in metrics.items():
            if isinstance(v, float) and np.isnan(v):
                continue
            loss_out[k] = float(v)
    return loss_out


@torch.no_grad()
def eval_one_batch(model, optimizer, criterion, data, lr_s, amp_enabled=False,
                  amp_dtype=torch.float16, scaler=None, metrics_cfg=None):
    """Eval forward pass. When `metrics_cfg` is provided, also runs
    decoder_kmeans postprocessing + tracking metrics and merges them into
    the returned dict alongside losses.
    """
    model.eval()
    collect_raw = metrics_cfg is not None
    result = _forward_all_losses(
        model, data, criterion, amp_enabled, amp_dtype, train=False,
        collect_raw=collect_raw,
    )
    if collect_raw:
        total_loss, loss_accum, raw_masks = result
    else:
        total_loss, loss_accum = result

    loss_out = {k: float(np.mean(v)) for k, v in loss_accum.items()} if loss_accum else {}
    loss_out["total_loss"] = float(total_loss.item()) if hasattr(total_loss, "item") else float(total_loss)
    loss_out["max_norm"] = 0.0

    if collect_raw:
        full_data, sub_events = data
        metrics = _metrics_from_raw_masks(raw_masks, full_data, sub_events, metrics_cfg)
        for k, v in metrics.items():
            if isinstance(v, float) and np.isnan(v):
                continue
            loss_out[k] = float(v)
    return loss_out


def run_one_epoch(model, optimizer, criterion, data_loader, phase, epoch, device,
                 lr_s, config, process_data_fn, amp_enabled=False, amp_dtype=torch.float16,
                 scaler=None, limit_batches: Optional[int] = None, metrics_cfg=None,
                 l2sp_refs=None, batch_end_callback=None):
    """One pass over the data loader. `process_data_fn(data)` returns `(full_data, sub_events)`
    so the caller controls the preprocess (train needs `build_targets=True`).
    Pass `limit_batches` to early-stop an epoch (useful for smoke tests).
    Pass `metrics_cfg` (only meaningful in eval phases) to also compute
    tracking metrics (dm, technical_efficiency, fake_rate, ...) per event.
    """
    torch.cuda.empty_cache()
    is_train = phase == "train"
    pbar = tqdm(data_loader, desc=f"[Epoch {epoch}] {phase}")
    totals = defaultdict(list)
    train_monitor = config.get("train_monitor", {}) or {}
    monitor_enabled = is_train and bool(train_monitor.get("compute_metrics", False))
    monitor_first = int(train_monitor.get("first_batches", 0))
    monitor_every = int(train_monitor.get("every_n_batches", 0))
    monitor_cfg = _tracking_metrics_cfg_from_config(config) if monitor_enabled else None
    probe_dm, probe_fake = [], []
    abort_min_probes = int(train_monitor.get("abort_min_probes", 4))
    abort_window = int(train_monitor.get("abort_window", 4))
    abort_dm_below = train_monitor.get("abort_avg_dm_below")
    abort_fake_above = train_monitor.get("abort_avg_fake_rate_above")
    l2sp_weight = float(config.get("finetune_l2sp_weight", 0.0) or 0.0)

    def should_probe(batch_idx):
        if not monitor_enabled:
            return False
        if batch_idx < monitor_first:
            return True
        return monitor_every > 0 and (batch_idx + 1) % monitor_every == 0

    for i, data in enumerate(pbar):
        if limit_batches is not None and i >= limit_batches:
            break
        data = data.clone().to(device)
        processed = process_data_fn(data)
        if is_train:
            batch_metrics_cfg = monitor_cfg if should_probe(i) else None
            out = train_one_batch(
                model, optimizer, criterion, processed, lr_s,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
                metrics_cfg=batch_metrics_cfg,
                l2sp_refs=l2sp_refs, l2sp_weight=l2sp_weight,
            )
        else:
            out = eval_one_batch(
                model, optimizer, criterion, processed, lr_s,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
                metrics_cfg=metrics_cfg,
            )
        for k, v in out.items():
            totals[k].append(v)
        if is_train and "dm" in out and "fake_rate" in out:
            probe_dm.append(out["dm"])
            probe_fake.append(out["fake_rate"])
            if len(probe_dm) >= abort_min_probes:
                dm_avg = float(np.mean(probe_dm[-abort_window:]))
                fake_avg = float(np.mean(probe_fake[-abort_window:]))
                abort = False
                reasons = []
                if abort_dm_below is not None and dm_avg < float(abort_dm_below):
                    abort = True
                    reasons.append(f"dm_avg={dm_avg:.4f} < {float(abort_dm_below):.4f}")
                if abort_fake_above is not None and fake_avg > float(abort_fake_above):
                    abort = True
                    reasons.append(f"fake_avg={fake_avg:.4f} > {float(abort_fake_above):.4f}")
                if abort:
                    pbar.write(
                        f"[Epoch {epoch}] early abort after batch {i}: " + ", ".join(reasons)
                    )
                    totals["early_aborted"].append(1.0)
                    break
        pbar.set_postfix_str(
            ", ".join(f"{k}:{v:.3f}" for k, v in out.items() if not k.split("_")[-1].isdigit())
        )
        if batch_end_callback is not None:
            partial = {k: float(np.mean(v)) for k, v in totals.items()}
            batch_end_callback(i, partial)
    return {k: float(np.mean(v)) for k, v in totals.items()}
