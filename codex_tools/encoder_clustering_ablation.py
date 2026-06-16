"""Encoder-only clustering ablation for HEPTv2.

Runs the best checkpoint through the encoder only, then clusters encoder
embeddings with DBSCAN or KMeans. Hyperparameters are selected on the valid
split and then evaluated once on test.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.cluster import DBSCAN
from torch.amp import autocast
from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from heptv2.data.dataset import TrackingTransform, TrackmlLarge
from heptv2.data.preprocess import process_inference_data
from heptv2.eval.metrics import compute_tracking_metrics_from_prediction
from heptv2.model import Transformer
from heptv2.run_inference import (
    _compile_kwargs_from_mode,
    _load_checkpoint,
    _parse_pad_bucket_lengths,
    _patch_dynamo_nvcc_probe,
)
from heptv2.utils.block_size_utils import resolve_block_sizes
from heptv2.utils.config_utils import apply_overrides


def _load_config(path: str, overrides: list[str]):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, overrides)
    return cfg


def _metric_stub(event):
    return SimpleNamespace(
        valid_mask=torch.ones_like(event["truth"], dtype=torch.bool),
        particle_id=event["truth"],
        pt=event["pt"],
        reconstructable=event["reconstructable"],
    )


def _event_metrics(predicted: torch.Tensor, event, cfg):
    return compute_tracking_metrics_from_prediction(
        predicted.to(event["truth"].device),
        _metric_stub(event),
        pt_thld=cfg["eval"].get("pt_thld", 0.9),
        predicted_count_thld=cfg["eval"].get("predicted_count_thld", 3),
        min_track_length=cfg["eval"].get("min_track_length", 0),
    )


def _mean_metrics(rows):
    keys = sorted({k for row in rows for k in row})
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        if vals:
            out[key] = float(np.mean(vals))
    return out


@torch.inference_mode()
def _encode_split(model, dataset, indices, cfg, device, amp_enabled, amp_dtype, split, limit_events=None):
    enc_cfg = cfg["model_kwargs"]
    enc_block, dec_block = resolve_block_sizes(enc_cfg)
    subset = dataset[indices]
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    events = []
    pbar = tqdm(loader, desc=f"encode-{split}", dynamic_ncols=True)
    for i, data in enumerate(pbar):
        if limit_events is not None and i >= int(limit_events):
            break
        data = data.clone().to(device)
        full_data, _ = process_inference_data(
            data,
            fixed_len=enc_cfg["fixed_seq_len"],
            encoder_block_size=enc_block,
            decoder_block_size=dec_block,
            regions=model.helper_funcs["regions"],
            num_sub_events=cfg["eval"].get("num_sub_events", 3),
            decoder_overlap=enc_cfg.get("decoder_overlap", False),
            eta_abs_max=cfg["eval"]["eta_abs_max"],
            return_sub_events=False,
            dynamic_pad=bool(cfg.get("inference_path", {}).get("dynamic_pad", False)),
            pad_bucket_lengths=_parse_pad_bucket_lengths(cfg.get("inference_path", {}).get("pad_buckets")),
            match_fixed_region_bins=bool(cfg.get("inference_path", {}).get("match_fixed_region_bins", False)),
            build_targets=True,
        )
        valid = full_data.valid_mask.bool()
        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            enc_out = model.encode_online(full_data, need_clf=True, need_emb=True)
        emb = enc_out["emb_all"][valid].detach().float().cpu()
        clf = enc_out["clf_probs"][valid].detach().squeeze(-1).float().cpu()
        events.append(
            {
                "emb": emb,
                "clf": clf,
                "truth": full_data.particle_id[valid].detach().cpu(),
                "pt": full_data.pt[valid].detach().cpu(),
                "reconstructable": full_data.reconstructable[valid].detach().cpu(),
            }
        )
        pbar.set_postfix(n=int(emb.shape[0]), clf=float(clf.mean()))
    return events


def _active_mask(clf: torch.Tensor, threshold: float, max_active: int = 0) -> torch.Tensor:
    mask = clf >= float(threshold)
    if max_active and int(mask.sum()) > int(max_active):
        top = torch.topk(clf, int(max_active), largest=True).indices
        capped = torch.zeros_like(mask)
        capped[top] = True
        mask = mask & capped
    return mask


@torch.no_grad()
def _kmeans_predict(emb, clf, device, threshold, hits_per_cluster, iters, chunk_size=4096):
    active = _active_mask(clf, threshold).to(device)
    predicted = torch.full((emb.shape[0],), -1, device=device, dtype=torch.long)
    if not bool(active.any()):
        return predicted.cpu()
    active_idx = active.nonzero(as_tuple=False).flatten()
    points = F.normalize(emb.to(device=device, dtype=torch.float32)[active_idx], p=2, dim=1, eps=1e-12)
    n = int(points.shape[0])
    k = max(1, int(round(n / float(hits_per_cluster))))
    k = min(k, n)
    if k == 1:
        predicted[active_idx] = 0
        return predicted.cpu()
    init = torch.div(torch.arange(k, device=device) * n, k, rounding_mode="floor")
    centroids = points.index_select(0, init).contiguous()
    assign = torch.zeros(n, device=device, dtype=torch.long)
    for _ in range(int(iters)):
        parts = []
        for start in range(0, n, int(chunk_size)):
            sim = points[start : start + int(chunk_size)] @ centroids.t()
            parts.append(sim.argmax(dim=1))
        assign = torch.cat(parts, dim=0)
        new_centroids = scatter_mean(points, assign, dim=0, dim_size=k)
        counts = torch.bincount(assign, minlength=k).to(points.dtype).unsqueeze(1)
        new_centroids = F.normalize(new_centroids, p=2, dim=1, eps=1e-12)
        centroids = torch.where(counts > 0, new_centroids, centroids)
    predicted[active_idx] = assign
    return predicted.cpu()


def _dbscan_predict(emb, clf, threshold, eps, min_samples, max_active):
    active = _active_mask(clf, threshold, max_active=max_active)
    predicted = torch.full((emb.shape[0],), -1, dtype=torch.long)
    if not bool(active.any()):
        return predicted
    active_idx = active.nonzero(as_tuple=False).flatten()
    points = F.normalize(emb[active_idx].float(), p=2, dim=1, eps=1e-12).numpy()
    labels = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="euclidean", n_jobs=1).fit_predict(points)
    predicted[active_idx] = torch.as_tensor(labels, dtype=torch.long)
    return predicted


def _evaluate_params(events, method, params, cfg, device):
    rows = []
    for event in tqdm(events, desc=f"{method}:{params}", leave=False, dynamic_ncols=True):
        if method == "kmeans":
            pred = _kmeans_predict(
                event["emb"],
                event["clf"],
                device,
                threshold=params["threshold"],
                hits_per_cluster=params["hits_per_cluster"],
                iters=params["iters"],
            )
        elif method == "dbscan":
            pred = _dbscan_predict(
                event["emb"],
                event["clf"],
                threshold=params["threshold"],
                eps=params["eps"],
                min_samples=params["min_samples"],
                max_active=params["max_active"],
            )
        else:
            raise ValueError(method)
        rows.append(_event_metrics(pred, event, cfg))
    return _mean_metrics(rows)


def _write_rows(path: Path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="heptv2/configs/finetune_phase48_scatter_amp.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default="codex_logs/decoder_ablation_0515/encoder_clustering")
    parser.add_argument("--limit-events", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dbscan-max-active", type=int, default=20000)
    args = parser.parse_args()

    cfg = _load_config(args.config, args.override)
    cfg["device"] = args.device
    cfg["split"] = "valid"
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(int(cfg.get("num_threads", 4)))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_enabled = bool(cfg.get("amp", {}).get("enabled", False))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.get("amp", {}).get("dtype", "float16")]

    dataset = TrackmlLarge(root=Path(cfg["data_root"]), transform=TrackingTransform())
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
    compile_mode = os.environ.get("HEPTV2_CODEX_COMPILE_MODE", "").strip()
    model.encode_online = torch.compile(model.encode_online, **_compile_kwargs_from_mode(compile_mode))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limit_events = 8 if args.quick and args.limit_events is None else args.limit_events

    valid_events = _encode_split(model, dataset, dataset.idx_split["valid"], cfg, device, amp_enabled, amp_dtype, "valid", limit_events)
    test_events = _encode_split(model, dataset, dataset.idx_split["test"], cfg, device, amp_enabled, amp_dtype, "test", limit_events)

    if args.quick:
        kmeans_grid = [
            {"threshold": t, "hits_per_cluster": h, "iters": it}
            for t, h, it in itertools.product([0.01, 0.05], [6, 8], [2])
        ]
        dbscan_grid = [
            {"threshold": t, "eps": e, "min_samples": m, "max_active": args.dbscan_max_active}
            for t, e, m in itertools.product([0.05, 0.10], [0.45, 0.55], [1])
        ]
    else:
        kmeans_grid = [
            {"threshold": t, "hits_per_cluster": h, "iters": it}
            for t, h, it in itertools.product([0.005, 0.01, 0.02, 0.05, 0.10], [5, 6, 7, 8, 10, 12], [2, 4])
        ]
        dbscan_grid = [
            {"threshold": t, "eps": e, "min_samples": m, "max_active": args.dbscan_max_active}
            for t, e, m in itertools.product([0.02, 0.05, 0.10, 0.20], [0.35, 0.45, 0.55, 0.65], [1, 2])
        ]

    all_rows = []
    selected = []
    for method, grid in (("kmeans", kmeans_grid), ("dbscan", dbscan_grid)):
        method_rows = []
        for params in grid:
            metrics = _evaluate_params(valid_events, method, params, cfg, device)
            row = {"method": method, "split": "valid", **params, **metrics}
            method_rows.append(row)
            all_rows.append(row)
            print(row, flush=True)
        best = max(method_rows, key=lambda row: row.get("dm", float("-inf")))
        test_metrics = _evaluate_params(test_events, method, {k: best[k] for k in grid[0]}, cfg, device)
        test_row = {
            "method": method,
            "split": "test",
            **{k: best[k] for k in grid[0]},
            **test_metrics,
            "selected_by_valid_dm": best.get("dm", float("nan")),
        }
        selected.append(test_row)
        all_rows.append(test_row)
        print(f"SELECTED {test_row}", flush=True)

    _write_rows(out_dir / "encoder_clustering_sweep.csv", all_rows)
    _write_rows(out_dir / "encoder_clustering_selected.csv", selected)


if __name__ == "__main__":
    main()
