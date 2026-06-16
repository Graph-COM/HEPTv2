import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from heptv2.data.dataset import TrackmlLarge, TrackingTransform
from heptv2.eval.metrics import compute_tracking_metrics_from_prediction
from heptv2.model import Transformer
from heptv2.run_inference import (
    _apply_fp16_include,
    _apply_torchao_quant,
    _load_checkpoint,
    _patch_dynamo_nvcc_probe,
    predict_compact_event,
)
from heptv2.utils.config_utils import apply_overrides


def _avg(metrics):
    keys = metrics[0].keys()
    return {k: float(np.nanmean([m[k] for m in metrics])) for k in keys}


def _metric_line(prefix, m):
    return (
        f"{prefix} "
        f"dm={m['dm']:.8f} "
        f"fake_rate={m['fake_rate']:.8f} "
        f"fake_double_majority={m['fake_double_majority']:.8f} "
        f"te={m['technical_efficiency']:.8f} "
        f"dup={m['dup_rate']:.8f} "
        f"clusters={m['n_cleaned_clusters']:.4f}"
    )


def _cluster_df(pred, full_data, pt_thld, predicted_count_thld, min_track_length, q_same):
    valid = full_data.valid_mask.bool().detach().cpu().numpy()
    labels = pred.detach().cpu().numpy()[valid].astype(np.int64)
    truth = full_data.particle_id.detach().cpu().numpy()[valid].astype(np.int64)
    pts = full_data.pt.detach().cpu().numpy()[valid]
    reco = full_data.reconstructable.detach().cpu().numpy()[valid].astype(bool)
    layer = full_data.layer.detach().cpu().numpy()[valid].astype(np.int64)
    coords = full_data.coords.detach().float().cpu().numpy()[valid]
    clf = full_data.clf_scores.detach().float().cpu().numpy()[valid].reshape(-1)
    emb = full_data.emb_all.detach().float().cpu().numpy()[valid]

    pid_hit_counts = Counter(truth.tolist())
    rows = []
    by_cluster = defaultdict(list)
    for i, c in enumerate(labels.tolist()):
        if c >= 0:
            by_cluster[int(c)].append(i)

    for c, idx_list in by_cluster.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        size = int(idx.size)
        if size < int(predicted_count_thld):
            continue
        ids, counts = np.unique(truth[idx], return_counts=True)
        maj_j = int(np.argmax(counts))
        maj_pid = int(ids[maj_j])
        maj_hits = int(counts[maj_j])
        maj_pid_hits = int(pid_hit_counts[maj_pid])
        maj_frac = maj_hits / max(1, size)
        maj_pid_frac = maj_hits / max(1, maj_pid_hits)
        pid_mask = truth[idx] == maj_pid
        maj_pt = float(pts[idx][pid_mask].mean())
        maj_reco = bool(reco[idx][pid_mask].mean() >= 0.5)
        c_mask = (
            maj_pt >= float(pt_thld)
            and maj_reco
            and maj_pid_hits >= int(min_track_length)
        )
        double = c_mask and maj_frac > 0.5 and maj_pid_frac > 0.5
        e = emb[idx]
        cen = e.mean(axis=0)
        cen = cen / max(float(np.linalg.norm(cen)), 1e-12)
        sim = e @ cen
        dup_layers = size - int(np.unique(layer[idx]).size)
        r = coords[idx, 2].astype(np.float64)
        phi = coords[idx, 3].astype(np.float64)
        z = coords[idx, 4].astype(np.float64)
        order = np.argsort(r)
        r0, r1 = float(r[order[0]]), float(r[order[-1]])
        z0, z1 = float(z[order[0]]), float(z[order[-1]])
        p0, p1 = float(phi[order[0]]), float(phi[order[-1]])
        dr = r1 - r0
        if abs(dr) < 1e-12:
            t = np.zeros_like(r)
        else:
            t = (r - r0) / dr
        z_res = np.abs(z - (z0 + (z1 - z0) * t))
        dphi01 = ((p1 - p0 + 0.5) % 1.0) - 0.5
        dphi0 = ((phi - p0 + 0.5) % 1.0) - 0.5
        phi_res = np.abs(((dphi0 - dphi01 * t + 0.5) % 1.0) - 0.5)
        rows.append(
            {
                "label": c,
                "query": int(c % int(q_same)),
                "sub": int(c // int(q_same)),
                "size": size,
                "maj_pid": maj_pid,
                "maj_hits": maj_hits,
                "maj_pid_hits": maj_pid_hits,
                "maj_frac": maj_frac,
                "maj_pid_frac": maj_pid_frac,
                "c_mask": bool(c_mask),
                "double": bool(double),
                "fake_double": bool(c_mask and not double),
                "dup_layers": int(dup_layers),
                "clf_mean": float(clf[idx].mean()),
                "clf_min": float(clf[idx].min()),
                "sim_mean": float(sim.mean()),
                "sim_min": float(sim.min()),
                "sim_gap": float(sim.mean() - sim.min()),
                "z_res_mean": float(z_res.mean()),
                "z_res_max": float(z_res.max()),
                "phi_res_mean": float(phi_res.mean()),
                "phi_res_max": float(phi_res.max()),
                "r_min": float(coords[idx, 2].min()),
                "r_max": float(coords[idx, 2].max()),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["same_query_double"] = False
    df["same_query_larger"] = False
    df["same_pid_double"] = False
    df["pid_has_other_cluster"] = False
    double_queries = set(df.loc[df["double"], "query"].astype(int).tolist())
    double_pids = set(df.loc[df["double"], "maj_pid"].astype(int).tolist())
    query_max_size = df.groupby("query")["size"].max().to_dict()
    pid_cluster_counts = df.groupby("maj_pid")["label"].count().to_dict()
    df["same_query_double"] = df["query"].map(lambda q: int(q) in double_queries)
    df["same_query_larger"] = df.apply(
        lambda r: query_max_size.get(int(r["query"]), 0) > int(r["size"]),
        axis=1,
    )
    df["same_pid_double"] = df["maj_pid"].map(lambda p: int(p) in double_pids)
    df["pid_has_other_cluster"] = df["maj_pid"].map(lambda p: pid_cluster_counts.get(int(p), 0) > 1)
    return df


def _drop_labels(pred, labels):
    if len(labels) == 0:
        return pred
    out = pred.clone()
    for label in labels:
        out[out == int(label)] = -1
    return out


def _merge_labels(pred, mapping):
    if not mapping:
        return pred
    out = pred.clone()
    for src, dst in mapping.items():
        out[out == int(src)] = int(dst)
    return out


def _same_query_drop_labels(df, max_drop_size, min_target_size=3, require_larger=True, clf_below=None, sim_min_below=None):
    labels = []
    for _query, group in df.groupby("query"):
        group = group[group["c_mask"]]
        if group.empty:
            continue
        target = group.sort_values(["size", "clf_mean"], ascending=False).iloc[0]
        if int(target["size"]) < int(min_target_size):
            continue
        for _, row in group.iterrows():
            if int(row["label"]) == int(target["label"]):
                continue
            if int(row["size"]) > int(max_drop_size):
                continue
            if require_larger and int(target["size"]) <= int(row["size"]):
                continue
            if clf_below is not None and float(row["clf_mean"]) >= float(clf_below):
                continue
            if sim_min_below is not None and float(row["sim_min"]) >= float(sim_min_below):
                continue
            labels.append(int(row["label"]))
    return labels


def _same_query_merge_map(df, max_src_size, min_target_size=3, max_merged_size=14, clf_below=None, sim_min_below=None):
    valid = df[df["c_mask"]].copy()
    if valid.empty:
        return {}
    valid["_score"] = valid["size"].astype(float) * 1000.0 + valid["clf_mean"].astype(float)
    target_idx = valid.groupby("query")["_score"].idxmax()
    target = valid.loc[target_idx, ["query", "label", "size"]].rename(
        columns={"label": "target_label", "size": "target_size"}
    )
    merged = valid.merge(target, on="query", how="left")
    mask = (
        (merged["label"] != merged["target_label"])
        & (merged["size"] <= int(max_src_size))
        & (merged["target_size"] >= int(min_target_size))
        & (merged["target_size"] > merged["size"])
        & ((merged["target_size"] + merged["size"]) <= int(max_merged_size))
    )
    if clf_below is not None:
        mask &= merged["clf_mean"] < float(clf_below)
    if sim_min_below is not None:
        mask &= merged["sim_min"] < float(sim_min_below)
    rows = merged.loc[mask, ["label", "target_label"]]
    return {int(r.label): int(r.target_label) for r in rows.itertuples(index=False)}


@torch.inference_mode()
def _predict_one(model, data, cfg, device, amp_dtype):
    amp_enabled = bool(cfg.get("amp", {}).get("enabled", False))
    return predict_compact_event(
        model,
        data,
        cfg,
        device,
        amp_enabled,
        amp_dtype,
        build_targets=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--skip-rules", action="store_true")
    parser.add_argument("--save-csv", default="")
    args = parser.parse_args()

    _patch_dynamo_nvcc_probe()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.override)
    cfg["device"] = args.device
    cfg["checkpoint_path"] = args.checkpoint
    recompile_limit = int(os.environ.get("HEPTV2_DYNAMO_RECOMPILE_LIMIT", "0") or 0)
    if recompile_limit > 0:
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = recompile_limit
        dynamo_config.accumulated_recompile_limit = max(
            int(getattr(dynamo_config, "accumulated_recompile_limit", recompile_limit)),
            recompile_limit * 16,
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(int(cfg.get("num_threads", 4)))

    dataset = TrackmlLarge(root=Path(cfg["data_root"]), transform=TrackingTransform())
    subset = dataset[dataset.idx_split[args.split]]
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    model = Transformer(
        attn_type="hept",
        in_dim=dataset.x_dim,
        coords_dim=dataset.coords_dim,
        task=cfg["dataset_name"],
        **cfg["model_kwargs"],
    ).to(device).eval()
    _load_checkpoint(model, args.checkpoint, device)
    model = _apply_torchao_quant(model)
    model = _apply_fp16_include(model)
    if args.compile:
        if hasattr(model, "encode_online"):
            model.encode_online = torch.compile(model.encode_online, dynamic=False)
        else:
            model.encode = torch.compile(model.encode, dynamic=False)
        model.decode_assignment_inputs = torch.compile(model.decode_assignment_inputs, dynamic=False)

    amp_dtype = torch.float16 if str(cfg.get("amp", {}).get("dtype", "float16")) == "float16" else torch.bfloat16
    n = len(subset) if args.limit <= 0 else min(args.limit, len(subset))
    metrics = []
    rows = []
    preds = []
    full_datas = []
    q_same = int(cfg["model_kwargs"].get("num_queries", 3000))
    for i, data in enumerate(tqdm(loader, total=n, desc="fake-double-diagnose")):
        if i >= n:
            break
        pred, full_data = _predict_one(model, data, cfg, device, amp_dtype)
        metrics.append(
            compute_tracking_metrics_from_prediction(
                pred,
                full_data,
                pt_thld=cfg["eval"]["pt_thld"],
                predicted_count_thld=cfg["eval"]["predicted_count_thld"],
                min_track_length=cfg["eval"]["min_track_length"],
            )
        )
        df = _cluster_df(
            pred,
            full_data,
            cfg["eval"]["pt_thld"],
            cfg["eval"]["predicted_count_thld"],
            cfg["eval"]["min_track_length"],
            q_same,
        )
        if not df.empty:
            df.insert(0, "event", i)
            rows.append(df)
        preds.append(pred)
        full_datas.append(full_data)

    avg = _avg(metrics)
    print(_metric_line("metrics", avg))
    all_df = pd.concat(rows, ignore_index=True)
    if args.save_csv:
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(args.save_csv, index=False)
        print(f"saved_clusters={args.save_csv}")
    f = all_df[all_df["fake_double"]].copy()
    d = all_df[all_df["double"]].copy()
    print(f"clusters valid={int(all_df['c_mask'].sum())} double={len(d)} fake_double={len(f)}")
    print("\nsize breakdown fake_double:")
    print(f.groupby("size").size().sort_index().tail(20).to_string())
    print("\nreason proxies fake_double:")
    proxies = [
        "same_query_double",
        "same_query_larger",
        "same_pid_double",
        "pid_has_other_cluster",
        "dup_layers",
    ]
    for key in proxies:
        vals = f[key].astype(bool) if key != "dup_layers" else (f[key] > 0)
        print(f"{key:24s} count={int(vals.sum())} frac={float(vals.mean()):.4f}")
    print("\nfeature means double vs fake_double:")
    cols = ["size", "maj_frac", "maj_pid_frac", "clf_mean", "clf_min", "sim_mean", "sim_min", "sim_gap", "dup_layers"]
    print(pd.concat({"double": d[cols].mean(), "fake_double": f[cols].mean()}, axis=1).to_string())
    print("\nfake_double by flags:")
    print(
        f.groupby(["same_query_double", "same_query_larger", "same_pid_double", "pid_has_other_cluster"])
        .size()
        .sort_values(ascending=False)
        .head(20)
        .to_string()
    )
    if args.skip_rules:
        return

    event_dfs = {int(df["event"].iloc[0]): df.drop(columns=["event"]) for df in rows}
    candidates = []
    for max_size in [3, 4, 5, 6]:
        for clf_thr in [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]:
            candidates.append((f"drop_s{max_size}_clf{clf_thr:g}", "drop", {"max_size": max_size, "clf_thr": clf_thr}))
        for sim_thr in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            candidates.append((f"drop_s{max_size}_simmin{sim_thr:g}", "drop_sim", {"max_size": max_size, "sim_thr": sim_thr}))
        for gap_thr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            candidates.append((f"drop_s{max_size}_simgap{gap_thr:g}", "drop_gap", {"max_size": max_size, "gap_thr": gap_thr}))
        # Same-query rules are evaluated separately; the simple threshold
        # rules below are vectorized and cheap enough for full-test sweeps.
        candidates.append((f"sameq_larger_s{max_size}", "combo", {"max_size": max_size, "sameq_larger": True}))
        candidates.append((f"duplayer_s{max_size}", "combo", {"max_size": max_size, "dup": True}))
        for clf_thr in [0.20, 0.35, 0.50, 0.65, 0.80]:
            candidates.append((f"sameq_larger_s{max_size}_clf{clf_thr:g}", "combo", {"max_size": max_size, "sameq_larger": True, "clf_thr": clf_thr}))
            candidates.append((f"duplayer_s{max_size}_clf{clf_thr:g}", "combo", {"max_size": max_size, "dup": True, "clf_thr": clf_thr}))
        for sim_thr in [0.75, 0.80, 0.85, 0.90]:
            candidates.append((f"sameq_larger_s{max_size}_sim{sim_thr:g}", "combo", {"max_size": max_size, "sameq_larger": True, "sim_thr": sim_thr}))
            candidates.append((f"duplayer_s{max_size}_sim{sim_thr:g}", "combo", {"max_size": max_size, "dup": True, "sim_thr": sim_thr}))
        for gap_thr in [0.06, 0.08, 0.10, 0.12, 0.15]:
            candidates.append((f"sameq_larger_s{max_size}_gap{gap_thr:g}", "combo", {"max_size": max_size, "sameq_larger": True, "gap_thr": gap_thr}))
            candidates.append((f"duplayer_s{max_size}_gap{gap_thr:g}", "combo", {"max_size": max_size, "dup": True, "gap_thr": gap_thr}))

    def _drop_mask(df_i, kind, params):
        if kind == "drop":
            return (
                df_i["c_mask"]
                & (df_i["size"] <= int(params["max_size"]))
                & (df_i["clf_mean"] < float(params["clf_thr"]))
            )
        if kind == "drop_sim":
            return (
                df_i["c_mask"]
                & (df_i["size"] <= int(params["max_size"]))
                & (df_i["sim_min"] < float(params["sim_thr"]))
            )
        if kind == "drop_gap":
            return (
                df_i["c_mask"]
                & (df_i["size"] <= int(params["max_size"]))
                & (df_i["sim_gap"] > float(params["gap_thr"]))
            )
        if kind == "sameq_drop":
            labels = set(
                _same_query_drop_labels(
                    df_i,
                    int(params["max_size"]),
                    clf_below=params["clf_thr"],
                )
            )
            return df_i["label"].map(lambda x: int(x) in labels)
        if kind == "combo":
            mask = df_i["c_mask"] & (df_i["size"] <= int(params["max_size"]))
            if params.get("sameq_larger"):
                mask = mask & df_i["same_query_larger"]
            if params.get("dup"):
                mask = mask & (df_i["dup_layers"] > 0)
            if "clf_thr" in params:
                mask = mask & (df_i["clf_mean"] < float(params["clf_thr"]))
            if "sim_thr" in params:
                mask = mask & (df_i["sim_min"] < float(params["sim_thr"]))
            if "gap_thr" in params:
                mask = mask & (df_i["sim_gap"] > float(params["gap_thr"]))
            return mask
        raise ValueError(kind)

    scored = []
    for name, kind, params in candidates:
        dm_vals = []
        fdm_vals = []
        cluster_vals = []
        drops = 0
        drop_double = 0
        for i in range(len(metrics)):
            df_i = event_dfs.get(i)
            if df_i is None or df_i.empty:
                dm_vals.append(metrics[i]["dm"])
                fdm_vals.append(metrics[i]["fake_double_majority"])
                cluster_vals.append(metrics[i]["n_cleaned_clusters"])
                continue
            drop = _drop_mask(df_i, kind, params)
            drop_c = drop & df_i["c_mask"]
            drop_d = drop & df_i["double"]
            drops += int(drop_c.sum())
            drop_double += int(drop_d.sum())
            n_particles = max(1.0, float(metrics[i]["n_particles"]))
            base_clusters = int(df_i["c_mask"].sum())
            base_double = int(df_i["double"].sum())
            new_clusters = base_clusters - int(drop_c.sum())
            new_double = base_double - int(drop_d.sum())
            dm_vals.append(new_double / n_particles)
            fdm_vals.append((new_clusters - new_double) / n_particles)
            cluster_vals.append(float(new_clusters))
        m = {
            "dm": float(np.nanmean(dm_vals)),
            "fake_double_majority": float(np.nanmean(fdm_vals)),
            "fake_rate": float("nan"),
            "technical_efficiency": float("nan"),
            "n_cleaned_clusters": float(np.nanmean(cluster_vals)),
        }
        ok = (m["dm"] >= 0.9855) and (m["fake_double_majority"] < 0.0075)
        objective = m["dm"] - 10.0 * max(0.0, m["fake_double_majority"] - 0.0075)
        scored.append((ok, objective, m["dm"], m["fake_double_majority"], drop_double / max(1, n), name, drops / max(1, n)))
    scored.sort(reverse=True)
    print("\nrule sweep top:")
    for ok, _obj, dm, fdm, drop_double_evt, name, drop_evt in scored[:40]:
        print(f"{name:32s} ok={int(ok)} dm={dm:.8f} fdm={fdm:.8f} drop_evt={drop_evt:.2f} drop_double_evt={drop_double_evt:.2f}")

    merge_specs = []
    for max_src in [3, 4, 5, 6]:
        merge_specs.append((f"sameq_merge_s{max_src}", max_src, None, None))
        for clf_thr in [0.50, 0.65, 0.80]:
            merge_specs.append((f"sameq_merge_s{max_src}_clf{clf_thr:g}", max_src, clf_thr, None))
        for sim_thr in [0.80, 0.85, 0.90]:
            merge_specs.append((f"sameq_merge_s{max_src}_sim{sim_thr:g}", max_src, None, sim_thr))
    merge_results = []
    for name, max_src, clf_thr, sim_thr in tqdm(merge_specs, desc="same-query merge rules"):
        cand_metrics = []
        remap_count = 0
        for i, (pred, full_data) in enumerate(zip(preds, full_datas)):
            df_i = event_dfs.get(i)
            if df_i is None or df_i.empty:
                out = pred
            else:
                mapping = _same_query_merge_map(
                    df_i,
                    max_src,
                    clf_below=clf_thr,
                    sim_min_below=sim_thr,
                )
                remap_count += len(mapping)
                out = _merge_labels(pred, mapping)
            cand_metrics.append(
                compute_tracking_metrics_from_prediction(
                    out,
                    full_data,
                    pt_thld=cfg["eval"]["pt_thld"],
                    predicted_count_thld=cfg["eval"]["predicted_count_thld"],
                    min_track_length=cfg["eval"]["min_track_length"],
                )
            )
        m = _avg(cand_metrics)
        ok = (m["dm"] >= 0.9855) and (m["fake_double_majority"] < 0.0075)
        objective = m["dm"] - 10.0 * max(0.0, m["fake_double_majority"] - 0.0075)
        merge_results.append((ok, objective, m["dm"], m["fake_double_majority"], m["fake_rate"], m["technical_efficiency"], name, remap_count / max(1, n)))
    merge_results.sort(reverse=True)
    print("\nsame-query merge top:")
    for ok, _obj, dm, fdm, fake, te, name, remap_evt in merge_results[:30]:
        print(f"{name:32s} ok={int(ok)} dm={dm:.8f} fdm={fdm:.8f} fake={fake:.8f} te={te:.8f} remap_evt={remap_evt:.2f}")


if __name__ == "__main__":
    main()
