"""Benchmark HEPTv2 latency scaling on synthetic large events.

This reproduces the 50k--500k scaling protocol used for the paper figure:
take one real event as a base, repeat/truncate its eta-filtered hits to target
input sizes, run a fixed number of warmup iterations, then summarize measured
iterations. The timed path calls ``run_timed_event`` from
``heptv2.benchmark_inference_latency`` so the preprocessing/model/postprocess
stages match the normal latency benchmark.

Examples:
    python heptv2/scripts/benchmark_scalability_latency.py \\
        --config heptv2/configs/finetune_phase48_scatter_amp.yaml \\
        --device cuda:0 \\
        --base-dataset-index 1846 \\
        --export-base-event codex_artifacts/scaling/base_event_29946_eta4.pt

    python heptv2/scripts/benchmark_scalability_latency.py \\
        --config heptv2/configs/finetune_phase48_scatter_amp.yaml \\
        --device cuda:0 \\
        --base-event codex_artifacts/scaling/base_event_29946_eta4.pt
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Data

from heptv2.benchmark_inference_latency import (
    _CompactPostprocessCUDAGraphCache,
    load_config,
    run_timed_event,
)
from heptv2.data.dataset import TrackingTransform, TrackmlLarge
from heptv2.model import Transformer
from heptv2.run_inference import (
    _apply_fp16_include,
    _apply_torchao_quant,
    _compact_assignment_active,
    _compile_kwargs_from_mode,
    _load_checkpoint,
    _patch_dynamo_nvcc_probe,
)
from heptv2.utils.config_utils import apply_overrides


DEFAULT_TARGETS = [50_000, 100_000, 200_000, 300_000, 400_000, 500_000]
PER_HIT_ATTRS = ["x", "pos", "coords", "layer", "particle_id", "pt", "reconstructable", "eta", "phi"]
STAGE_FIELDS = ["pre_ms", "encode_ms", "decode_ms", "model_ms", "post_ms", "total_ms"]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_base_from_dataset(cfg: dict, base_dataset_index: int) -> dict:
    dataset = TrackmlLarge(root=Path(cfg["data_root"]), transform=TrackingTransform())
    base = dataset[base_dataset_index]
    eta = base.eta if hasattr(base, "eta") and base.eta is not None else base.pos[:, 0]
    keep = (eta.abs() <= float(cfg["eval"].get("eta_abs_max", 4.0))).cpu()
    attrs = {}
    for name in PER_HIT_ATTRS:
        if not hasattr(base, name):
            continue
        val = getattr(base, name)
        if val is None or not torch.is_tensor(val) or val.shape[0] != base.x.shape[0]:
            continue
        attrs[name] = val[keep].cpu().contiguous()
    base_evtid = int(base.evtid[0]) if hasattr(base, "evtid") else int(base_dataset_index)
    return {
        "attrs": attrs,
        "base_evtid": base_evtid,
        "base_dataset_index": int(base_dataset_index),
        "base_kept_hits": int(keep.sum()),
        "x_dim": int(dataset.x_dim),
        "coords_dim": int(dataset.coords_dim),
    }


def _load_base_event(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "attrs" not in payload:
        raise ValueError(f"{path} is not a benchmark base-event payload")
    return payload


def _make_synth(base_payload: dict, target_n: int) -> Data:
    attrs = base_payload["attrs"]
    base_n = int(base_payload["base_kept_hits"])
    idx = torch.arange(target_n, dtype=torch.long) % base_n
    kwargs = {}
    for name, src in attrs.items():
        kwargs[name] = src.index_select(0, idx).contiguous()
    kwargs["hit_id"] = torch.arange(target_n, dtype=torch.long)
    kwargs["evtid"] = torch.tensor(
        [int(base_payload["base_evtid"]) * 1_000_000 + int(target_n)],
        dtype=torch.long,
    )
    data = Data(**kwargs)
    data.edge_index = None
    return data


def _setup_torch(cfg: dict, device: torch.device) -> tuple[bool, torch.dtype]:
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
    torch.manual_seed(int(cfg.get("seed", 42)))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    amp_enabled = bool(cfg.get("amp", {}).get("enabled", False))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg.get("amp", {}).get("dtype", "bfloat16")
    ]
    return amp_enabled, amp_dtype


def _build_model(cfg: dict, base_payload: dict, device: torch.device) -> tuple[Transformer, bool, object]:
    model = Transformer(
        attn_type="hept",
        in_dim=int(base_payload.get("x_dim", 15)),
        coords_dim=int(base_payload.get("coords_dim", 6)),
        task=cfg["dataset_name"],
        **cfg["model_kwargs"],
    ).to(device)
    model.eval()
    _load_checkpoint(model, cfg["checkpoint_path"], device)
    model = _apply_torchao_quant(model)
    model = _apply_fp16_include(model)
    _patch_dynamo_nvcc_probe()

    compile_mode = os.environ.get("HEPTV2_CODEX_COMPILE_MODE", "default")
    decode_compile_mode = os.environ.get("HEPTV2_CODEX_DECODE_COMPILE_MODE", compile_mode).strip()
    compile_kwargs = _compile_kwargs_from_mode(compile_mode)
    decode_compile_kwargs = _compile_kwargs_from_mode(decode_compile_mode)
    if compile_mode != "default":
        compile_kwargs["mode"] = compile_mode
    if decode_compile_mode and decode_compile_mode != "default":
        decode_compile_kwargs["mode"] = decode_compile_mode

    if hasattr(model, "encode_online"):
        model.encode_online = torch.compile(model.encode_online, **compile_kwargs)
    else:
        model.encode = torch.compile(model.encode, **compile_kwargs)

    pp_cfg = cfg.get("postprocessing", {})
    compact_assignment = _compact_assignment_active(cfg, model)
    compile_batched_decode = bool(pp_cfg.get("decoder_compile_batched_decode", False))
    if compact_assignment and compile_batched_decode:
        model.decode_assignment_inputs = torch.compile(model.decode_assignment_inputs, **decode_compile_kwargs)
    elif compile_batched_decode or not pp_cfg.get("decoder_batched_decode", False):
        model.decode = torch.compile(model.decode, **decode_compile_kwargs)

    post_graph_cache = (
        _CompactPostprocessCUDAGraphCache(cfg)
        if compact_assignment and bool(int(os.environ.get("HEPTV2_POST_CUDAGRAPH", "0")))
        else None
    )
    return model, compact_assignment, post_graph_cache


def _summary_row(target_n: int, rows: list[dict], args: argparse.Namespace, base_payload: dict) -> dict:
    arr = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in STAGE_FIELDS}
    prepost = arr["pre_ms"] + arr["post_ms"]
    row = {
        "target_n": int(target_n),
        "status": "ok",
        "error": "",
        "warmup_repeats": int(args.warmup),
        "measure_repeats": int(args.repeats),
        "base_evtid": int(base_payload["base_evtid"]),
        "base_kept_hits": int(base_payload["base_kept_hits"]),
        "padded_n": int(rows[0]["padded_n"]),
        "assigned_mean": float(np.mean([r["assigned"] for r in rows])),
        "prepost_ms_mean": float(prepost.mean()),
        "prepost_ms_p50": float(np.percentile(prepost, 50)),
        "prepost_ms_p90": float(np.percentile(prepost, 90)),
        "peak_allocated_mb": float(max(r["peak_allocated_mb"] for r in rows)),
        "peak_reserved_mb": float(max(r["peak_reserved_mb"] for r in rows)),
    }
    for key in STAGE_FIELDS:
        row[f"{key}_mean"] = float(arr[key].mean())
        row[f"{key}_p50"] = float(np.percentile(arr[key], 50))
        row[f"{key}_p90"] = float(np.percentile(arr[key], 90))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("heptv2/configs/finetune_phase48_scatter_amp.yaml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--base-event", type=Path, default=None)
    parser.add_argument("--base-dataset-index", type=int, default=1846)
    parser.add_argument("--export-base-event", type=Path, default=None)
    parser.add_argument("--export-base-event-only", action="store_true")
    parser.add_argument("--targets", type=str, default=",".join(str(x) for x in DEFAULT_TARGETS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("codex_logs/scalability_50k_500k_10warmup_20measure"))
    args = parser.parse_args()

    targets = _parse_int_list(args.targets)
    cfg = load_config(args.config)
    overrides = []
    if args.device:
        overrides.append(f"device={args.device}")
    apply_overrides(cfg, overrides)
    path_cfg = cfg.setdefault("inference_path", {})
    path_cfg["dynamic_pad"] = True
    path_cfg["pad_buckets"] = sorted(set(int(x) for x in list(path_cfg.get("pad_buckets") or []) + targets))
    pad_bucket_lengths = [int(x) for x in path_cfg["pad_buckets"]]

    if args.base_event is not None:
        base_payload = _load_base_event(args.base_event)
    else:
        base_payload = _load_base_from_dataset(cfg, args.base_dataset_index)
    if args.export_base_event is not None:
        args.export_base_event.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base_payload, args.export_base_event)
        print(f"exported base event: {args.export_base_event}")
        if args.export_base_event_only:
            return

    device = torch.device(cfg["device"])
    amp_enabled, amp_dtype = _setup_torch(cfg, device)
    model, compact_assignment, post_graph_cache = _build_model(cfg, base_payload, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "detail.csv"
    summary_path = args.out_dir / "summary.csv"
    effective_path = args.out_dir / "effective_config.yaml"
    with effective_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    detail_fields = [
        "target_n",
        "repeat_idx",
        "warmup",
        "status",
        "error",
        "padded_n",
        "assigned",
        *STAGE_FIELDS,
        "peak_allocated_mb",
        "peak_reserved_mb",
    ]
    summary_fields = [
        "target_n",
        "status",
        "error",
        "warmup_repeats",
        "measure_repeats",
        "base_evtid",
        "base_kept_hits",
        "padded_n",
        "assigned_mean",
    ]
    for key in STAGE_FIELDS:
        summary_fields += [f"{key}_mean", f"{key}_p50", f"{key}_p90"]
    summary_fields += ["prepost_ms_mean", "prepost_ms_p50", "prepost_ms_p90", "peak_allocated_mb", "peak_reserved_mb"]

    print(
        f"base_evtid={base_payload['base_evtid']} base_kept_hits={base_payload['base_kept_hits']} "
        f"targets={targets} warmup={args.warmup} repeats={args.repeats}",
        flush=True,
    )
    with detail_path.open("w", newline="") as df, summary_path.open("w", newline="") as sf:
        detail_writer = csv.DictWriter(df, fieldnames=detail_fields)
        summary_writer = csv.DictWriter(sf, fieldnames=summary_fields)
        detail_writer.writeheader()
        summary_writer.writeheader()

        for target_n in targets:
            print(f"\n[target {target_n}] building synthetic event", flush=True)
            data_gpu = _make_synth(base_payload, target_n).to(device)
            rows: list[dict] = []
            for rep in range(-int(args.warmup), int(args.repeats)):
                warmup = rep < 0
                if rep == 0 and device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                timed = run_timed_event(
                    model,
                    data_gpu,
                    cfg,
                    amp_enabled,
                    amp_dtype,
                    "minimal",
                    materialize_decode_output=False,
                    compact_assignment=compact_assignment,
                    dynamic_pad=True,
                    pad_bucket_lengths=pad_bucket_lengths,
                    lazy_embedding=False,
                    post_graph_cache=post_graph_cache,
                )
                rec = {
                    "target_n": int(target_n),
                    "repeat_idx": int(rep),
                    "warmup": int(warmup),
                    "status": "ok",
                    "error": "",
                    "padded_n": int(timed["n_total"]),
                    "assigned": int(timed["assigned"]),
                    "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
                    "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else 0.0,
                }
                for key in STAGE_FIELDS:
                    rec[key] = float(timed[key])
                detail_writer.writerow(rec)
                df.flush()
                if not warmup:
                    rows.append(rec)

            summary = _summary_row(target_n, rows, args, base_payload)
            summary_writer.writerow(summary)
            sf.flush()
            print(
                f"[target {target_n}] total_mean={summary['total_ms_mean']:.3f} ms "
                f"p50={summary['total_ms_p50']:.3f} model_mean={summary['model_ms_mean']:.3f} "
                f"post_mean={summary['post_ms_mean']:.3f} peak_alloc={summary['peak_allocated_mb']:.1f} MB",
                flush=True,
            )

    print(f"\nwrote {detail_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
