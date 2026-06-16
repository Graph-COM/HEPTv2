"""Benchmark latency on a saved list of PyG Data events.

This is useful on machines where copying the full processed TrackML tensor is
unnecessary. The timed path is shared with ``benchmark_inference_latency`` via
``run_timed_event``.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from heptv2.benchmark_inference_latency import (
    _CompactPostprocessCUDAGraphCache,
    load_config,
    run_timed_event,
)
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


STAGE_FIELDS = ["pre_ms", "encode_ms", "decode_ms", "model_ms", "post_ms", "total_ms"]


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


def _build_model(cfg: dict, x_dim: int, coords_dim: int, device: torch.device):
    model = Transformer(
        attn_type="hept",
        in_dim=int(x_dim),
        coords_dim=int(coords_dim),
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


def _summarize(rows: list[dict]) -> dict:
    out = {}
    for key in STAGE_FIELDS:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_p50"] = float(np.percentile(vals, 50))
        out[f"{key}_p90"] = float(np.percentile(vals, 90))
    prepost = np.asarray([row["pre_ms"] + row["post_ms"] for row in rows], dtype=np.float64)
    out["prepost_ms_mean"] = float(prepost.mean())
    out["prepost_ms_p50"] = float(np.percentile(prepost, 50))
    out["prepost_ms_p90"] = float(np.percentile(prepost, 90))
    out["assigned_mean"] = float(np.asarray([row["assigned"] for row in rows], dtype=np.float64).mean())
    out["padded_n_p50"] = float(np.percentile([row["padded_n"] for row in rows], 50))
    out["peak_allocated_mb"] = float(max(row["peak_allocated_mb"] for row in rows))
    out["peak_reserved_mb"] = float(max(row["peak_reserved_mb"] for row in rows))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("heptv2/configs/finetune_phase48_scatter_amp.yaml"))
    parser.add_argument("--eventfile", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup-events", type=int, default=190)
    parser.add_argument("--measure-events", type=int, default=190)
    parser.add_argument("--out-dir", type=Path, default=Path("codex_logs/a100_best_190events_eventfile"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, [f"device={args.device}"])
    path_cfg = cfg.get("inference_path", {})
    path_cfg["dynamic_pad"] = bool(path_cfg.get("dynamic_pad", True))
    pad_buckets = path_cfg.get("pad_buckets") or []
    pad_bucket_lengths = [int(x) for x in pad_buckets] or None

    payload = torch.load(args.eventfile, map_location="cpu", weights_only=False)
    events = payload["events"]
    x_dim = int(payload.get("x_dim", 15))
    coords_dim = int(payload.get("coords_dim", 6))
    if args.warmup_events > len(events) or args.measure_events > len(events):
        raise ValueError(f"eventfile has {len(events)} events")

    device = torch.device(cfg["device"])
    amp_enabled, amp_dtype = _setup_torch(cfg, device)
    model, compact_assignment, post_graph_cache = _build_model(cfg, x_dim, coords_dim, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "detail.csv"
    summary_path = args.out_dir / "summary.csv"
    effective_path = args.out_dir / "effective_config.yaml"
    with effective_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    fieldnames = [
        "phase",
        "event_i",
        "evtid",
        "padded_n",
        "assigned",
        *STAGE_FIELDS,
        "peak_allocated_mb",
        "peak_reserved_mb",
    ]
    measured_rows: list[dict] = []
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        print(
            f"loaded eventfile={args.eventfile} events={len(events)} "
            f"warmup={args.warmup_events} measure={args.measure_events}",
            flush=True,
        )
        for phase, count in (("warmup", args.warmup_events), ("measure", args.measure_events)):
            if phase == "measure" and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for i in range(count):
                data_gpu = events[i].to(device)
                row = run_timed_event(
                    model,
                    data_gpu,
                    cfg,
                    amp_enabled,
                    amp_dtype,
                    "minimal",
                    materialize_decode_output=False,
                    compact_assignment=compact_assignment,
                    dynamic_pad=bool(path_cfg.get("dynamic_pad", True)),
                    pad_bucket_lengths=pad_bucket_lengths,
                    lazy_embedding=False,
                    post_graph_cache=post_graph_cache,
                )
                rec = {
                    "phase": phase,
                    "event_i": i,
                    "evtid": int(events[i].evtid[0]) if hasattr(events[i], "evtid") else i,
                    "padded_n": int(row["n_total"]),
                    "assigned": int(row["assigned"]),
                    "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0,
                    "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2
                    if device.type == "cuda"
                    else 0.0,
                }
                for key in STAGE_FIELDS:
                    rec[key] = float(row[key])
                writer.writerow(rec)
                f.flush()
                if phase == "measure":
                    measured_rows.append(rec)
            print(f"{phase} done: {count} events", flush=True)

    summary = _summarize(measured_rows)
    summary.update(
        {
            "eventfile": str(args.eventfile),
            "warmup_events": int(args.warmup_events),
            "measure_events": int(args.measure_events),
        }
    )
    with summary_path.open("w", newline="") as f:
        fieldnames = ["eventfile", "warmup_events", "measure_events", *summary.keys()]
        # Preserve order while removing duplicates from the three manually
        # inserted names above.
        fieldnames = list(dict.fromkeys(fieldnames))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)

    print("\n=== latency summary ===")
    for key in STAGE_FIELDS:
        print(
            f"{key:>10s}: mean={summary[f'{key}_mean']:.3f} "
            f"p50={summary[f'{key}_p50']:.3f} p90={summary[f'{key}_p90']:.3f}",
            flush=True,
        )
    print(f"prepost_ms: mean={summary['prepost_ms_mean']:.3f} p50={summary['prepost_ms_p50']:.3f}")
    print(f"peak_allocated_mb={summary['peak_allocated_mb']:.1f}")
    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
