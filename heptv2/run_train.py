"""Single-GPU training entry point for heptv2.

Usage:
    python -m heptv2.run_train --config heptv2/config_train.yaml

Mirrors `src/tracking_trainer.py:run_one_seed` in pared-down form:
no DDP, no wandb, no efficiency plots, no GNN reference, no freeze/reinit.
Saves `best_checkpoint.pt` whenever validation total loss improves.
"""
import argparse
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from .utils.block_size_utils import resolve_block_sizes
from .utils.config_utils import apply_overrides
from .data.dataset import TrackingTransform, TrackmlLarge
from .model import Transformer
from .data.preprocess import process_data, process_inference_data


def log(*args):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args)


def _as_list(obj):
    if obj is None:
        return []
    return list(obj) if isinstance(obj, (list, tuple)) else [obj]


def _strip_shell_quotes(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _apply_env_file(env_file):
    """Apply simple `export KEY=value` / `unset KEY` lines before postprocess import."""
    if not env_file:
        return
    path = Path(env_file)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"env_file not found: {path}")

    export_re = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    unset_re = re.compile(r"^unset\s+(.+)$")
    applied = 0
    unset = 0
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        unset_match = unset_re.match(line)
        if unset_match:
            for name in unset_match.group(1).split():
                os.environ.pop(name, None)
                unset += 1
            continue
        export_match = export_re.match(line)
        if export_match:
            os.environ[export_match.group(1)] = _strip_shell_quotes(export_match.group(2))
            applied += 1
    log(f"Applied env_file={path} ({applied} exports, {unset} unsets)")


def _apply_dynamo_recompile_limit_from_env():
    recompile_limit = int(os.environ.get("HEPTV2_DYNAMO_RECOMPILE_LIMIT", "0") or 0)
    if recompile_limit <= 0:
        return
    import torch._dynamo.config as dynamo_config

    dynamo_config.recompile_limit = recompile_limit
    dynamo_config.accumulated_recompile_limit = max(
        int(getattr(dynamo_config, "accumulated_recompile_limit", recompile_limit)),
        recompile_limit * 16,
    )
    log(f"Set torch._dynamo recompile_limit={recompile_limit}")


def _metric_inference_path_enabled(cfg):
    eval_cfg = cfg.get("eval", {}) or {}
    path = str(eval_cfg.get("metric_path", "")).strip().lower()
    return bool(eval_cfg.get("metric_inference_path", False)) or path in {
        "inference",
        "compact",
        "run_inference",
    }


def _strip_prefixes(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in ("module.", "_orig_mod."):
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return {k.replace("_orig_mod.", ""): v for k, v in out.items()}


def _load_model_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = _strip_prefixes(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"Resumed weights: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        log(f"  first missing: {missing[:5]}")
    if unexpected:
        log(f"  first unexpected: {unexpected[:5]}")
    return ckpt


def _load_optimizer_state(optimizers, ckpt, resume_optimizer):
    if "optimizer_state_dict" not in ckpt or not resume_optimizer:
        if not resume_optimizer and "optimizer_state_dict" in ckpt:
            log("Skipping optimizer state (resume_optimizer=false)")
        return False
    try:
        state = ckpt["optimizer_state_dict"]
        if isinstance(state, list):
            for opt, st in zip(optimizers, state):
                opt.load_state_dict(st)
        else:
            optimizers[0].load_state_dict(state)
        log("Loaded optimizer state")
        return True
    except ValueError as exc:
        log(f"Skipping optimizer state (mismatch): {exc}")
        return False


def _load_scheduler_state(schedulers, ckpt, loaded_optimizer, resume_lr_scheduler):
    if "lr_scheduler_state_dict" not in ckpt or not schedulers:
        return
    if not resume_lr_scheduler:
        log("Skipping lr_scheduler state (resume_lr_scheduler=false)")
        return
    if not loaded_optimizer:
        log("Skipping lr_scheduler state (optimizer state not loaded)")
        return
    try:
        state = ckpt["lr_scheduler_state_dict"]
        if isinstance(state, list):
            for s, st in zip(schedulers, state):
                s.load_state_dict(st)
        else:
            schedulers[0].load_state_dict(state)
        log("Loaded lr_scheduler state")
    except Exception as exc:
        log(f"Skipping lr_scheduler state (mismatch): {exc}")


def _save_checkpoint(path, model, optimizers, schedulers, epoch, metrics):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            [o.state_dict() for o in optimizers] if len(optimizers) > 1 else optimizers[0].state_dict()
        ),
        **{k: v for k, v in metrics.items()},
    }
    if schedulers:
        payload["lr_scheduler_state_dict"] = (
            [s.state_dict() for s in schedulers] if len(schedulers) > 1 else schedulers[0].state_dict()
        )
    torch.save(payload, path)


def _apply_trainable_prefixes(model, cfg):
    trainable_prefixes = cfg.get("trainable_prefixes")
    freeze_prefixes = cfg.get("freeze_prefixes")
    if trainable_prefixes is not None:
        trainable_prefixes = tuple(str(p) for p in trainable_prefixes)
        for _, p in model.named_parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if name.startswith(trainable_prefixes):
                p.requires_grad_(True)
    if freeze_prefixes:
        freeze_prefixes = tuple(str(p) for p in freeze_prefixes)
        for name, p in model.named_parameters():
            if name.startswith(freeze_prefixes):
                p.requires_grad_(False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable parameters: {trainable:,}/{total:,}")


def _optimizer_parameters(model, cfg):
    groups_cfg = cfg.get("optimizer_param_groups")
    include_frozen = bool(cfg.get("optimizer_include_frozen_params", False))
    trainable_named = [
        (name, p)
        for name, p in model.named_parameters()
        if include_frozen or p.requires_grad
    ]
    if not groups_cfg:
        return [p for _, p in trainable_named]

    if cfg.get("optimizer_name") == "muon":
        raise ValueError("optimizer_param_groups is only wired for adam/adamw optimizers.")

    base_kwargs = dict(cfg.get("optimizer_kwargs", {}) or {})
    assigned = set()
    groups = []
    for raw_group in groups_cfg:
        group_cfg = dict(raw_group or {})
        prefixes = tuple(str(p) for p in _as_list(group_cfg.pop("prefixes", ())))
        name = str(group_cfg.pop("name", ",".join(prefixes) or f"group{len(groups)}"))
        if not prefixes:
            raise ValueError(f"optimizer_param_groups entry {name!r} has no prefixes")
        params = [
            p
            for param_name, p in trainable_named
            if id(p) not in assigned and param_name.startswith(prefixes)
        ]
        if not params:
            log(f"Optimizer param group {name}: 0 tensors (skipped)")
            continue
        assigned.update(id(p) for p in params)
        group = {"params": params}
        for key, value in base_kwargs.items():
            group.setdefault(key, value)
        for key, value in group_cfg.items():
            group[key] = value
        groups.append(group)
        n_params = sum(p.numel() for p in params)
        lr = group.get("lr", base_kwargs.get("lr"))
        log(f"Optimizer param group {name}: {len(params)} tensors, {n_params:,} params, lr={lr}")

    rest = [p for _, p in trainable_named if id(p) not in assigned]
    if rest:
        group = {"params": rest}
        group.update(base_kwargs)
        groups.append(group)
        n_params = sum(p.numel() for p in rest)
        log(
            "Optimizer param group default: "
            f"{len(rest)} tensors, {n_params:,} params, lr={group.get('lr')}"
        )
    if not groups:
        raise ValueError("No trainable parameters found for optimizer initialization.")
    return groups


def _build_l2sp_refs(model, cfg):
    weight = float(cfg.get("finetune_l2sp_weight", 0.0) or 0.0)
    if weight <= 0:
        return None
    refs = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    log(f"L2-SP refs: {len(refs)} tensors, weight={weight:g}")
    return refs


def _add_selection_metrics(metrics, cfg):
    selection_cfg = cfg.get("selection", {}) or {}
    if not selection_cfg:
        return metrics
    dm = metrics.get("dm")
    fake_key = str(selection_cfg.get("fake_key", "fake_rate"))
    fake_value = metrics.get(fake_key)
    if dm is not None and fake_value is not None:
        fake_limit = float(
            selection_cfg.get(
                f"{fake_key}_limit",
                selection_cfg.get("fake_rate_limit", 0.008),
            )
        )
        fake_penalty = float(
            selection_cfg.get(
                f"{fake_key}_penalty",
                selection_cfg.get("fake_rate_penalty", selection_cfg.get("fake_penalty", 4.0)),
            )
        )
        metrics["dm_fake_score"] = float(dm) - fake_penalty * max(0.0, float(fake_value) - fake_limit)
    return metrics


@torch.no_grad()
def _run_inference_metrics_epoch(
    model,
    loader,
    phase,
    epoch,
    device,
    cfg,
    amp_enabled,
    amp_dtype,
    limit_batches=None,
):
    """Run the exact inference metric path used by `heptv2.run_inference`."""
    from tqdm import tqdm

    from .run_inference import run_event as run_inference_event

    was_training = model.training
    model.eval()
    sums = defaultdict(float)
    counts = defaultdict(int)
    try:
        pbar = tqdm(loader, desc=f"[Epoch {epoch}] {phase}-inference", dynamic_ncols=True)
        for batch_idx, data in enumerate(pbar):
            if limit_batches is not None and batch_idx >= int(limit_batches):
                break
            event_metrics = run_inference_event(model, data, cfg, device, amp_enabled, amp_dtype)
            for key, value in event_metrics.items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    sums[key] += value
                    counts[key] += 1
            if "dm" in counts and counts["dm"]:
                pbar.set_postfix(dm=f"{sums['dm'] / counts['dm']:.5f}")
    finally:
        if was_training:
            model.train()

    return {key: sums[key] / counts[key] for key in sums if counts[key]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "train.yaml"))
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted YAML override, e.g. --override device=cuda:1 --override optimizer_kwargs.lr=1e-5",
    )
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, args.override)
    _apply_env_file(cfg.get("env_file"))
    _apply_dynamo_recompile_limit_from_env()

    # Import after env_file is applied so postprocess constants match the
    # inference/benchmark path when configs source the production env helper.
    from .run_inference import (
        _compact_assignment_active,
        _compile_kwargs_from_mode,
        _parse_pad_bucket_lengths,
    )
    from .training.losses import build_set_criterion
    from .training.train import _tracking_metrics_cfg_from_config, run_one_epoch
    from .training.train_utils import (
        get_lr_scheduler,
        get_optimizer,
        patch_dynamo_nvcc_probe,
        set_seed,
        setup_amp,
    )

    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(int(cfg.get("num_threads", 4)))
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_enabled, amp_dtype, scaler = setup_amp(cfg, device)

    data_root = Path(cfg["data_root"])
    dataset = TrackmlLarge(root=data_root, transform=TrackingTransform())
    loaders = {
        split: DataLoader(
            dataset[dataset.idx_split[split]],
            batch_size=cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=cfg.get("num_workers", 4),
            pin_memory=False,
        )
        for split in ("train", "valid", "test")
    }

    model = Transformer(
        attn_type="hept",
        in_dim=dataset.x_dim,
        coords_dim=dataset.coords_dim,
        task=cfg["dataset_name"],
        **cfg["model_kwargs"],
    ).to(device)

    start_epoch = 0
    best_valid_loss = float("inf")
    loaded_optimizer = False
    resume_ckpt = None
    if cfg.get("resume"):
        ckpt_path = Path(cfg["resume"])
        if not ckpt_path.is_absolute() and not ckpt_path.exists():
            candidate = data_root / "logs" / cfg["resume"] / "best_checkpoint.pt"
            if candidate.exists():
                ckpt_path = candidate
        log(f"Resume from {ckpt_path}")
        resume_ckpt = _load_model_weights(model, ckpt_path, device)

    _apply_trainable_prefixes(model, cfg)

    patch_dynamo_nvcc_probe()
    compile_mode = os.environ.get("HEPTV2_CODEX_COMPILE_MODE", "default")
    decode_compile_mode = os.environ.get("HEPTV2_CODEX_DECODE_COMPILE_MODE", compile_mode)
    encode_compile_kwargs = _compile_kwargs_from_mode(compile_mode)
    decode_compile_kwargs = _compile_kwargs_from_mode(decode_compile_mode)
    compact_assignment = _compact_assignment_active(cfg, model)
    compile_train_methods = bool(cfg.get("compile_train_methods", not compact_assignment))
    if compile_train_methods:
        model.encode = torch.compile(model.encode, **encode_compile_kwargs)
    if hasattr(model, "encode_online"):
        model.encode_online = torch.compile(model.encode_online, **encode_compile_kwargs)
    if compile_train_methods:
        model.decode = torch.compile(model.decode, **decode_compile_kwargs)
    if (
        compact_assignment
        and cfg.get("postprocessing", {}).get("decoder_compile_batched_decode", False)
        and hasattr(model, "decode_assignment_inputs")
    ):
        model.decode_assignment_inputs = torch.compile(
            model.decode_assignment_inputs,
            **decode_compile_kwargs,
        )
    if bool(cfg.get("compile_whole_model", not compact_assignment)):
        model = torch.compile(model, **encode_compile_kwargs)
    l2sp_refs = _build_l2sp_refs(model, cfg)

    opt = get_optimizer(_optimizer_parameters(model, cfg), cfg["optimizer_name"], cfg["optimizer_kwargs"])
    optimizers = _as_list(opt)

    schedulers = []
    if cfg.get("lr_scheduler_name") is not None:
        for optimizer in optimizers:
            kwargs = dict(cfg["lr_scheduler_kwargs"])
            kwargs["num_training_steps"] = cfg["num_epochs"] * len(loaders["train"])
            schedulers.append(get_lr_scheduler(optimizer, cfg["lr_scheduler_name"], kwargs))
    lr_s = None if not schedulers else (schedulers[0] if len(schedulers) == 1 else schedulers)

    criterion = build_set_criterion(cfg["loss_kwargs"], cfg["model_kwargs"]).to(device)

    if resume_ckpt is not None:
        loaded_optimizer = _load_optimizer_state(
            optimizers, resume_ckpt, cfg.get("resume_optimizer", True)
        )
        _load_scheduler_state(
            schedulers, resume_ckpt, loaded_optimizer, cfg.get("resume_lr_scheduler", False)
        )
        if cfg.get("resume_start_epoch") is not None:
            start_epoch = int(cfg["resume_start_epoch"])
        elif "epoch" in resume_ckpt:
            start_epoch = int(resume_ckpt["epoch"]) + 1
        best_valid_loss = float(resume_ckpt.get("best_valid_loss", best_valid_loss))

    stamp = datetime.now().strftime("%m_%d-%H_%M_%S.%f")[:-4]
    rand = np.random.randint(10, 100)
    note = cfg.get("note", "train")
    log_dir = data_root / "logs" / f"{stamp}{rand}_{cfg['model_name']}_{cfg['seed']}_{note}"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    log(f"Log dir: {log_dir}")

    enc_block, dec_block = resolve_block_sizes(cfg["model_kwargs"])
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    regions = base_model.helper_funcs["regions"]

    train_split_cfg = cfg.get("train_decoder_splitting", {}) or {}

    path_cfg = cfg.get("inference_path", {}) or {}
    train_uses_inference_preprocess = (
        str(cfg.get("train_preprocess_path", "")).strip().lower()
        in {"inference", "compact", "run_inference"}
    )

    def _process(data, randomize_decoder_split: bool):
        if train_uses_inference_preprocess:
            return process_inference_data(
                data,
                fixed_len=cfg["model_kwargs"]["fixed_seq_len"],
                encoder_block_size=enc_block,
                decoder_block_size=dec_block,
                regions=regions,
                num_sub_events=cfg.get("num_sub_events", cfg.get("eval", {}).get("num_sub_events", 3)),
                decoder_overlap=cfg["model_kwargs"].get("decoder_overlap", True),
                eta_abs_max=cfg.get("eval", {}).get("eta_abs_max", 4.0),
                return_sub_events=True,
                dynamic_pad=bool(path_cfg.get("dynamic_pad", False)),
                pad_bucket_lengths=_parse_pad_bucket_lengths(path_cfg.get("pad_buckets")),
                match_fixed_region_bins=bool(path_cfg.get("match_fixed_region_bins", False)),
                build_targets=True,
            )
        return process_data(
            data,
            fixed_len=cfg["model_kwargs"]["fixed_seq_len"],
            encoder_block_size=enc_block,
            decoder_block_size=dec_block,
            regions=regions,
            num_sub_events=cfg.get("num_sub_events", cfg.get("eval", {}).get("num_sub_events", 3)),
            decoder_overlap=cfg["model_kwargs"].get("decoder_overlap", True),
            random_split_shift=bool(randomize_decoder_split and train_split_cfg.get("random_shift", False)),
            random_split_shift_max_frac=train_split_cfg.get("random_shift_max_frac", 1.0),
            overlap_shift_jitter_frac=(
                train_split_cfg.get("overlap_shift_jitter_frac", 0.0)
                if randomize_decoder_split else 0.0
            ),
            eta_abs_max=cfg.get("eval", {}).get("eta_abs_max", 4.0),
            build_targets=True,
        )

    def process_train(data):
        return _process(data, randomize_decoder_split=True)

    def process_eval(data):
        return _process(data, randomize_decoder_split=False)

    limit_batches = cfg.get("limit_batches")
    only_eval = bool(cfg.get("only_eval", False))

    eval_cfg = cfg.get("eval", {}) or {}
    compute_metrics = bool(eval_cfg.get("compute_metrics", True))
    metrics_cfg = _tracking_metrics_cfg_from_config(cfg) if compute_metrics else None
    use_inference_metrics = compute_metrics and _metric_inference_path_enabled(cfg)
    loss_metrics_cfg = None if use_inference_metrics else metrics_cfg
    compute_eval_loss = bool(eval_cfg.get("compute_loss", True))
    if use_inference_metrics:
        log("Validation metrics use the run_inference compact/inference path")
    best_metric_key = cfg.get("best_metric_key", "total_loss")
    best_metric_mode = cfg.get("best_metric_mode", "min")  # "min" for loss, "max" for dm

    if only_eval:
        eval_splits = _as_list(cfg.get("eval_splits", ("valid", "test")))
        eval_splits = [str(split) for split in eval_splits]
        log(f"only_eval=true — running eval splits: {eval_splits}")
        for split in eval_splits:
            if split not in loaders:
                raise ValueError(f"Unknown eval split {split!r}; expected one of {tuple(loaders)}")
            res = {}
            if compute_eval_loss:
                res = run_one_epoch(
                    model, optimizers, criterion, loaders[split], split, start_epoch, device,
                    lr_s, cfg, process_eval, amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
                    limit_batches=limit_batches, metrics_cfg=loss_metrics_cfg,
                )
            if use_inference_metrics:
                res.update(
                    _run_inference_metrics_epoch(
                        model, loaders[split], split, start_epoch, device, cfg,
                        amp_enabled, amp_dtype, limit_batches=limit_batches,
                    )
                )
                res = _add_selection_metrics(res, cfg)
            log(f"{split}: {res}")
        return

    best_value = float("inf") if best_metric_mode == "min" else float("-inf")
    if best_metric_key == "total_loss":
        best_value = best_valid_loss
    for epoch in range(start_epoch, cfg["num_epochs"]):
        checkpoint_every = int(cfg.get("checkpoint_every_n_train_batches", 0) or 0)
        checkpoint_batches = {
            int(v) for v in _as_list(cfg.get("checkpoint_train_batches"))
            if v is not None
        }

        def _mid_epoch_checkpoint(batch_idx, partial_train_metrics):
            batch_num = batch_idx + 1
            should_save = batch_num in checkpoint_batches
            if checkpoint_every > 0 and batch_num % checkpoint_every == 0:
                should_save = True
            if not should_save:
                return
            path = log_dir / f"epoch{epoch:03d}_batch{batch_num:05d}_checkpoint.pt"
            _save_checkpoint(
                path,
                model,
                optimizers,
                schedulers,
                epoch,
                {"best_valid_loss": best_valid_loss, "best_metric_key": best_metric_key,
                 "best_metric_value": best_value, "train_metrics": partial_train_metrics,
                 "mid_epoch_batch": batch_num},
            )
            log(f"  -> saved mid-epoch checkpoint at batch {batch_num}: {path}")

        train_res = run_one_epoch(
            model, optimizers, criterion, loaders["train"], "train", epoch, device,
            lr_s, cfg, process_train, amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
            limit_batches=limit_batches, l2sp_refs=l2sp_refs,
            batch_end_callback=_mid_epoch_checkpoint if (checkpoint_every > 0 or checkpoint_batches) else None,
        )
        if train_res.get("early_aborted", 0.0) > 0:
            log(f"epoch {epoch} train: {train_res}")
            log("Early-stopped this experiment from train-batch DM/fake monitor")
            return
        valid_res = {}
        if compute_eval_loss:
            valid_res = run_one_epoch(
                model, optimizers, criterion, loaders["valid"], "valid", epoch, device,
                lr_s, cfg, process_eval, amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
                limit_batches=limit_batches, metrics_cfg=loss_metrics_cfg,
            )
        if use_inference_metrics:
            valid_res.update(
                _run_inference_metrics_epoch(
                    model, loaders["valid"], "valid", epoch, device, cfg,
                    amp_enabled, amp_dtype, limit_batches=limit_batches,
                )
            )
        valid_res = _add_selection_metrics(valid_res, cfg)

        # Step epoch-level schedulers (LambdaLR steps per-iter inside train_one_batch).
        for sched in schedulers:
            if not isinstance(sched, LambdaLR):
                sched.step()

        log(f"epoch {epoch} train: {train_res}")
        log(f"epoch {epoch} valid: {valid_res}")

        current = valid_res.get(best_metric_key, float("inf") if best_metric_mode == "min" else float("-inf"))
        improved = current < best_value if best_metric_mode == "min" else current > best_value
        if improved:
            best_value = current
            if best_metric_key == "total_loss":
                best_valid_loss = best_value
            _save_checkpoint(
                log_dir / "best_checkpoint.pt",
                model, optimizers, schedulers, epoch,
                {"best_valid_loss": best_valid_loss, "best_metric_key": best_metric_key,
                 "best_metric_value": best_value, "valid_metrics": valid_res, "train_metrics": train_res},
            )
            log(f"  -> new best ({best_metric_key}={best_value:.4f}), saved checkpoint")

        _save_checkpoint(
            log_dir / "last_checkpoint.pt",
            model, optimizers, schedulers, epoch,
            {"best_valid_loss": best_valid_loss, "best_metric_key": best_metric_key,
             "best_metric_value": best_value, "valid_metrics": valid_res, "train_metrics": train_res},
        )


if __name__ == "__main__":
    main()
