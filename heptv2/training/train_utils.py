"""Single-GPU training utilities: seed, AMP, optimizer/scheduler factories, logging.

Drops DDP, ReduceLROnPlateau, and cosine/warmup_hold_cosine variants that the
current checkpoints do not use. Add them back here if a future config needs them.
"""
import functools
import inspect
import math
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from torch.amp import GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR


def patch_dynamo_nvcc_probe():
    """Wrap torch._dynamo's nvcc --version probe so a PermissionError doesn't
    abort compilation (debug_utils rebinds the name in after_aot — patch both).
    """
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
    try:
        from torch._dynamo.repro import after_aot as dynamo_after_aot
        dynamo_after_aot._cuda_system_info_comment = safe
    except Exception:
        pass


def log(*args):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_amp(config: dict, device: torch.device) -> Tuple[bool, torch.dtype, GradScaler]:
    amp_config = config.get("amp", {}) or {}
    if isinstance(amp_config, bool):
        amp_enabled, amp_dtype_name = amp_config, "float16"
        scaler_kwargs = {}
    else:
        amp_enabled = bool(amp_config.get("enabled", False))
        amp_dtype_name = str(amp_config.get("dtype", "float16")).lower()
        grad_scaler_enabled = bool(amp_config.get("grad_scaler", True))
        scaler_kwargs = {
            key: amp_config[key]
            for key in ("init_scale", "growth_factor", "backoff_factor", "growth_interval")
            if key in amp_config
        }
    if isinstance(amp_config, bool):
        grad_scaler_enabled = True
    if device.type != "cuda":
        amp_enabled = False
    amp_dtype = torch.bfloat16 if amp_dtype_name in ("bf16", "bfloat16") else torch.float16
    scaler = GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16 and grad_scaler_enabled,
        **scaler_kwargs,
    )
    return amp_enabled, amp_dtype, scaler


def _as_list(obj):
    if obj is None:
        return []
    return list(obj) if isinstance(obj, (list, tuple)) else [obj]


def get_optimizer(parameters, optimizer_name: str, optimizer_kwargs: dict):
    """Returns a single optimizer, or a list when `muon` is used (Muon for
    2D+ params, AdamW for 1D params — matches src/utils/utils.py).
    """
    if optimizer_name == "adam":
        return Adam(parameters, **optimizer_kwargs)
    if optimizer_name == "adamw":
        return AdamW(parameters, **optimizer_kwargs)
    if optimizer_name == "muon":
        try:
            from torch.optim import Muon  # type: ignore[attr-defined]
        except Exception as exc:
            raise ImportError(
                "optimizer_name='muon' requires torch.optim.Muon (not available in this build)."
            ) from exc

        params = [p for p in parameters if p.requires_grad]
        muon_params = [p for p in params if getattr(p, "ndim", 0) >= 2]
        adamw_params = [p for p in params if getattr(p, "ndim", 0) < 2]

        muon_lr = optimizer_kwargs.get("lr", 1e-3)
        muon_sig = inspect.signature(Muon.__init__)
        muon_valid_args = {k for k in muon_sig.parameters if k not in {"self", "params"}}
        muon_kwargs = {}
        for key, value in optimizer_kwargs.items():
            if key in {"adamw_lr", "adamw_betas", "adamw_weight_decay", "betas"}:
                continue
            if key in muon_valid_args:
                muon_kwargs[key] = value
        muon_kwargs.setdefault("lr", muon_lr)

        adamw_kwargs = {
            "lr": optimizer_kwargs.get("adamw_lr", max(1e-6, float(muon_lr) * 0.1)),
            "betas": optimizer_kwargs.get("adamw_betas", optimizer_kwargs.get("betas", (0.9, 0.95))),
            "weight_decay": optimizer_kwargs.get(
                "adamw_weight_decay", optimizer_kwargs.get("weight_decay", 0.01)
            ),
        }

        optimizers = []
        if muon_params:
            optimizers.append(Muon(muon_params, **muon_kwargs))
        if adamw_params:
            optimizers.append(AdamW(adamw_params, **adamw_kwargs))
        if not optimizers:
            raise ValueError("No trainable parameters found for optimizer initialization.")
        return optimizers[0] if len(optimizers) == 1 else optimizers
    raise ValueError(f"Optimizer {optimizer_name} not supported.")


def _warmup_hold_cosine(optimizer, num_training_steps, total_epochs, warmup_epochs,
                       hold_until_epoch, decay_end_epoch, decay_min_factor=0.1,
                       last_epoch=-1, **_):
    """Per-iter LambdaLR: linear warmup → hold at 1.0 → cosine decay to `decay_min_factor`."""
    steps_per_epoch = float(num_training_steps) / float(max(1, int(total_epochs)))
    warmup_steps = max(1.0, float(warmup_epochs) * steps_per_epoch)
    hold_end_step = max(warmup_steps, float(hold_until_epoch) * steps_per_epoch)
    decay_end_step = max(hold_end_step, float(decay_end_epoch) * steps_per_epoch)
    decay_span = max(1.0, decay_end_step - hold_end_step)
    min_factor = float(decay_min_factor)

    def lr_lambda(step):
        s = float(step)
        if s < warmup_steps:
            return s / warmup_steps
        if s < hold_end_step:
            return 1.0
        if s < decay_end_step:
            progress = (s - hold_end_step) / decay_span
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr_scheduler(optimizer, lr_scheduler_name, lr_scheduler_kwargs):
    if lr_scheduler_name is None:
        return None
    kwargs = dict(lr_scheduler_kwargs)
    if lr_scheduler_name == "step":
        kwargs.pop("num_training_steps", None)
        return StepLR(optimizer, **kwargs)
    if lr_scheduler_name == "warmup_hold_cosine":
        return _warmup_hold_cosine(optimizer, **kwargs)
    raise ValueError(
        f"LR scheduler {lr_scheduler_name!r} not supported. Only 'step' and 'warmup_hold_cosine' "
        "are wired up in heptv2 (port over from src/utils/utils.py if you need more)."
    )
