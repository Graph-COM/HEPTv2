import math
import socket
import random
import inspect
import torch
import numpy as np
from datetime import datetime
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from torch.amp import GradScaler
from torch_geometric.utils import batched_negative_sampling

from typing import Optional, Tuple
from torch import Tensor

from utils.losses import SetCriterion, HungarianMatcher


def _scatter_broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    if dim < 0:
        dim = ref.dim() + dim
    if src.dim() == 1:
        for _ in range(dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), ref.dim()):
        src = src.unsqueeze(-1)
    return src.expand(ref.size())


def _scatter_sum(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    index = _scatter_broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    return out.scatter_add_(dim, index, src)


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    out = _scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = _scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _scatter_broadcast(count, out, dim)

    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


def compute_edge_weight(data):
    node_positions = data.pos
    node_indices = data.edge_index
    dist = torch.sum((node_positions[node_indices[0]] - node_positions[node_indices[1]]) ** 2, dim=1)
    dist = torch.unsqueeze(dist, dim=-1)
    edge_weights = -dist  # Calculate torch.exp(-dist / w) in the forward process
    return edge_weights


def log(*args):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', *args)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_amp(config, device):
    amp_config = config.get("amp", {})
    if isinstance(amp_config, bool):
        amp_enabled = amp_config
        amp_dtype_name = "float16"
    else:
        amp_config = amp_config or {}
        amp_enabled = bool(amp_config.get("enabled", False))
        amp_dtype_name = str(amp_config.get("dtype", "float16")).lower()
    if device.type != "cuda":
        amp_enabled = False
    if amp_dtype_name in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    return amp_enabled, amp_dtype, scaler


def get_loss(loss_name, loss_kwargs, model_kwargs):
    if loss_name == "set":
        weight_dict = {"loss_ce": loss_kwargs["loss_ce"],
                       "loss_mask": loss_kwargs["loss_mask"],
                       "loss_dice": loss_kwargs["loss_dice"],
                       "clf_loss": loss_kwargs["clf_loss"],
                       "clf_pos_weight": loss_kwargs["clf_pos_weight"],
                       "nce_weight": loss_kwargs["nce_weight"]}

        if model_kwargs['deep_supervision']:
            dec_layers = model_kwargs["num_dec_layers"] + 1
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        matcher = HungarianMatcher(cost_class=weight_dict["loss_ce"],
                                   cost_mask=weight_dict["loss_mask"],
                                   cost_dice=weight_dict["loss_dice"])
        eos_coef = loss_kwargs["eos_coef"]
        losses = ["labels", "masks"]
        return SetCriterion(matcher, weight_dict, eos_coef, losses)
    else:
        raise NotImplementedError


def get_optimizer(parameters, optimizer_name, optimizer_kwargs):
    if optimizer_name == "adam":
        return Adam(parameters, **optimizer_kwargs)
    elif optimizer_name == "adamw":
        return AdamW(parameters, **optimizer_kwargs)
    elif optimizer_name == "muon":
        try:
            from torch.optim import Muon  # type: ignore[attr-defined]
        except Exception as exc:
            raise ImportError(
                "optimizer_name='muon' requested, but torch.optim.Muon is unavailable "
                "in this PyTorch build."
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
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported!")


def get_lr_scheduler(optimizer, lr_scheduler_name, lr_scheduler_kwargs):
    if lr_scheduler_name is None:
        return None
    elif lr_scheduler_name == "impatient":
        lr_scheduler_kwargs.pop("num_training_steps", None)
        return ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_name == "warmup_hold_cosine":
        return get_warmup_hold_cosine_schedule(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_name == "step":
        lr_scheduler_kwargs.pop("num_training_steps", None)
        return StepLR(optimizer, **lr_scheduler_kwargs)
    else:
        raise ValueError(f"LR scheduler {lr_scheduler_name} not supported!")


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, eta_min, num_cycles=0.5, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_warmup_hold_cosine_schedule(
    optimizer,
    num_training_steps,
    total_epochs,
    warmup_epochs,
    hold_until_epoch,
    decay_end_epoch,
    decay_min_factor=0.1,
    rebound_peak_epoch=None,
    rebound_end_epoch=None,
    rebound_peak_factor=None,
    last_epoch: int = -1,
):
    """
    Piecewise per-iteration LR schedule (LambdaLR):
      - Warmup linearly from 0 -> 1 over `warmup_epochs`
      - Hold at 1 until `hold_until_epoch`
      - Cosine decay from 1 -> `decay_min_factor` by `decay_end_epoch`
      - Optional rebound:
          * Cosine raise `decay_min_factor` -> `rebound_peak_factor` by `rebound_peak_epoch`
          * Cosine cool back to `decay_min_factor` by `rebound_end_epoch`
      - Clamp at `decay_min_factor` afterwards
    Epoch boundaries are converted using `num_training_steps / total_epochs`.
    """
    steps_per_epoch = float(num_training_steps) / float(max(1, int(total_epochs)))
    warmup_steps = float(warmup_epochs) * steps_per_epoch
    hold_end_step = float(hold_until_epoch) * steps_per_epoch
    decay_end_step = float(decay_end_epoch) * steps_per_epoch
    min_factor = float(decay_min_factor)

    warmup_steps = max(1.0, warmup_steps)
    hold_end_step = max(warmup_steps, hold_end_step)
    decay_end_step = max(hold_end_step, decay_end_step)
    decay_span = max(1.0, decay_end_step - hold_end_step)

    use_rebound = (
        rebound_peak_epoch is not None and rebound_end_epoch is not None and rebound_peak_factor is not None
    )
    if use_rebound:
        rebound_peak_step = float(rebound_peak_epoch) * steps_per_epoch
        rebound_end_step = float(rebound_end_epoch) * steps_per_epoch
        rebound_peak = float(rebound_peak_factor)

        rebound_peak_step = max(decay_end_step, rebound_peak_step)
        rebound_end_step = max(rebound_peak_step, rebound_end_step)
        rebound_up_span = max(1.0, rebound_peak_step - decay_end_step)
        rebound_down_span = max(1.0, rebound_end_step - rebound_peak_step)
    else:
        rebound_peak_step = rebound_end_step = rebound_peak = None
        rebound_up_span = rebound_down_span = None

    def cosine_interp(start, end, progress):
        p = min(max(float(progress), 0.0), 1.0)
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * p))

    def lr_lambda(current_step: int):
        s = float(current_step)
        if s < warmup_steps:
            return s / warmup_steps
        if s < hold_end_step:
            return 1.0
        if s < decay_end_step:
            progress = (s - hold_end_step) / decay_span
            return cosine_interp(1.0, min_factor, progress)
        if use_rebound:
            if s < rebound_peak_step:
                progress = (s - decay_end_step) / rebound_up_span
                return cosine_interp(min_factor, rebound_peak, progress)
            if s < rebound_end_step:
                progress = (s - rebound_peak_step) / rebound_down_span
                return cosine_interp(rebound_peak, min_factor, progress)
        return min_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def find_first_available_port(starting_port):
    port = starting_port
    while True:
        try:
            # Attempt to create a socket and bind it to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                # If successful, return the port
                return port
        except OSError:
            # If the port is already in use, try the next one
            port += 1


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/augmentation.html#add_random_edge
def add_random_edge(
    edge_index,
    p: float = 0.5,
    force_undirected: bool = False,
    batch: Optional[Tensor] = None,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Randomly adds edges to :obj:`edge_index`.

    The method returns (1) the retained :obj:`edge_index`, (2) the added
    edge indices.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float): Ratio of added edges to the existing edges.
            (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`,
            added edges will be undirected.
            (default: :obj:`False`)
        num_nodes (int, Tuple[int], optional): The overall number of nodes,
            *i.e.* :obj:`max_val + 1`, or the number of source and
            destination nodes, *i.e.* :obj:`(max_src_val + 1, max_dst_val + 1)`
            of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`)

    Examples:

        >>> # Standard case
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3],
                [1, 0, 2, 1, 3, 2, 0, 2, 1]])
        >>> added_edges
        tensor([[2, 1, 3],
                [0, 2, 1]])

        >>> # The returned graph is kept undirected
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3, 0, 2, 1],
                [1, 0, 2, 1, 3, 2, 0, 2, 1, 2, 1, 3]])
        >>> added_edges
        tensor([[2, 1, 3, 0, 2, 1],
                [0, 2, 1, 2, 1, 3]])

        >>> # For bipartite graphs
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
        ...                            [2, 3, 1, 4, 2, 1]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           num_nodes=(6, 5))
        >>> edge_index
        tensor([[0, 1, 2, 3, 4, 5, 3, 4, 1],
                [2, 3, 1, 4, 2, 1, 1, 3, 2]])
        >>> added_edges
        tensor([[3, 4, 1],
                [1, 3, 2]])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Ratio of added edges has to be between 0 and 1 " f"(got '{p}')")

    device = edge_index.device
    if not training or p == 0.0:
        edge_index_to_add = torch.tensor([[], []], device=device)
        return edge_index_to_add

    edge_index_to_add = batched_negative_sampling(
        edge_index=edge_index,
        batch=batch,
        num_neg_samples=round(edge_index.size(1) * p / (batch.max().item() + 1)),
        force_undirected=force_undirected,
    )

    # edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)

    return edge_index_to_add
