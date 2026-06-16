"""Set-prediction loss for the mask decoder.

Ported from `src/utils/losses.py` with DDP-only branches removed. Supports
deep supervision via `aux_outputs` (the decoder emits one entry per layer)
and honors the serialized point order returned by the decoder.
"""
from typing import List, Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


class _NestedTensor:
    def __init__(self, tensors: Tensor, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask


def _nested_tensor_from_mask_list(tensor_list: List[Tensor]) -> _NestedTensor:
    """Stack a list of [Q, N_i, 1] masks to a dense [B, Q, N_max, 1] tensor
    plus a [B, N_max, 1] valid-region mask. Matches the detectron2 layout
    used by `SetCriterion.loss_masks`.
    """
    assert tensor_list and tensor_list[0].dim() == 3, "expect [Q, N, 1] per batch element"
    shapes = [list(t.shape) for t in tensor_list]
    max_q = max(s[0] for s in shapes)
    max_n = max(s[1] for s in shapes)
    max_c = max(s[2] for s in shapes)
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    out = torch.zeros((len(tensor_list), max_q, max_n, max_c), dtype=dtype, device=device)
    mask = torch.ones((len(tensor_list), max_n, max_c), dtype=torch.bool, device=device)
    for i, t in enumerate(tensor_list):
        q, n, c = t.shape
        out[i, :q, :n, :c].copy_(t)
        mask[i, :n, :c] = False
    return _NestedTensor(out, mask)


def _is_identity_order(order: Tensor) -> bool:
    ref = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
    return bool(torch.equal(order, ref))


def _maybe_get_point_order(outputs, batch_idx=None, device=None):
    order = outputs.get("_serialized_point_order")
    if order is None:
        return None
    if order.dim() == 2:
        if batch_idx is None:
            raise ValueError("Batched serialized point order requires a batch index.")
        order = order[batch_idx]
    elif order.dim() != 1:
        raise ValueError(f"Unexpected serialized point order shape: {tuple(order.shape)}")
    order = order.to(dtype=torch.long)
    if device is not None and order.device != device:
        order = order.to(device)
    return order


def _apply_point_order(mask_tensor: Tensor, point_order: Optional[Tensor]):
    if point_order is None:
        return mask_tensor
    point_dim = -2 if mask_tensor.dim() >= 2 and mask_tensor.shape[-1] == 1 else -1
    if mask_tensor.shape[point_dim] != point_order.numel():
        raise ValueError(
            f"Serialized point order length {point_order.numel()} does not match mask point "
            f"dimension {mask_tensor.shape[point_dim]} (shape={tuple(mask_tensor.shape)})."
        )
    if _is_identity_order(point_order):
        return mask_tensor
    return mask_tensor.index_select(point_dim, point_order)


def dice_loss(inputs: Tensor, targets: Tensor, num_masks: float) -> Tensor:
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def dice_loss_per_mask(inputs: Tensor, targets: Tensor) -> Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return 1 - (numerator + 1) / (denominator + 1)


def sigmoid_focal_loss(inputs: Tensor, targets: Tensor, num_masks: float, alpha: float = 0.25, gamma: float = 2) -> Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks


def sigmoid_focal_loss_per_mask(inputs: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2) -> Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1)


def batch_dice_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def batch_sigmoid_focal_loss(inputs: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2) -> Tensor:
    hw = inputs.shape[1]
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)
    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum("nc,mc->nm", focal_neg, 1 - targets)
    return loss / hw


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)
            out_mask = outputs["pred_masks"][b]
            tgt_ids = targets[b]["labels"]
            tgt_mask = targets[b]["masks"].to(out_mask)
            tgt_mask = _apply_point_order(tgt_mask, _maybe_get_point_order(outputs, batch_idx=b, device=tgt_mask.device))

            cost_class = -out_prob[:, tgt_ids]
            out_flat = out_mask.flatten(1)
            tgt_flat = tgt_mask.flatten(1)
            cost_mask = batch_sigmoid_focal_loss(out_flat, tgt_flat)
            cost_dice = batch_dice_loss(out_flat, tgt_flat)

            C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            C = C.reshape(num_queries, -1).float().cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    """DETR-style set loss: Hungarian matching + cross-entropy + focal+dice on masks.
    Supports deep supervision via `aux_outputs`. Weights are applied in
    `weighted_loss` (see heptv2/train.py) using `self.weight_dict`.
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_dict: dict,
        eos_coef: float,
        losses: List[str],
        short_track_weight: float = 0.0,
        short_track_class_weight: float = 0.0,
        short_track_max_hits: int = 5,
        short_track_count_source: str = "local",
        hard_example_weight: float = 0.0,
        hard_example_source: str = "loss",
        hard_example_max_scale: float = 3.0,
    ):
        super().__init__()
        self.num_classes = 1
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.short_track_weight = float(short_track_weight)
        self.short_track_class_weight = float(short_track_class_weight)
        self.short_track_max_hits = int(short_track_max_hits)
        self.short_track_count_source = str(short_track_count_source)
        self.hard_example_weight = float(hard_example_weight)
        self.hard_example_source = str(hard_example_source)
        self.hard_example_max_scale = float(hard_example_max_scale)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _src_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _tgt_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_masks):
        src_logits = outputs["pred_logits"]
        idx = self._src_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight.to(src_logits.device),
            reduction="none",
        )
        if self.short_track_class_weight > 0 and idx[0].numel() > 0:
            weights = torch.ones_like(ce)
            count_parts = []
            for target, (_, matched_tgt) in zip(targets, indices):
                if self.short_track_count_source == "global" and "hit_counts" in target:
                    count_parts.append(target["hit_counts"][matched_tgt].to(ce))
                else:
                    count_parts.append(target["masks"].flatten(1).sum(dim=1)[matched_tgt].to(ce))
            if count_parts:
                hit_counts = torch.cat(count_parts)
                short = hit_counts <= float(self.short_track_max_hits)
                weights[idx] = weights[idx] + self.short_track_class_weight * short.to(weights.dtype)
                weights = weights / weights.mean().clamp_min(1e-6)
                ce = ce * weights
        loss_ce = ce.mean()
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices, num_masks):
        src_idx = self._src_idx(indices)
        tgt_idx = self._tgt_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, _ = _nested_tensor_from_mask_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        target_masks = _apply_point_order(target_masks, _maybe_get_point_order(outputs, device=target_masks.device))
        src_flat = src_masks[:, None][:, 0].flatten(1)
        tgt_flat = target_masks.flatten(1).view(src_flat.shape)
        loss_mask = sigmoid_focal_loss_per_mask(src_flat, tgt_flat)
        loss_dice = dice_loss_per_mask(src_flat, tgt_flat)
        weights = torch.ones_like(loss_mask)
        if self.short_track_weight > 0:
            if self.short_track_count_source == "global":
                count_parts = []
                for target, (_, matched_tgt) in zip(targets, indices):
                    if "hit_counts" in target:
                        count_parts.append(target["hit_counts"][matched_tgt].to(tgt_flat))
                    else:
                        local_counts = target["masks"].flatten(1).sum(dim=1)
                        count_parts.append(local_counts[matched_tgt].to(tgt_flat))
                hit_counts = torch.cat(count_parts) if count_parts else tgt_flat.sum(dim=1)
            else:
                hit_counts = tgt_flat.sum(dim=1)
            short = hit_counts <= float(self.short_track_max_hits)
            weights = weights + self.short_track_weight * short.to(weights.dtype)
        if self.hard_example_weight > 0 and loss_mask.numel() > 0:
            if self.hard_example_source in ("positive", "positive_miss", "pos_miss"):
                pos_count = tgt_flat.sum(dim=1).clamp_min(1.0)
                pos_prob = (src_flat.detach().sigmoid() * tgt_flat).sum(dim=1) / pos_count
                hardness = 1.0 - pos_prob
            elif self.hard_example_source == "dice":
                hardness = loss_dice.detach()
            else:
                hardness = loss_mask.detach() + loss_dice.detach()
            hardness = hardness / hardness.mean().clamp_min(1e-6)
            hardness = hardness.clamp(min=0.0, max=self.hard_example_max_scale)
            weights = weights + self.hard_example_weight * hardness
        weights = weights / weights.mean().clamp_min(1e-6)
        return {
            "loss_mask": (loss_mask * weights).sum() / num_masks,
            "loss_dice": (loss_dice * weights).sum() / num_masks,
        }

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        outputs_main = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_main, targets)
        num_masks = max(1, sum(len(t["labels"]) for t in targets))
        num_masks = float(num_masks)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_masks)
                    losses.update({k + f"_{i}": v for k, v in l_dict.items()})
        return losses


def build_set_criterion(loss_kwargs: dict, model_kwargs: dict) -> SetCriterion:
    """Constructs the set criterion from yaml `loss_kwargs` + `model_kwargs`."""
    weight_dict = {
        "loss_ce": loss_kwargs["loss_ce"],
        "loss_mask": loss_kwargs["loss_mask"],
        "loss_dice": loss_kwargs["loss_dice"],
        "clf_loss": loss_kwargs["clf_loss"],
        "clf_pos_weight": loss_kwargs["clf_pos_weight"],
        "nce_weight": loss_kwargs["nce_weight"],
    }
    if model_kwargs.get("deep_supervision", False):
        aux_layers = model_kwargs["num_dec_layers"] + 1
        aux_weight = {}
        for i in range(aux_layers - 1):
            aux_weight.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight)
    matcher = HungarianMatcher(
        cost_class=weight_dict["loss_ce"],
        cost_mask=weight_dict["loss_mask"],
        cost_dice=weight_dict["loss_dice"],
    )
    return SetCriterion(
        matcher,
        weight_dict,
        loss_kwargs["eos_coef"],
        ["labels", "masks"],
        short_track_weight=loss_kwargs.get("short_track_weight", 0.0),
        short_track_class_weight=loss_kwargs.get("short_track_class_weight", 0.0),
        short_track_max_hits=loss_kwargs.get("short_track_max_hits", 5),
        short_track_count_source=loss_kwargs.get("short_track_count_source", "local"),
        hard_example_weight=loss_kwargs.get("hard_example_weight", 0.0),
        hard_example_source=loss_kwargs.get("hard_example_source", "loss"),
        hard_example_max_scale=loss_kwargs.get("hard_example_max_scale", 3.0),
    )
