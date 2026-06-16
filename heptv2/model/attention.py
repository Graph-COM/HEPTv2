"""HEPTAttention — bucketed attention with LSH-based spatial locality.

Only `encoder_serialization_type=random_lsh` is supported, which matches the
training config. Parameter names and submodule names (`e2lsh`, `e2lsh_new`,
`out_linear`) are identical to the original `src/models/attention/hept.py`.
"""
import os

import torch
import torch.nn as nn
from einops import rearrange
from torch.amp import autocast

from ..utils.hash_utils import (
    E2LSH,
    batched_index_select,
    invert_permutation,
)
from ..utils.block_size_utils import resolve_block_sizes

try:
    from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attention
    import torch as _torch
    # The eager flex_attention kernel prints a warning that it "may produce
    # incorrect results" — wrap with torch.compile to get the fused path.
    flex_attention = _torch.compile(_raw_flex_attention, dynamic=False)
except ImportError:
    flex_attention = None

_FLEX_ATTN_BF16 = os.environ.get("HEPTV2_FLEX_ATTN_BF16", "0").lower() in {"1", "true", "yes", "on"}


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int_limit(name, default, maximum):
    raw = os.environ.get(name, "")
    if not raw:
        return int(default)
    value = int(raw)
    if value <= 0:
        return int(default)
    return max(1, min(int(value), int(maximum)))


def _env_int_layer_limit(name, layer_idx, default, maximum):
    raw = os.environ.get(name, "")
    if raw and layer_idx is not None:
        values = [part.strip() for part in raw.split(",")]
        if int(layer_idx) < len(values) and values[int(layer_idx)]:
            value = int(values[int(layer_idx)])
            if value > 0:
                return max(1, min(value, int(maximum)))
    return _env_int_limit("HEPTV2_ENCODER_NUM_HASHES", default, maximum)


def _flex_attention(q, k, v, **kwargs):
    if _FLEX_ATTN_BF16 and q.is_cuda:
        out = None
        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            out = flex_attention(q, k, v, **kwargs)
        if isinstance(out, tuple):
            return tuple(x.to(q.dtype) if torch.is_tensor(x) else x for x in out)
        return out.to(q.dtype)
    return flex_attention(q, k, v, **kwargs)


def sort_to_buckets(x, perm, bucketsz):
    if _env_bool("HEPTV2_DIRECT_BUCKET_GATHER"):
        num_hashes = int(perm.shape[0])
        num_heads = int(perm.shape[1])
        seq_len = int(perm.shape[2])
        head_dim = int(x.shape[-1])
        gathered = x.unsqueeze(0).expand(num_hashes, -1, -1, -1).gather(
            2,
            perm[..., None].expand(num_hashes, num_heads, seq_len, head_dim),
        )
        return gathered.reshape(num_hashes, num_heads, seq_len // bucketsz, bucketsz, head_dim)
    return rearrange(
        batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
        "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
        bucketsz=bucketsz,
    )


def sort_qkv_to_buckets(q, k, v, perm, bucketsz):
    if not _env_bool("HEPTV2_COMBINED_BUCKET_GATHER"):
        return (
            sort_to_buckets(q, perm, bucketsz),
            sort_to_buckets(k, perm, bucketsz),
            sort_to_buckets(v, perm, bucketsz),
        )
    qkv = torch.cat((q, k, v), dim=-1)
    s_qkv = sort_to_buckets(qkv, perm, bucketsz)
    return s_qkv.split(q.shape[-1], dim=-1)


def unsort_from_buckets(s_x, perm_inverse):
    if _env_bool("HEPTV2_DIRECT_BUCKET_GATHER"):
        flat = s_x.reshape(s_x.shape[0], s_x.shape[1], -1, s_x.shape[-1])
        return flat.gather(2, perm_inverse[..., None].expand(*perm_inverse.shape, s_x.shape[-1]))
    b_x = rearrange(s_x, "h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d")
    return batched_index_select(b_x, perm_inverse)


def qkv_res(s_query, s_key, s_value):
    if flex_attention is None:
        raise RuntimeError("heptv2 requires torch.nn.attention.flex_attention.")
    t_query = rearrange(s_query, "c h nbuckets b d -> (c h) nbuckets b d").contiguous()
    t_key = rearrange(s_key, "c h nbuckets b d -> (c h) nbuckets b d").contiguous()
    t_value = rearrange(s_value, "c h nbuckets b d -> (c h) nbuckets b d").contiguous()
    out, lse = _flex_attention(t_query, t_key, t_value, return_lse=True)
    out = rearrange(out, "(c h) nbuckets b d -> c h nbuckets b d", h=s_query.shape[1])
    lse = rearrange(lse, "(c h) nbuckets b -> c h nbuckets b 1", h=s_query.shape[1])
    return lse, out


def lsh_coords(e2lsh, coords2, num_heads):
    if _env_bool("HEPTV2_E2LSH_COORDS_EINSUM"):
        return torch.einsum("nd,hdc->chn", coords2, e2lsh.alpha)
    pos = coords2.repeat(num_heads, 1, 1)
    return e2lsh(pos)


@torch.no_grad()
def get_geo_shift(regions_h, hash_shift, region_indices, num_or_hashes):
    region_indices_eta, region_indices_phi = region_indices
    q_hash_shift_eta = region_indices_eta * hash_shift
    k_hash_shift_eta = region_indices_eta * hash_shift
    q_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    k_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    res = torch.stack([q_hash_shift_phi + q_hash_shift_eta, k_hash_shift_phi + k_hash_shift_eta], dim=0)
    return rearrange(res, "a (c h) n -> a c h n", c=num_or_hashes)


@torch.no_grad()
def get_geo_shift_single(regions_h, hash_shift, region_indices, num_or_hashes):
    region_indices_eta, region_indices_phi = region_indices
    hash_shift_eta = region_indices_eta * hash_shift
    hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    return rearrange(hash_shift_phi + hash_shift_eta, "(c h) n -> c h n", c=num_or_hashes)


class HEPTAttention(nn.Module):
    def __init__(self, hash_dim, **kwargs):
        super().__init__()
        self.num_heads = kwargs["num_heads"]
        self.dim_per_head = kwargs["head_dim"]
        self.model_dim = kwargs.get("model_dim", self.num_heads * self.dim_per_head)
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.model_dim)

        self.block_size, _ = resolve_block_sizes(kwargs)
        self.n_hashes = kwargs["n_hashes"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        # e2lsh operates on hash_dim (unused at inference) and e2lsh_new on the
        # 2-D (phi, eta) coordinates. Both buffers are kept to match the checkpoint.
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)
        self.e2lsh_new = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=2)

    def forward(self, query, key, value, **kwargs):
        query = query.view(-1, self.num_heads, self.dim_per_head)
        key = key.view(-1, self.num_heads, self.dim_per_head)
        value = value.view(-1, self.num_heads, self.dim_per_head)

        q_hat = rearrange(query, "n h d -> h n d")
        k_hat = rearrange(key, "n h d -> h n d")
        value = rearrange(value, "n h d -> h n d")

        valid_mask = kwargs.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(q_hat.shape[1], dtype=torch.bool, device=q_hat.device)
        coords2 = kwargs["coords"][..., :2]

        invalid = ~valid_mask
        active_hashes = _env_int_layer_limit(
            "HEPTV2_ENCODER_NUM_HASHES_BY_LAYER",
            kwargs.get("_encoder_layer_idx"),
            self.n_hashes,
            self.n_hashes,
        )
        active_width = int(active_hashes) * int(self.num_heads)
        if _env_bool("HEPTV2_ENCODER_HASH_FP32"):
            with autocast(device_type="cuda", enabled=False):
                hashed = lsh_coords(self.e2lsh_new, coords2.float(), self.num_heads)
                if active_hashes < self.n_hashes:
                    hashed = hashed[:active_hashes]
                max_hash_shift = hashed.max(-1, keepdim=True).values
                min_hash_shift = hashed.min(-1, keepdim=True).values
                hash_shift = max_hash_shift - min_hash_shift
                hashed[..., invalid] = float("inf")
                hash_shift = rearrange(hash_shift, "c h d -> (c h) d")
                regions_h = kwargs["regions_h"].float()[:, :active_width]
                region_indices = [idx[:active_width] for idx in kwargs["region_indices"]]
                shifts = get_geo_shift_single(
                    regions_h, hash_shift, region_indices, active_hashes
                )
                positions = (hashed + shifts).argsort(dim=-1)
        else:
            hashed = lsh_coords(self.e2lsh_new, coords2, self.num_heads)
            if active_hashes < self.n_hashes:
                hashed = hashed[:active_hashes]
            max_hash_shift = hashed.max(-1, keepdim=True).values
            min_hash_shift = hashed.min(-1, keepdim=True).values
            hash_shift = max_hash_shift - min_hash_shift
            hashed[..., invalid] = float("inf")

            hash_shift = rearrange(hash_shift, "c h d -> (c h) d")
            regions_h = kwargs["regions_h"][:, :active_width]
            region_indices = [idx[:active_width] for idx in kwargs["region_indices"]]
            shifts = get_geo_shift_single(
                regions_h, hash_shift, region_indices, active_hashes
            )
            hashed = hashed + shifts
            positions = hashed.argsort(dim=-1)

        s_query, s_key, s_value = sort_qkv_to_buckets(q_hat, k_hat, value, positions, self.block_size)

        denom, so = qkv_res(s_query, s_key, s_value)
        q_rev_positions = invert_permutation(positions)
        if _env_bool("HEPTV2_COMBINED_UNSORT"):
            combined = torch.cat([so, denom], dim=-1)
            combined = unsort_from_buckets(combined, q_rev_positions)
            o = combined[..., : self.dim_per_head]
            logits = combined[..., self.dim_per_head :]
        else:
            o = unsort_from_buckets(so, q_rev_positions)
            logits = unsort_from_buckets(denom, q_rev_positions)
        if _env_bool("HEPTV2_MANUAL_HASH_SOFTMAX"):
            weights = torch.exp(logits - logits.max(dim=0, keepdim=True).values)
            out = torch.sum(o * weights, dim=0) / torch.sum(weights, dim=0)
        else:
            probs = torch.softmax(logits, dim=0)
            out = torch.sum(o * probs, dim=0)
        return self.out_linear(rearrange(out, "h n d -> n (h d)"))
