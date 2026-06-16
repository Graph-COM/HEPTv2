import torch
import torch.nn as nn
from typing import List

from einops import rearrange
from ..model_utils.hash_utils import lsh_mapping, batched_index_select, invert_permutation, E2LSH
from ..model_utils.serialization_utils import canonicalize_serialization_type, compute_serialization_order
from utils.block_size_utils import resolve_block_sizes
try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    flex_attention = None
# from flash_attn import flash_attn_func


def sort_to_buckets(x, perm, bucketsz):
    return rearrange(
        batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
        "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
        bucketsz=bucketsz,
    )


def unsort_from_buckets(s_x, perm_inverse):
    b_x = rearrange(s_x, "h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d")
    return batched_index_select(b_x, perm_inverse)


def softmax_with_denom(x, dim=-1, return_log=True):
    # logZ = log(sum_j exp(x_j)) — stable
    logZ = torch.logsumexp(x, dim=dim, keepdim=True)
    probs = torch.exp(x - logZ)             # softmax
    if return_log:
        return probs, logZ                   # return log-denom (preferred)
    else:
        return probs, torch.exp(logZ)        # return denom


def qkv_res(s_query, s_key, s_value):

    if flex_attention is not None:
        t_query = rearrange(s_query, "c h nbuckets b d -> (c h) nbuckets b d").contiguous()
        t_key = rearrange(s_key, "c h nbuckets b d -> (c h) nbuckets b d").contiguous()
        t_value = rearrange(s_value, "c h nbuckets b d -> (c h) nbuckets b d").contiguous()

        # flex_attention expects [B, H, L, D] and returns output + lse if requested
        out, lse = flex_attention(t_query, t_key, t_value, return_lse=True)
        out = rearrange(out, "(c h) nbuckets b d -> c h nbuckets b d", h=s_query.shape[1])
        lse = rearrange(lse, "(c h) nbuckets b -> c h nbuckets b 1", h=s_query.shape[1])
        return lse, out

    # use flash attention
    # new_q = rearrange(s_query, "c h nbuckets b d -> (c nbuckets) b h d").contiguous().bfloat16()
    # new_k = rearrange(s_key, "c h nbuckets b d -> (c nbuckets) b h d").contiguous().bfloat16()
    # new_v = rearrange(s_value, "c h nbuckets b d -> (c nbuckets) b h d").contiguous().bfloat16()

    # out, lse, _ = flash_attn_func(new_q, new_k, new_v, return_attn_probs=True, window_size=(512, 512))
    # out = rearrange(out, "(c nbuckets) b h d -> c h nbuckets b d", c=s_query.shape[0])
    # lse = rearrange(lse, "(c nbuckets) h b -> c h nbuckets b 1", c=s_query.shape[0])
    # return lse.float(), out.float()

    # handle numerical stability of exp
    # scale_factor = 1.0 / (s_query.shape[-1] ** 0.5)
    # clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key) * scale_factor
    # qk, denom = softmax_with_denom(clustered_dists, dim=-1, return_log=True)
    # return denom, torch.einsum("...ij,...jd->...id", qk, s_value)


def prep_qk(query, key, w, coords):
    # qw = w.sum(dim=1).clamp(max=50).relu().sum(dim=-1)
    # new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)

    # sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * coords[:, None]
    # q_hat = torch.cat([query, sqrt_w_r], dim=-1)
    # k_hat = torch.cat([key, sqrt_w_r], dim=-1)
    q_hat, k_hat = query, key
    return q_hat, k_hat


@torch.no_grad()
def get_geo_shift(regions_h: List[List[int]], hash_shift, region_indices, num_or_hashes):
    region_indices_eta, region_indices_phi = region_indices

    q_hash_shift_eta = region_indices_eta * hash_shift
    k_hash_shift_eta = region_indices_eta * hash_shift

    q_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    k_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    res = torch.stack([q_hash_shift_phi + q_hash_shift_eta, k_hash_shift_phi + k_hash_shift_eta], dim=0)
    return rearrange(res, "a (c h) n -> a c h n", c=num_or_hashes)


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
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)
        self.e2lsh_new = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=2)
        encoder_serialization = kwargs.get("encoder_serialization_type", kwargs.get("serialization_type", "random_lsh"))
        self.encoder_serialization_type = canonicalize_serialization_type(encoder_serialization, stage="encoder")
        self.encoder_hilbert_bits = kwargs.get("encoder_hilbert_bits", kwargs.get("num_bits", 10))

    def _build_deterministic_positions(self, coords: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        order, _ = compute_serialization_order(
            point_coords=coords,
            serialization_type=self.encoder_serialization_type,
            valid_mask=valid_mask,
            num_bits=self.encoder_hilbert_bits,
            stage="encoder",
        )
        per_head_order = order.unsqueeze(0).expand(self.num_heads, -1)
        return per_head_order.unsqueeze(0).expand(self.n_hashes, -1, -1)

    def forward(self, query, key, value, **kwargs):
        clf_hard, clf_prob = kwargs["clf_res"]
        # TODO: support batched inputs
        query = query.view(-1, self.num_heads, self.dim_per_head)
        key = key.view(-1, self.num_heads, self.dim_per_head)
        value = value.view(-1, self.num_heads, self.dim_per_head)

        w = rearrange(
            kwargs["w_rpe"].weight,
            "(h d) (r k) -> h d r k",
            h=self.num_heads,
            d=self.dim_per_head,
            k=self.num_w_per_dist,
        )
        q_hat, k_hat = prep_qk(query, key, w, kwargs["coords"])

        q_hat = rearrange(q_hat, "n h d -> h n d")
        k_hat = rearrange(k_hat, "n h d -> h n d")
        value = rearrange(value, "n h d -> h n d")
        # valid = kwargs["valid_mask"].view(1, -1, 1)
        # q_hat, k_hat, value = q_hat * valid, k_hat * valid, value * valid
        # q_hat = q_hat * clf_prob[None]
        # k_hat = k_hat * clf_prob[None]
        # value = value * clf_prob[None]

        # q_hat = q_hat * clf_hard[None]
        # k_hat = k_hat * clf_hard[None]
        # value = value * clf_hard[None]
        valid_mask = kwargs["valid_mask"] if "valid_mask" in kwargs else torch.ones(
            q_hat.shape[1], dtype=torch.bool, device=q_hat.device
        )
        pos = kwargs["coords"][..., :2].repeat(self.num_heads, 1, 1)

        if self.encoder_serialization_type == "random_lsh":
            q_hashed, k_hashed, hash_shift = lsh_mapping(self.e2lsh_new, pos, pos)
            # q_hashed, k_hashed, hash_shift = lsh_mapping(self.e2lsh, q_hat, k_hat)
            invalid = ~valid_mask
            q_hashed[..., invalid] = float("inf")
            k_hashed[..., invalid] = float("inf")
            # q_hashed[..., clf_hard.squeeze(1) == 0] = float("inf")
            # k_hashed[..., clf_hard.squeeze(1) == 0] = float("inf")

            hash_shift = rearrange(hash_shift, "c h d -> (c h) d")
            q_shifts, k_shifts = get_geo_shift(kwargs["regions_h"], hash_shift, kwargs["region_indices"], self.n_hashes)
            q_hashed = q_hashed + q_shifts
            k_hashed = k_hashed + k_shifts
            q_positions = q_hashed.argsort(dim=-1)
            k_positions = k_hashed.argsort(dim=-1)
        else:
            q_positions = self._build_deterministic_positions(kwargs["coords"], valid_mask)
            k_positions = q_positions

        # s_query = sort_to_buckets(q_hat, q_positions, q_hat.shape[-2])
        # s_key = sort_to_buckets(k_hat, k_positions, q_hat.shape[-2])
        # s_value = sort_to_buckets(value, k_positions, q_hat.shape[-2])

        s_query = sort_to_buckets(q_hat, q_positions, self.block_size)
        s_key = sort_to_buckets(k_hat, k_positions, self.block_size)
        s_value = sort_to_buckets(value, k_positions, self.block_size)

        denom, so = qkv_res(s_query, s_key, s_value)

        q_rev_positions = invert_permutation(q_positions)
        o = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(denom, q_rev_positions)
        # probs = logits / torch.sum(logits, dim=0, keepdim=True)
        # now denom is log
        probs = torch.softmax(logits, dim=0)
        # probs = logits.mean(dim=0)

        out = torch.sum(o * probs, dim=0)
        # out = o.sum(dim=0) / logits.sum(dim=0)
        out = self.out_linear(rearrange(out, "h n d -> n (h d)"))
        return out
