"""LSH utilities: E2LSH, region vectors, bucketed index select.

Parameter names kept identical to `src/models/model_utils/hash_utils.py` so
state_dicts load unchanged.
"""
import os

import torch
import torch.nn as nn
from einops import rearrange


def quantile_partition(sorted_indices, num_regions):
    total_elements = sorted_indices.shape[-1]
    region_size = torch.ceil(total_elements / num_regions)
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    base = torch.arange(total_elements, device=sorted_indices.device)[None]
    region_indices = base // region_size + 1
    return region_indices[:, inverse_indices]


def get_regions(num_regions, num_or_hashes, num_heads, num_and_hashes=2):
    lb = 2
    ub = 2 * num_regions ** (1 / num_and_hashes) - lb
    regions = []
    for _ in range(num_or_hashes * num_heads):
        region = [torch.rand(1).item() * (ub - lb) + lb for _ in range(num_and_hashes)]
        regions.append(region)
    regions = torch.tensor(regions)
    regions = (num_regions / regions.prod(dim=1, keepdim=True)) ** (1 / num_and_hashes) * regions
    regions = torch.round(regions * 3) / 3
    return rearrange(regions, "(h c) a -> c a h", h=num_heads)


# def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
#     arange = torch.arange(perm.shape[-1], device=perm.device).expand_as(perm)
#     return torch.empty_like(perm).scatter_(-1, perm, arange)
def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """
    Params:
        perm: (..., n)
    Return:
        inverse_perm: (..., n)
    """
    if os.environ.get("HEPTV2_SCATTER_INVERT_PERM", "0").lower() in {"1", "true", "yes", "on"}:
        arange = torch.arange(perm.shape[-1], device=perm.device, dtype=perm.dtype).expand_as(perm)
        return torch.zeros_like(perm).scatter(-1, perm, arange)
    if os.environ.get("HEPTV2_FAST_INVERT_PERM", "0").lower() in {"1", "true", "yes", "on"}:
        arange = torch.arange(perm.shape[-1], device=perm.device, dtype=perm.dtype).expand_as(perm)
        return torch.empty_like(perm).scatter_(-1, perm, arange)
    # Avoid in-place scatter here: recent Inductor can generate unsafe
    # strided layouts for the inverse permutation in HEPT attention.
    return torch.argsort(perm, dim=-1)

def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    last_dim = values.shape[-1]
    indices_expanded = rearrange(indices, "... -> ... 1").expand(*indices.shape, last_dim)
    return values.expand(*indices_expanded.shape[:-2], *values.shape[-2:]).gather(-2, indices_expanded)


def uniform(a, b, shape, device="cpu"):
    return (b - a) * torch.rand(shape, device=device) + a


@torch.no_grad()
def lsh_mapping(e2lsh, queries, keys):
    queries_hashed = e2lsh(queries)
    keys_hashed = e2lsh(keys)
    max_hash_shift = torch.max(queries_hashed.max(-1, keepdim=True).values, keys_hashed.max(-1, keepdim=True).values)
    min_hash_shift = torch.min(queries_hashed.min(-1, keepdim=True).values, keys_hashed.min(-1, keepdim=True).values)
    hash_shift = max_hash_shift - min_hash_shift
    return queries_hashed, keys_hashed, hash_shift


class E2LSH(nn.Module):
    def __init__(self, n_hashes, n_heads, dim, r=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.normal(0, 1, (n_heads, dim, n_hashes)))
        self.beta = nn.Parameter(uniform(0, r, shape=(1, n_hashes)))
        self.alpha.requires_grad = False
        self.beta.requires_grad = False

    def forward(self, vecs):
        projection = torch.bmm(vecs, self.alpha)
        return projection.permute(2, 0, 1)
