"""Positional embeddings used at inference.

`PELearned` is the active encoder PE (`pe_type: learned`).
`PositionEmbeddingCoordsSine` is constructed by Mask3DHead but its forward is
not called; we keep it so the `mask3d_head.pos_enc.gauss_B` buffer loads from
the checkpoint.
"""
import math
import torch
from torch import nn


class PELearned(nn.Module):
    def __init__(self, input_channel, **kwargs):
        super().__init__()
        num_pos_feats = kwargs["h_dim"] * 4
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.SiLU(),
            nn.Linear(num_pos_feats, kwargs["h_dim"]),
        )

    def forward(self, xyz):
        return self.position_embedding_head(xyz)


def get_pe_func(pe_type, coords_size, kwargs):
    if pe_type == "learned":
        return PELearned(input_channel=coords_size, **kwargs)
    raise NotImplementedError(f"heptv2 only supports pe_type='learned', got {pe_type}")


class PositionEmbeddingCoordsSine(nn.Module):
    """Kept only to materialize the `gauss_B` buffer present in the checkpoint.
    Forward is not exercised at inference."""

    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None and d_pos % 2 == 0
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)

    def forward(self, xyz, num_channels=None, input_range=None):
        raise RuntimeError("PositionEmbeddingCoordsSine.forward is not used at inference in heptv2.")
