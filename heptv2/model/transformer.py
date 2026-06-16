"""Transformer (encoder + Mask3DHead decoder) — inference only.

The submodule hierarchy and parameter names are kept identical to
`src/models/baselines/transformer.py` so the existing checkpoints load with
strict=True after stripping `_orig_mod.` / `module.` prefixes.

Dropped relative to the training model: training-time branches (Gumbel
sampling is always off), freeze helpers, re-init helpers, dead classes
(MaskMLP, make_new_pred_masks, etc.), non-flex attention fallbacks, and
serialization types other than `random_lsh` (encoder) / `hilbert` or `none`
(decoder).

Kept: `clf_head`, `learnable_thres`, `inter_mapping`, and `mlp_out` — all
present in the checkpoint. `clf_head` feeds post-processing; the other three
are unused at inference but their weights must be loadable.
"""
from typing import List, Optional
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.amp import autocast
from torch_geometric.nn import MLP

from ..utils.block_size_utils import resolve_block_sizes
from ..utils.hash_utils import get_regions
from .attention import HEPTAttention
from .pos_emb import PositionEmbeddingCoordsSine, get_pe_func
from ..utils.serialization import canonicalize_serialization_type, compute_serialization_order

try:
    from torch.nn.attention.flex_attention import create_block_mask as _raw_create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attention
    # Compile both; the eager path prints a warning and "may produce
    # incorrect results". Compiled path matches training-time numerics.
    flex_attention = torch.compile(_raw_flex_attention, dynamic=False)
    create_block_mask = _raw_create_block_mask
except ImportError:
    create_block_mask = None
    flex_attention = None

_FLEX_ATTN_BF16 = os.environ.get("HEPTV2_FLEX_ATTN_BF16", "0").lower() in {"1", "true", "yes", "on"}


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int_set(name):
    raw = os.environ.get(name, "")
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _env_int_limit(name, default, maximum):
    raw = os.environ.get(name, "")
    if not raw:
        return int(default)
    value = int(raw)
    if value <= 0:
        return int(default)
    return max(0, min(int(value), int(maximum)))


def _env_layer_enabled(prefix, layer_idx):
    layers = _env_int_set(f"{prefix}_LAYERS")
    if layers:
        return int(layer_idx) in layers
    return _env_bool(prefix)


def _flex_attention(q, k, v, **kwargs):
    if _FLEX_ATTN_BF16 and q.is_cuda:
        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            out = flex_attention(q, k, v, **kwargs)
        return out.to(q.dtype)
    return flex_attention(q, k, v, **kwargs)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(input_dtype) * hidden_states.to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, act_fn=F.silu):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LearnableClf(nn.Module):
    """Kept to satisfy checkpoint keys (`learnable_thres.{i}.sharpness|threshold`);
    unused at inference."""

    def __init__(self):
        super().__init__()
        self.sharpness = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.threshold = nn.Parameter(torch.tensor(0.0), requires_grad=True)


class Attn(nn.Module):
    def __init__(self, attn_type, coords_dim, **kwargs):
        super().__init__()
        if attn_type != "hept":
            raise NotImplementedError(f"heptv2 only supports attn_type='hept', got {attn_type}")
        self.attn_type = attn_type
        self.model_dim = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        if self.model_dim % self.num_heads != 0:
            raise ValueError(f"h_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads}).")
        self.dim_per_head = self.model_dim // self.num_heads

        self.w_q = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.w_k = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.w_v = nn.Linear(self.model_dim, self.model_dim, bias=False)

        self.attn = HEPTAttention(
            self.dim_per_head + coords_dim,
            head_dim=self.dim_per_head,
            model_dim=self.model_dim,
            **kwargs,
        )

        self.dropout = nn.Dropout(0.0)
        self.norm1 = Qwen3RMSNorm(self.model_dim)
        self.norm2 = Qwen3RMSNorm(self.model_dim)
        self.q_norm = Qwen3RMSNorm(self.dim_per_head)
        self.k_norm = Qwen3RMSNorm(self.dim_per_head)
        mlp_hidden_dim = max(1, int(self.model_dim * kwargs["mlp_ratio"]))
        self.ff = Qwen3MLP(self.model_dim, mlp_hidden_dim, act_fn=F.silu)

        self.w_rpe = nn.Linear(kwargs["num_w_per_dist"] * (coords_dim - 1), self.num_heads * self.dim_per_head)
        self.pe_func = get_pe_func(kwargs["pe_type"], coords_dim, kwargs)

    def forward(self, x, kwargs):
        if _env_bool("HEPTV2_ENCODER_ATTN_BLOCKS_FP32") and x.is_cuda:
            with autocast(device_type="cuda", enabled=False):
                return self._forward_impl(x.float(), kwargs)
        return self._forward_impl(x, kwargs)

    def _forward_impl(self, x, kwargs):
        pe = kwargs["coords"] if self.pe_func is None else self.pe_func(kwargs["coords"])
        x_pe = x + pe if self.pe_func is not None else x
        x_normed = self.norm1(x_pe)
        if _env_bool("HEPTV2_ENCODER_FUSED_QKV"):
            qkv = F.linear(
                x_normed,
                torch.cat((self.w_q.weight, self.w_k.weight, self.w_v.weight), dim=0),
                None,
            )
            q, k, v = qkv.split(self.model_dim, dim=-1)
        else:
            q = self.w_q(x_normed)
            k = self.w_k(x_normed)
            v = self.w_v(x_normed)
        q = q.view(-1, self.num_heads, self.dim_per_head)
        k = k.view(-1, self.num_heads, self.dim_per_head)
        v = v.view(-1, self.num_heads, self.dim_per_head)
        q = self.q_norm(q).view(-1, self.num_heads * self.dim_per_head)
        k = self.k_norm(k).view(-1, self.num_heads * self.dim_per_head)
        v = v.view(-1, self.num_heads * self.dim_per_head)
        aggr_out = self.attn(q, k, v, pe=pe, w_rpe=self.w_rpe, **kwargs)
        x = x + self.dropout(aggr_out)
        ff_out = self.ff(self.norm2(x))
        return x + self.dropout(ff_out)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = Qwen3RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.q_norm = Qwen3RMSNorm(int(d_model // nhead))
        self.k_norm = Qwen3RMSNorm(int(d_model // nhead))
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _with_pos(tensor, pos):
        return tensor if pos is None else tensor + pos

    def _flex_cross_attn(self, tgt, memory, memory_mask=None, pos=None, query_pos=None):
        q = self._with_pos(tgt, query_pos)
        k = self._with_pos(memory, pos)
        v = memory

        w = self.multihead_attn.in_proj_weight
        b = self.multihead_attn.in_proj_bias
        embed_dim = w.shape[1]
        w_q, w_k, w_v = w.split(embed_dim, dim=0)
        b_q, b_k, b_v = (None, None, None) if b is None else b.split(embed_dim)
        q = F.linear(q, w_q, b_q)
        k = F.linear(k, w_k, b_k)
        v = F.linear(v, w_v, b_v)

        q_len, bsz, embed_dim = q.shape
        k_len = k.shape[0]
        num_heads = self.multihead_attn.num_heads
        head_dim = embed_dim // num_heads

        q = q.contiguous().view(q_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
        k = k.contiguous().view(k_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
        v = v.contiguous().view(k_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        block_mask = None
        if memory_mask is not None and create_block_mask is not None:
            mem_mask = memory_mask.contiguous().to(torch.bool)
            if mem_mask.dim() == 3:
                mem_mask = mem_mask.view(bsz, q_len, k_len)

                def mask_mod(b, h, q_idx, kv_idx):
                    return ~mem_mask[b, q_idx, kv_idx]

            elif mem_mask.dim() == 4:
                mem_mask = mem_mask.view(bsz, num_heads, q_len, k_len)

                def mask_mod(b, h, q_idx, kv_idx):
                    return ~mem_mask[b, h, q_idx, kv_idx]

            else:
                raise ValueError(f"Unexpected memory_mask shape: {tuple(mem_mask.shape)}")

            block_mask = create_block_mask(mask_mod, B=bsz, H=num_heads, Q_LEN=q_len, KV_LEN=k_len, device=q.device)

        out = _flex_attention(q, k, v, block_mask=block_mask)
        out = out.permute(2, 0, 1, 3).reshape(q_len, bsz, embed_dim)
        return self.multihead_attn.out_proj(out)

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if flex_attention is None:
            raise RuntimeError("heptv2 requires torch.nn.attention.flex_attention.")
        tgt2 = self.norm(tgt)
        tgt2 = self._flex_cross_attn(tgt2, memory, memory_mask=memory_mask, pos=pos, query_pos=query_pos)
        return tgt + self.dropout(tgt2)


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = Qwen3RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.q_norm = Qwen3RMSNorm(int(d_model // nhead))
        self.k_norm = Qwen3RMSNorm(int(d_model // nhead))
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _with_pos(tensor, pos):
        return tensor if pos is None else tensor + pos

    def _flex_self_attn(self, tgt, query_pos=None):
        q = self._with_pos(tgt, query_pos)
        k = q
        v = tgt

        w = self.self_attn.in_proj_weight
        b = self.self_attn.in_proj_bias
        embed_dim = w.shape[1]
        w_q, w_k, w_v = w.split(embed_dim, dim=0)
        b_q, b_k, b_v = (None, None, None) if b is None else b.split(embed_dim)
        q = F.linear(q, w_q, b_q)
        k = F.linear(k, w_k, b_k)
        v = F.linear(v, w_v, b_v)

        q_len, bsz, embed_dim = q.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads

        q = q.contiguous().view(q_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
        k = k.contiguous().view(q_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
        v = v.contiguous().view(q_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        out = _flex_attention(q, k, v)
        out = out.permute(2, 0, 1, 3).reshape(q_len, bsz, embed_dim)
        return self.self_attn.out_proj(out)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if flex_attention is None:
            raise RuntimeError("heptv2 requires torch.nn.attention.flex_attention.")
        head_dim = self.self_attn.embed_dim // self.self_attn.num_heads
        if head_dim < 16:
            raise RuntimeError(f"flex_attention requires head_dim >= 16, got {head_dim}.")
        tgt2 = self.norm(tgt)
        tgt2 = self._flex_self_attn(tgt2, query_pos=query_pos)
        return tgt + self.dropout(tgt2)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, normalize_before=False):
        super().__init__()
        self.mlp = Qwen3MLP(d_model, dim_feedforward, act_fn=F.silu)
        self.dropout = nn.Dropout(dropout)
        self.norm = Qwen3RMSNorm(d_model)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        return tgt + self.dropout(self.mlp(self.norm(tgt)))


class Mask3DHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_decoder_layers: int,
        num_heads: int,
        deep_supervision: bool,
        use_attn_mask: bool = False,
        mlp_ratio: float = 4.0,
        decoder_serialization_type: str = "hilbert",
        decoder_hilbert_bits: int = 10,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_attn_mask = use_attn_mask
        self.decoder_serialization_type = canonicalize_serialization_type(
            decoder_serialization_type, stage="decoder"
        )
        self.decoder_hilbert_bits = decoder_hilbert_bits

        self.instance_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.instance_pos_emb = nn.Parameter(torch.randn(num_queries, hidden_dim))
        # Present in the checkpoint via its `gauss_B` buffer; not called at inference.
        self.pos_enc = PositionEmbeddingCoordsSine(
            pos_type="fourier", d_pos=hidden_dim, gauss_scale=1.0, d_in=2, normalize=True
        )

        self.num_heads = num_heads
        self.num_layers = num_decoder_layers

        mlp_hidden_dim = max(1, int(hidden_dim * mlp_ratio))
        self.mask_embed = Qwen3MLP(hidden_dim, mlp_hidden_dim, act_fn=F.silu)
        self.decoder_norm = Qwen3RMSNorm(hidden_dim)
        self.class_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.0, normalize_before=True)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.0, normalize_before=True)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=mlp_hidden_dim, dropout=0.0, normalize_before=True)
            )
        self._query_index_cache_key = None
        self._query_index_cache = None

    def _serialize_decoder_points(self, point_feats, point_coords, valid_mask):
        if self.decoder_serialization_type == "none":
            return point_feats, None, None
        if point_feats.dim() == 3:
            order, inverse_order = compute_serialization_order(
                point_coords=point_coords,
                serialization_type=self.decoder_serialization_type,
                valid_mask=valid_mask,
                num_bits=self.decoder_hilbert_bits,
            )
            gather_idx = order[:, :, None].expand(-1, -1, point_feats.shape[-1])
            return point_feats.gather(1, gather_idx), order, inverse_order
        order, inverse_order = compute_serialization_order(
            point_coords=point_coords,
            serialization_type=self.decoder_serialization_type,
            valid_mask=valid_mask,
            num_bits=self.decoder_hilbert_bits,
        )
        identity_order = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
        if torch.equal(order, identity_order):
            return point_feats, order, inverse_order
        return point_feats.index_select(0, order), order, inverse_order

    @property
    def needs_point_coords(self):
        return self.decoder_serialization_type != "none"

    def _inference_queries(self):
        path = os.environ.get("HEPTV2_DECODER_QUERY_INDICES_PATH", "").strip()
        if path:
            if self._query_index_cache_key != path or self._query_index_cache is None:
                query_index = torch.load(path, map_location=self.instance_queries.device)
                if not torch.is_tensor(query_index):
                    query_index = torch.as_tensor(query_index, device=self.instance_queries.device)
                self._query_index_cache = query_index.to(device=self.instance_queries.device, dtype=torch.long)
                self._query_index_cache_key = path
            return (
                self.instance_queries.index_select(0, self._query_index_cache),
                self.instance_pos_emb.index_select(0, self._query_index_cache),
            )
        limit = int(os.environ.get("HEPTV2_DECODER_NUM_QUERIES", "0") or 0)
        if limit <= 0 or limit >= int(self.instance_queries.shape[0]):
            return self.instance_queries, self.instance_pos_emb
        return self.instance_queries[:limit], self.instance_pos_emb[:limit]

    def forward(self, point_feats, point_coords, valid_mask=None):
        point_feats, serialization_order, _ = self._serialize_decoder_points(
            point_feats, point_coords, valid_mask
        )
        instance_queries, instance_pos_emb = self._inference_queries()
        if point_feats.dim() == 3:
            batch_size = point_feats.shape[0]
            memory = point_feats.transpose(0, 1).contiguous()
            output = instance_queries[:, None, :].expand(-1, batch_size, -1).contiguous()
            query_pos = instance_pos_emb[:, None, :].expand(-1, batch_size, -1).contiguous()
            mask_features = rearrange(point_feats, "b h c -> b c h 1")
        else:
            memory = point_feats.unsqueeze(1)
            output = instance_queries.unsqueeze(1)
            query_pos = instance_pos_emb.unsqueeze(1)
            mask_features = rearrange(point_feats, "h c -> 1 c h 1")

        collect_aux = bool(self.deep_supervision and self.training)
        if not self.use_attn_mask and not collect_aux:
            active_layers = _env_int_limit("HEPTV2_DECODER_NUM_LAYERS", self.num_layers, self.num_layers)
            for i in range(active_layers):
                output = self.transformer_cross_attention_layers[i](
                    output,
                    memory,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None,
                    query_pos=query_pos,
                )
                if not _env_layer_enabled("HEPTV2_DECODER_SKIP_SELF_ATTN", i):
                    output = self.transformer_self_attention_layers[i](
                        output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos
                    )
                if not _env_layer_enabled("HEPTV2_DECODER_SKIP_FFN", i):
                    output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, _ = self.forward_prediction_heads(output, mask_features)
            return {
                "pred_logits": outputs_class,
                "pred_masks": outputs_mask,
                "_serialized_point_order": serialization_order,
            }

        predictions_class: List[Tensor] = []
        predictions_mask: List[Tensor] = []

        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        active_layers = _env_int_limit("HEPTV2_DECODER_NUM_LAYERS", self.num_layers, self.num_layers)
        for i in range(active_layers):
            if attn_mask is not None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output,
                memory,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=query_pos,
            )
            if not _env_layer_enabled("HEPTV2_DECODER_SKIP_SELF_ATTN", i):
                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos
                )
            if not _env_layer_enabled("HEPTV2_DECODER_SKIP_FFN", i):
                output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "_serialized_point_order": serialization_order,
        }
        if self.deep_supervision:
            out["aux_outputs"] = self._set_aux_loss(predictions_class, predictions_mask, serialization_order)
        return out

    def _set_aux_loss(self, outputs_class, outputs_seg_masks, serialization_order=None):
        aux_outputs = []
        for cls, mask in zip(outputs_class[:-1], outputs_seg_masks[:-1]):
            aux = {"pred_logits": cls, "pred_masks": mask}
            if serialization_order is not None:
                aux["_serialized_point_order"] = serialization_order
            aux_outputs.append(aux)
        return aux_outputs

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output).transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        if self.use_attn_mask:
            attn_mask = (outputs_mask.flatten(2) < 0).bool().detach()
        else:
            attn_mask = None
        return outputs_class, outputs_mask, attn_mask

    def forward_assignment_inputs(self, point_feats, point_coords, valid_mask=None):
        if self.use_attn_mask:
            raise RuntimeError("forward_assignment_inputs requires use_attn_mask=False.")
        point_feats, serialization_order, _ = self._serialize_decoder_points(
            point_feats, point_coords, valid_mask
        )
        instance_queries, instance_pos_emb = self._inference_queries()
        if point_feats.dim() == 3:
            batch_size = point_feats.shape[0]
            memory = point_feats.transpose(0, 1).contiguous()
            output = instance_queries[:, None, :].expand(-1, batch_size, -1).contiguous()
            query_pos = instance_pos_emb[:, None, :].expand(-1, batch_size, -1).contiguous()
        else:
            memory = point_feats.unsqueeze(1)
            output = instance_queries.unsqueeze(1)
            query_pos = instance_pos_emb.unsqueeze(1)

        active_layers = _env_int_limit("HEPTV2_DECODER_NUM_LAYERS", self.num_layers, self.num_layers)
        for i in range(active_layers):
            output = self.transformer_cross_attention_layers[i](
                output,
                memory,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=query_pos,
            )
            if not _env_layer_enabled("HEPTV2_DECODER_SKIP_SELF_ATTN", i):
                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos
                )
            if not _env_layer_enabled("HEPTV2_DECODER_SKIP_FFN", i):
                output = self.transformer_ffn_layers[i](output)

        decoder_output = self.decoder_norm(output).transpose(0, 1)
        return {
            "pred_logits": self.class_embed(decoder_output),
            "mask_embed": self.mask_embed(decoder_output),
            "point_feats": point_feats,
            "_serialized_point_order": serialization_order,
        }


class Transformer(nn.Module):
    def __init__(self, attn_type, in_dim, coords_dim, task, **kwargs):
        super().__init__()
        if attn_type != "hept":
            raise NotImplementedError(f"heptv2 only supports attn_type='hept', got {attn_type}")
        self.attn_type = attn_type
        self.n_layers = kwargs["n_layers"]
        self.h_dim = kwargs["h_dim"]
        self.task = task
        self.encoder_block_size, self.decoder_block_size = resolve_block_sizes(kwargs)

        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, self.h_dim),
            nn.SiLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        # Resolve per-layer serialization types. At inference we only support
        # `random_lsh` — but we preserve the config plumbing for parity.
        requested = kwargs.get("encoder_serialization_types", kwargs.get("serialization_types"))
        if requested is None:
            requested = [kwargs.get("encoder_serialization_type", "random_lsh")] * self.n_layers
        elif isinstance(requested, str):
            requested = [requested] * self.n_layers
        else:
            requested = list(requested)
            if len(requested) == 1:
                requested = requested * self.n_layers
        if len(requested) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} encoder serialization entries, got {len(requested)}."
            )
        self.encoder_serialization_types = [
            canonicalize_serialization_type(s, stage="encoder") for s in requested
        ]

        self.attns = nn.ModuleList()
        self.learnable_thres = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            layer_kwargs = dict(kwargs)
            layer_kwargs["encoder_serialization_type"] = self.encoder_serialization_types[layer_idx]
            layer_kwargs["encoder_block_size"] = self.encoder_block_size
            self.attns.append(Attn(attn_type, coords_dim, **layer_kwargs))
            self.learnable_thres.append(LearnableClf())

        self.final_norm = Qwen3RMSNorm(self.h_dim)
        self.mlp_out = MLP(
            in_channels=int(self.h_dim),
            out_channels=int(self.h_dim),
            hidden_channels=kwargs.get("mlp_out_hidden_channels", 256),
            num_layers=kwargs.get("mlp_out_num_layers", 5),
            norm="layer_norm",
            act="swish",
            norm_kwargs={"mode": "node"},
        )

        self.mask3d_head = Mask3DHead(
            hidden_dim=self.h_dim,
            num_queries=kwargs["num_queries"],
            num_decoder_layers=kwargs["num_dec_layers"],
            num_heads=kwargs["num_heads"],
            deep_supervision=kwargs["deep_supervision"],
            use_attn_mask=kwargs.get("use_attn_mask", True),
            mlp_ratio=kwargs["mlp_ratio"],
            decoder_serialization_type=kwargs.get("decoder_serialization_type", "hilbert"),
            decoder_hilbert_bits=kwargs.get("decoder_hilbert_bits", kwargs.get("encoder_hilbert_bits", 10)),
        )

        self.inter_mapping = nn.Linear(self.h_dim, 1)
        self.clf_head = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.SiLU(),
            nn.Linear(self.h_dim, 1),
        )

        # HEPT helper: the `regions` buffer (non-trainable) is populated from the
        # checkpoint. It is referenced by per-event preprocessing.
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False
        )
        self.helper_funcs = {
            "block_size": self.encoder_block_size,
            "encoder_block_size": self.encoder_block_size,
            "decoder_block_size": self.decoder_block_size,
            "regions": self.regions,
        }

    def forward(self, data):
        enc_out = self.encode(data)
        decoded = self.decode(enc_out["encoded"], data.attn_kwargs["coords"], data.attn_kwargs.get("valid_mask"))
        decoded["clf_probs"] = enc_out["clf_probs"]
        decoded["clf_logits"] = enc_out["clf_logits"]
        decoded["emb_all"] = enc_out["emb_all"]
        decoded["emb_all_labels"] = enc_out["emb_all_labels"]
        return decoded

    def encode(self, data):
        x, kwargs = data.x, data.attn_kwargs
        if _env_bool("HEPTV2_FEAT_ENCODER_FP32") and x.is_cuda:
            with autocast(device_type="cuda", enabled=False):
                encoded_x = self.feat_encoder(x.float())
        else:
            encoded_x = self.feat_encoder(x)
        fp32_layers = _env_int_set("HEPTV2_ENCODER_FP32_LAYERS")
        active_layers = _env_int_limit("HEPTV2_ENCODER_NUM_LAYERS", self.n_layers, self.n_layers)
        for i in range(active_layers):
            layer_kwargs = dict(kwargs)
            layer_kwargs["_encoder_layer_idx"] = i
            if i in fp32_layers and encoded_x.is_cuda:
                with autocast(device_type="cuda", enabled=False):
                    encoded_x = self.attns[i](encoded_x.float(), layer_kwargs)
            else:
                encoded_x = self.attns[i](encoded_x, layer_kwargs)
        if _env_bool("HEPTV2_ENCODER_HEADS_FP32"):
            with autocast(device_type="cuda", enabled=False):
                encoded_x_f = encoded_x.float()
                out = self.final_norm(encoded_x_f)
                clf_logits = self.clf_head(out)
                emb_all = F.normalize(self.mlp_out(encoded_x_f), p=2, dim=-1)
        else:
            out = self.final_norm(encoded_x)
            clf_logits = self.clf_head(out)
            emb_all = F.normalize(self.mlp_out(encoded_x), p=2, dim=-1)
        clf_probs = clf_logits.sigmoid()
        return {
            "encoded": out,
            "clf_probs": clf_probs,
            "clf_logits": clf_logits,
            "emb_all": emb_all,
            "emb_all_labels": getattr(data, "particle_id", None),
        }

    def encode_online(
        self,
        data,
        need_clf: bool = False,
        need_emb: bool = True,
        return_emb_source: bool = False,
    ):
        x, kwargs = data.x, data.attn_kwargs
        if _env_bool("HEPTV2_FEAT_ENCODER_FP32") and x.is_cuda:
            with autocast(device_type="cuda", enabled=False):
                encoded_x = self.feat_encoder(x.float())
        else:
            encoded_x = self.feat_encoder(x)
        fp32_layers = _env_int_set("HEPTV2_ENCODER_FP32_LAYERS")
        active_layers = _env_int_limit("HEPTV2_ENCODER_NUM_LAYERS", self.n_layers, self.n_layers)
        for i in range(active_layers):
            layer_kwargs = dict(kwargs)
            layer_kwargs["_encoder_layer_idx"] = i
            if i in fp32_layers and encoded_x.is_cuda:
                with autocast(device_type="cuda", enabled=False):
                    encoded_x = self.attns[i](encoded_x.float(), layer_kwargs)
            else:
                encoded_x = self.attns[i](encoded_x, layer_kwargs)
        if _env_bool("HEPTV2_ENCODER_HEADS_FP32"):
            with autocast(device_type="cuda", enabled=False):
                encoded_x_f = encoded_x.float()
                out = self.final_norm(encoded_x_f)
                if bool(need_clf):
                    clf_logits = self.clf_head(out)
                if bool(need_emb):
                    emb_all = F.normalize(self.mlp_out(encoded_x_f), p=2, dim=-1)
        else:
            out = self.final_norm(encoded_x)
        result = {"encoded": out}
        if bool(need_clf):
            if not _env_bool("HEPTV2_ENCODER_HEADS_FP32"):
                clf_logits = self.clf_head(out)
            result["clf_logits"] = clf_logits
            result["clf_probs"] = clf_logits.sigmoid()
        if bool(need_emb):
            if not _env_bool("HEPTV2_ENCODER_HEADS_FP32"):
                emb_all = F.normalize(self.mlp_out(encoded_x), p=2, dim=-1)
            result["emb_all"] = emb_all
        if bool(return_emb_source):
            result["emb_source"] = encoded_x
        return result

    def decode(self, point_feats, point_coords, valid_mask=None):
        return self.mask3d_head(point_feats, point_coords, valid_mask)

    def decode_assignment_inputs(self, point_feats, point_coords, valid_mask=None):
        return self.mask3d_head.forward_assignment_inputs(point_feats, point_coords, valid_mask)
