import torch
from torch import nn
from torch_geometric.nn import MLP
from ..attention import HEPTAttention
from ..model_utils.hash_utils import get_regions
from ..model_utils.serialization_utils import canonicalize_serialization_type, compute_serialization_order
from ..model_utils.window_utils import get_pe_func
from ..model_utils.pos_emb import PositionEmbeddingCoordsSine
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import copy
from typing import List, Optional
from torch import Tensor
import torch.nn.functional as F
from utils.block_size_utils import resolve_block_sizes

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except ImportError:
    flex_attention = None
    create_block_mask = None

# TODO: wandb tuning
# TODO: softmax loss
# TODO: sigmoid thres then argmax for final assignment; assignment with class logits
# TODO: more coords in PE?

# TODO: Better assignment loss that gumbel can help with
# TODO: no gumbel
# TODO: No h_dim // 2


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(input_dtype) * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, act_fn=F.silu):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Mask3DHead(nn.Module):
    """
    Mask3D head for 3D semantic instance segmentation.
    Given per-point features from your backbone, it predicts:
      - instance masks: [Q, N]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_decoder_layers: int,
        num_heads: int,
        deep_supervision: bool,
        use_attn_mask: bool = False,
        use_ckpt: bool = False,
        mlp_ratio: float = 4.0,
        decoder_serialization_type: str = "phi",
        decoder_hilbert_bits: int = 10,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_attn_mask = use_attn_mask
        self.use_ckpt = use_ckpt
        self.decoder_serialization_type = canonicalize_serialization_type(
            decoder_serialization_type, stage="decoder"
        )
        self.decoder_hilbert_bits = decoder_hilbert_bits
        # learnable instance queries (Q, D)
        # self.instance_queries_nn = nn.Parameter(torch.randn(num_queries, hidden_dim))
        # self.instance_pos_emb_nn = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.instance_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.instance_pos_emb = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.pos_enc = PositionEmbeddingCoordsSine(
            pos_type="fourier", d_pos=hidden_dim, gauss_scale=1.0, d_in=2, normalize=True
        )

        # Transformer decoder
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=hidden_dim,
        #     nhead=num_heads,
        #     dim_feedforward=hidden_dim * 4,
        # )
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.num_heads = num_heads
        self.num_layers = num_decoder_layers
        # decoder_layer = TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4, 0.0, "relu", False)
        # decoder_norm = nn.LayerNorm(hidden_dim)
        # self.decoder = TransformerDecoder(
        #     decoder_layer,
        #     num_decoder_layers,
        #     decoder_norm,
        #     return_intermediate=deep_supervision,
        # )

        # mask head
        mlp_hidden_dim = max(1, int(hidden_dim * mlp_ratio))
        self.mask_embed = Qwen3MLP(hidden_dim, mlp_hidden_dim, act_fn=F.silu)
        self.decoder_norm = Qwen3RMSNorm(hidden_dim)
        # class head
        self.class_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        # self.class_embed_new = nn.Linear(hidden_dim, 1)
        # self.class_embed_new = MaskMLP(hidden_dim, hidden_dim, 1, 3)

        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):

            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dropout=0.0,
                    normalize_before=True,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dropout=0.0,
                    normalize_before=True,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=mlp_hidden_dim,
                    dropout=0.0,
                    normalize_before=True,
                )
            )

    def reinit_queries(self, num_queries, hidden_dim):
        """Reinitialize instance queries and position embeddings."""
        self.instance_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.instance_pos_emb = nn.Parameter(torch.randn(num_queries, hidden_dim))
        print(f"Reinitialized instance queries and position embeddings with shape {num_queries}, {hidden_dim}")

    def _serialize_decoder_points(
        self, point_feats: torch.Tensor, point_coords: torch.Tensor, valid_mask: Optional[torch.Tensor]
    ):
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

    @staticmethod
    def _restore_point_order(mask_logits: torch.Tensor, inverse_order: torch.Tensor) -> torch.Tensor:
        identity_order = torch.arange(inverse_order.numel(), device=inverse_order.device, dtype=inverse_order.dtype)
        if torch.equal(inverse_order, identity_order):
            return mask_logits
        return mask_logits.index_select(2, inverse_order)

    def forward(self, point_feats: torch.Tensor, point_coords: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        """
        Args:
            point_feats: [N, D] per-point embeddings
        Returns:
            mask_logits: [Q, N]
        """
        point_feats, serialization_order, _ = self._serialize_decoder_points(
            point_feats, point_coords, valid_mask
        )

        N, D = point_feats.shape
        Q = self.instance_queries.shape[0]

        # prepare for Transformer: shape [N, B, D], with B=1
        memory = point_feats.unsqueeze(1)  # [N, B, D]

        # prepare queries: [Q, B, D]
        output = self.instance_queries.unsqueeze(1)  # [Q, B, D]
        query_pos = self.instance_pos_emb.unsqueeze(1)  # [Q, B, D]

        # point_coords = point_coords[..., :2]
        # min_coords = point_coords.min(dim=0)[0][None]
        # max_coords = point_coords.max(dim=0)[0][None]
        # pos = self.pos_enc(point_coords[None], input_range=(min_coords, max_coords)).permute(2, 0, 1)  # [N, B, D]
        # decode
        # decoded = self.decoder(tgt=queries + query_pos, memory=memory + pos)  # [Q, B, D]
        # decoded = decoded.permute(1, 0, 2)  # [B, Q, D]

        predictions_class = []
        predictions_mask = []

        mask_features = rearrange(point_feats, "h c -> 1 c h 1")
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            if attn_mask is not None:
                # Avoid all-True rows, which would block every key for a query.
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            if self.use_ckpt and torch.is_grad_enabled():
                # Keep only `output` as checkpoint input. Capturing large constant tensors
                # (memory/attn_mask/query_pos) in closures avoids storing them per-checkpoint.
                def _cross_attn_block(tgt):
                    return self.transformer_cross_attention_layers[i](
                        tgt,
                        memory,
                        memory_mask=attn_mask,
                        memory_key_padding_mask=None,  # here we do not apply masking on padded region
                        pos=None,
                        query_pos=query_pos,
                    )

                def _self_attn_block(tgt):
                    return self.transformer_self_attention_layers[i](
                        tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos
                    )

                output = checkpoint(_cross_attn_block, output, use_reentrant=False)
                output = checkpoint(_self_attn_block, output, use_reentrant=False)
                output = checkpoint(self.transformer_ffn_layers[i], output, use_reentrant=False)
            else:
                output = self.transformer_cross_attention_layers[i](
                    output,
                    memory,
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=None,
                    query_pos=query_pos,
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos
                )

                # FFN
                output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "_serialized_point_order": serialization_order,
            "aux_outputs": self._set_aux_loss(predictions_class, predictions_mask, serialization_order),
        }

        return out

    def _set_aux_loss(
        self,
        outputs_class: List[torch.Tensor],
        outputs_seg_masks: List[torch.Tensor],
        serialization_order: Optional[torch.Tensor] = None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        aux_outputs = []
        for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1]):
            aux = {"pred_logits": a, "pred_masks": b}
            if serialization_order is not None:
                aux["_serialized_point_order"] = serialization_order
            aux_outputs.append(aux)
        return aux_outputs

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        if self.use_attn_mask:
            # Keep a shared [B, Q, K] mask and broadcast it over heads inside cross-attention.
            # `sigmoid(x) < 0.5` is equivalent to `x < 0`, so avoid materializing sigmoid activations here.
            attn_mask = (outputs_mask.flatten(2) < 0).bool()
            attn_mask = attn_mask.detach()
        else:
            attn_mask = None

        return outputs_class, outputs_mask, attn_mask


class LearnableClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.sharpness = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.threshold = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        clf_soft = torch.sigmoid(self.sharpness * (x - self.threshold))
        clf_hard = (clf_soft > 0.1).float()  # Thresholding at 0.2 for hard selection
        return clf_hard, clf_soft


def gumbel(att_log_logit, training, do_gumbel):
    if not do_gumbel:
        training = False
    temp = 1
    if training:
        random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        y_soft = ((att_log_logit + random_noise) / temp).sigmoid()
        y_hard = (y_soft > 0.1).float()
        y_hard = y_hard - y_soft.detach() + y_soft
    else:
        y_soft = att_log_logit.sigmoid()
        y_hard = (y_soft > 0.1).float()
        y_logit = att_log_logit
    return y_hard, y_soft, y_logit


class Transformer(nn.Module):
    def __init__(self, attn_type, in_dim, coords_dim, task, **kwargs):
        super().__init__()
        self.attn_type = attn_type
        self.n_layers = kwargs["n_layers"]
        self.h_dim = kwargs["h_dim"]
        self.task = task
        self.use_ckpt = kwargs.get("use_ckpt", False)
        self.gumbel = kwargs["gumbel"]
        self.encoder_block_size, self.decoder_block_size = resolve_block_sizes(kwargs)

        # Add parameters for gradual noise introduction
        self.current_epoch = 0
        self.total_epochs = 250
        self.noise_schedule = kwargs.get("noise_schedule", "linear")  # "linear" or "cosine"
        self.noise_start_epoch = 0  # When to start introducing noise
        # When to reach full noise (as fraction of total epochs)
        self.noise_full_epoch = 0.1

        # Only support tracking task - remove pileup-related code

        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, self.h_dim),
            nn.SiLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        requested_serializations = kwargs.get("encoder_serialization_types", kwargs.get("serialization_types"))
        if requested_serializations is None:
            explicit_single_serialization = kwargs.get("encoder_serialization_type")
            if explicit_single_serialization is not None:
                requested_serializations = [explicit_single_serialization]
            else:
                mixed_pattern = ["zorder", "trans_zorder", "hilbert", "trans_hilbert"]
                repeats = (self.n_layers + len(mixed_pattern) - 1) // len(mixed_pattern)
                requested_serializations = (mixed_pattern * repeats)[: self.n_layers]
        elif isinstance(requested_serializations, str):
            requested_serializations = [requested_serializations]
        else:
            requested_serializations = list(requested_serializations)

        if len(requested_serializations) == 1:
            requested_serializations = requested_serializations * self.n_layers
        elif len(requested_serializations) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} encoder serialization entries, got {len(requested_serializations)}."
            )

        self.encoder_serialization_types = [
            canonicalize_serialization_type(serialization_type, stage="encoder")
            for serialization_type in requested_serializations
        ]

        self.attns = nn.ModuleList()
        self.learnable_thres = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            layer_kwargs = kwargs.copy()
            layer_kwargs["encoder_serialization_type"] = self.encoder_serialization_types[layer_idx]
            layer_kwargs["encoder_block_size"] = self.encoder_block_size
            self.attns.append(Attn(attn_type, coords_dim, **layer_kwargs))
            self.learnable_thres.append(LearnableClf())

        self.final_norm = Qwen3RMSNorm(self.h_dim)
        mlp_out_hidden_channels = kwargs.get("mlp_out_hidden_channels", 256)
        mlp_out_num_layers = kwargs.get("mlp_out_num_layers", 5)
        self.mlp_out = MLP(
            in_channels=int(self.h_dim),
            out_channels=int(self.h_dim),
            hidden_channels=mlp_out_hidden_channels,
            num_layers=mlp_out_num_layers,
            norm="layer_norm",
            # act="tanh",
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
            use_ckpt=self.use_ckpt,
            mlp_ratio=kwargs["mlp_ratio"],
            decoder_serialization_type=kwargs.get("decoder_serialization_type", "phi"),
            decoder_hilbert_bits=kwargs.get("decoder_hilbert_bits", kwargs.get("encoder_hilbert_bits", 10)),
        )

        self.inter_mapping = nn.Linear(self.h_dim, 1)
        self.clf_head = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.SiLU(),
            nn.Linear(self.h_dim, 1),
        )

        self.helper_funcs = {}
        # Only HEPT-specific setup
        if self.attn_type == "hept":
            # Keep the legacy alias for older callers that still inspect `block_size`.
            self.helper_funcs["block_size"] = self.encoder_block_size
            self.helper_funcs["encoder_block_size"] = self.encoder_block_size
            self.helper_funcs["decoder_block_size"] = self.decoder_block_size
            self.regions = nn.Parameter(
                get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False
            )
            self.helper_funcs["regions"] = self.regions
        else:
            raise NotImplementedError(f"Only HEPT attention is supported, got {self.attn_type}")

        # Remove pileup-related code since we only support tracking

    def reinit_regions(self, num_regions, n_hashes, num_heads):
        """Reinitialize regions parameter with new parameters."""
        if self.attn_type == "hept":
            new_regions = get_regions(num_regions, n_hashes, num_heads)
            self.regions = nn.Parameter(new_regions, requires_grad=False)
            self.helper_funcs["regions"] = self.regions
        else:
            raise NotImplementedError("Region reinitialization only supported for HEPT attention")

    def set_epoch(self, epoch):
        """Set current epoch for noise scheduling."""
        self.current_epoch = epoch

    def freeze_mask3d_head(self):
        """Freeze all parameters in the mask3d_head module for fine-tuning, except instance_pos_emb and instance_queries."""
        for name, param in self.mask3d_head.named_parameters():
            param.requires_grad = False
            # if name not in ['instance_pos_emb', 'instance_queries']:
            #     param.requires_grad = False
            # else:
            #     param.requires_grad = True
            #     print(f"Keeping {name} trainable for fine-tuning")
        print("Mask3D head weights frozen for fine-tuning")

    def freeze_encoder_modules(self):
        """Freeze all encoder modules except clf_head and inter_mapping for fine-tuning."""
        # Freeze feat_encoder
        for param in self.feat_encoder.parameters():
            param.requires_grad = False

        # Freeze attention layers
        for param in self.attns.parameters():
            param.requires_grad = False

        # Freeze dropout (no parameters to freeze)
        # Freeze W linear layer
        for param in self.W.parameters():
            param.requires_grad = False

        # Freeze final_norm
        for param in self.final_norm.parameters():
            param.requires_grad = False

        # Keep inter_mapping and clf_head trainable (they remain unfrozen)
        print("Encoder modules frozen for fine-tuning (kept clf_head trainable)")

    def forward(self, data):
        enc_out = self.encode(data)
        decoded_out = self.decode(enc_out["encoded"], data.attn_kwargs["coords"], data.attn_kwargs.get("valid_mask"))
        decoded_out["clf_probs"] = enc_out["clf_probs"]
        decoded_out["clf_logits"] = enc_out["clf_logits"]
        decoded_out["emb_all"] = enc_out["emb_all"]
        decoded_out["emb_all_labels"] = enc_out["emb_all_labels"]
        return decoded_out

    def encode(self, data):
        x, kwargs = data.x, data.attn_kwargs

        encoded_x = self.feat_encoder(x)
        for i in range(self.n_layers):
            if self.use_ckpt and torch.is_grad_enabled():
                encoded_x = checkpoint(self.attns[i], encoded_x, kwargs, [None, None], use_reentrant=False)
            else:
                encoded_x = self.attns[i](encoded_x, kwargs, [None, None])

        out = self.final_norm(encoded_x)
        _, clf_probs, clf_logits = gumbel(self.clf_head(out), self.training, self.gumbel)

        return {
            "encoded": out,
            "clf_probs": clf_probs,
            "clf_logits": clf_logits,
            "emb_all": F.normalize(self.mlp_out(encoded_x), p=2, dim=-1),
            "emb_all_labels": data.particle_id,
        }

    def decode(self, point_feats, point_coords, valid_mask=None):
        return self.mask3d_head(point_feats, point_coords, valid_mask)


def make_new_pred_masks(pred_good_masks, pred_bad_masks, y_hard):
    """This function modifies the pred_masks to only include the masks corresponding to y_true.
    It also ensures that the masks are in the order of y_hard.
    """
    ttt = torch.empty(
        (pred_good_masks.shape[0], pred_good_masks.shape[1], y_hard.shape[0], pred_good_masks.shape[3]),
        device=pred_good_masks.device,
    )
    ttt[:, :, y_hard == 1] = pred_good_masks
    ttt[:, :, y_hard == 0] = pred_bad_masks
    return ttt


def make_new_pred_masks_new(pred_good, pred_bad, y_hard):
    """
    pred_good: [B, C, G, D]
    pred_bad:  [B, C, B_bad, D]
    y_hard:    [T]  of 0/1, with T = G + B_bad

    returns: [B, C, T, D] where for each t∈[0,T),
             out[:,:,t,:] = pred_good[:,:,j,:] if y_hard[t]==1
                             pred_bad[:,:,k,:] otherwise
    """
    B, C, G, D = pred_good.shape
    _, _, B_bad, _ = pred_bad.shape
    T = y_hard.size(0)
    device = pred_good.device

    # 1) compute, for each output slot t, its index within good/bad
    y = y_hard.to(torch.int64)
    good_rank = (torch.cumsum(y, dim=0) - 1).clamp(min=0)  # [T]
    bad_rank = (torch.cumsum(1 - y, dim=0) - 1).clamp(min=0)  # [T]

    # 2) build the final “cat‐index”:
    #    if y_hard[t]==1 → take slot good_rank[t] from pred_good (i.e. index < G)
    #    else           → take slot bad_rank[t]  from pred_bad  (i.e. index ≥ G)
    idx = torch.where(y == 1, good_rank, good_rank.new_full((T,), G) + bad_rank)  # 0…G-1  # G…G+B_bad-1

    # 3) cat and reorder in one go
    pred_cat = torch.cat([pred_good, pred_bad], dim=2)  # [B, C, G+B_bad, D]
    out = pred_cat.index_select(2, idx.to(device))  # [B, C, T, D]

    return out


class Attn(nn.Module):
    def __init__(self, attn_type, coords_dim, **kwargs):
        super().__init__()
        self.attn_type = attn_type
        self.model_dim = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        if self.model_dim % self.num_heads != 0:
            raise ValueError(f"h_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads}).")
        self.dim_per_head = self.model_dim // self.num_heads

        self.w_q = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.w_k = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.w_v = nn.Linear(self.model_dim, self.model_dim, bias=False)

        if attn_type == "hept":
            # +2 for data.pos
            self.attn = HEPTAttention(self.dim_per_head + coords_dim, head_dim=self.dim_per_head, model_dim=self.model_dim, **kwargs)
        else:
            raise NotImplementedError(f"Only HEPT attention is supported, got {attn_type}")

        self.dropout = nn.Dropout(0.0)
        self.norm1 = Qwen3RMSNorm(self.model_dim)
        self.norm2 = Qwen3RMSNorm(self.model_dim)
        self.q_norm = Qwen3RMSNorm(self.dim_per_head)
        self.k_norm = Qwen3RMSNorm(self.dim_per_head)
        mlp_ratio = kwargs["mlp_ratio"]
        mlp_hidden_dim = max(1, int(self.model_dim * mlp_ratio))
        self.ff = Qwen3MLP(self.model_dim, mlp_hidden_dim, act_fn=F.silu)

        # eta/phi from data.pos use the same weights as they are used to calc dR
        self.w_rpe = nn.Linear(kwargs["num_w_per_dist"] * (coords_dim - 1), self.num_heads * self.dim_per_head)
        self.pe_func = get_pe_func(kwargs["pe_type"], coords_dim, kwargs)

    def forward(self, x, kwargs, clf_res):
        pe = kwargs["coords"] if self.pe_func is None else self.pe_func(kwargs["coords"])
        x_pe = x + pe if self.pe_func is not None else x
        x_normed = self.norm1(x_pe)
        q = self.w_q(x_normed).view(-1, self.num_heads, self.dim_per_head)
        k = self.w_k(x_normed).view(-1, self.num_heads, self.dim_per_head)
        v = self.w_v(x_normed).view(-1, self.num_heads, self.dim_per_head)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(-1, self.num_heads * self.dim_per_head)
        k = k.view(-1, self.num_heads * self.dim_per_head)
        v = v.view(-1, self.num_heads * self.dim_per_head)
        aggr_out = self.attn(q, k, v, pe=pe, w_rpe=self.w_rpe, clf_res=clf_res, **kwargs)

        x = x + self.dropout(aggr_out)
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "silu":
        return F.silu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/silu, not {activation}.")


class MaskMLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.silu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation="silu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = Qwen3RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.q_norm = Qwen3RMSNorm(int(d_model // nhead))
        self.k_norm = Qwen3RMSNorm(int(d_model // nhead))

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _flex_cross_attn(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory

        w = self.multihead_attn.in_proj_weight
        b = self.multihead_attn.in_proj_bias
        embed_dim = w.shape[1]
        w_q, w_k, w_v = w.split(embed_dim, dim=0)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.split(embed_dim)
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

        out = flex_attention(q, k, v, block_mask=block_mask)
        out = out.permute(2, 0, 1, 3).reshape(q_len, bsz, embed_dim)
        return self.multihead_attn.out_proj(out)

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        if flex_attention is not None and (memory_mask is None or create_block_mask is not None):
            tgt2 = self._flex_cross_attn(tgt2, memory, memory_mask=memory_mask, pos=pos, query_pos=query_pos)
        else:
            raise NotImplementedError("Only flex attention is supported in this implementation.")
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation="silu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = Qwen3RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.q_norm = Qwen3RMSNorm(int(d_model // nhead))
        self.k_norm = Qwen3RMSNorm(int(d_model // nhead))

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _flex_self_attn(self, tgt, query_pos: Optional[Tensor] = None):
        q = self.with_pos_embed(tgt, query_pos)
        k = q
        v = tgt

        w = self.self_attn.in_proj_weight
        b = self.self_attn.in_proj_bias
        embed_dim = w.shape[1]
        w_q, w_k, w_v = w.split(embed_dim, dim=0)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.split(embed_dim)
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

        out = flex_attention(q, k, v)
        out = out.permute(2, 0, 1, 3).reshape(q_len, bsz, embed_dim)
        return self.self_attn.out_proj(out)

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        head_dim = self.self_attn.embed_dim // self.self_attn.num_heads
        if flex_attention is not None and tgt_mask is None and head_dim >= 16:
            tgt2 = self._flex_self_attn(tgt2, query_pos=query_pos)
        else:
            raise NotImplementedError("Only flex attention is supported in this implementation.")
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="silu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.mlp = Qwen3MLP(d_model, dim_feedforward, act_fn=_get_activation_fn(activation))
        self.dropout = nn.Dropout(dropout)

        self.norm = Qwen3RMSNorm(d_model)

        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.mlp(tgt2)
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        return self.forward_pre(tgt)
