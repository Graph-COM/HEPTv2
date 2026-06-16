import torch
import torch.nn.functional as F
from einops import rearrange

from torch_scatter import scatter_mean
from models.model_utils.hash_utils import quantile_partition


def apply_setting_mask(data, condition):
    data.x = data.x[condition]
    data.layer = data.layer[condition]
    data.particle_id = data.particle_id[condition]
    data.pt = data.pt[condition]
    data.reconstructable = data.reconstructable[condition]
    data.pos = data.pos[condition]
    data.coords = data.coords[condition]
    for attr in ("eta", "phi", "hit_id"):
        if hasattr(data, attr) and getattr(data, attr) is not None:
            setattr(data, attr, getattr(data, attr)[condition])
    data.edge_index = None
    return data


def build_cluster_masks(data, thld=0.9):
    pids = data.particle_id
    unique_pids, inv_idx = torch.unique(pids, return_inverse=True)
    pt_list = scatter_mean(data.pt, inv_idx, dim=0)
    rec_list = scatter_mean(data.reconstructable.float(), inv_idx, dim=0)

    good_mask = (pt_list > thld) & (rec_list == 1.0)
    good_idx = torch.nonzero(good_mask, as_tuple=False).squeeze()

    full_masks = (inv_idx[None, :] == torch.arange(len(unique_pids), device=inv_idx.device)[:, None]).long().unsqueeze(-1)

    masks = (inv_idx[None, :] == good_idx[:, None]).long().unsqueeze(-1)

    labels = torch.zeros(len(good_idx), dtype=torch.long, device=pids.device)
    y_true = (data.pt >= thld) & (data.reconstructable == 1)
    return labels, masks, full_masks, y_true


def pad_to_fixed_len(data, masks, full_masks, fixed_len, block_size):
    if fixed_len % block_size != 0:
        fixed_len = ((fixed_len + block_size - 1) // block_size) * block_size

    raw_n = data.x.shape[0]
    if raw_n > fixed_len:
        raise ValueError(f"Input length {raw_n} exceeds fixed_seq_len {fixed_len}")

    pad_n = fixed_len - raw_n
    valid_mask = torch.cat(
        [
            torch.ones(raw_n, dtype=torch.bool, device=data.x.device),
            torch.zeros(pad_n, dtype=torch.bool, device=data.x.device),
        ]
    )

    if pad_n > 0:
        data.x = F.pad(data.x, (0, 0, 0, pad_n))
        data.coords = F.pad(data.coords, (0, 0, 0, pad_n), value=0.0)

    return data, masks, full_masks, valid_mask


def _pad_first_dim(tensor, pad_n, value=0):
    if pad_n <= 0:
        return tensor
    if tensor.dim() == 1:
        return F.pad(tensor, (0, pad_n), value=value)
    pad = [0, 0] * (tensor.dim() - 1) + [0, pad_n]
    return F.pad(tensor, pad, value=value)


def pad_to_fixed_len_all(data, fixed_len, block_size):
    if fixed_len % block_size != 0:
        fixed_len = ((fixed_len + block_size - 1) // block_size) * block_size

    raw_n = data.x.shape[0]
    if raw_n > fixed_len:
        raise ValueError(f"Input length {raw_n} exceeds fixed_seq_len {fixed_len}")

    pad_n = fixed_len - raw_n
    valid_mask = torch.cat(
        [
            torch.ones(raw_n, dtype=torch.bool, device=data.x.device),
            torch.zeros(pad_n, dtype=torch.bool, device=data.x.device),
        ]
    )

    if pad_n > 0:
        data.x = _pad_first_dim(data.x, pad_n, value=0.0)
        data.coords = _pad_first_dim(data.coords, pad_n, value=0.0)
        data.pos = _pad_first_dim(data.pos, pad_n, value=0.0)
        data.layer = _pad_first_dim(data.layer, pad_n, value=0)
        data.particle_id = _pad_first_dim(data.particle_id, pad_n, value=-1)
        data.pt = _pad_first_dim(data.pt, pad_n, value=0.0)
        data.reconstructable = _pad_first_dim(data.reconstructable, pad_n, value=0)
        for attr, pad_value in (("eta", 0.0), ("phi", 0.0), ("hit_id", -1)):
            if hasattr(data, attr) and getattr(data, attr) is not None:
                setattr(data, attr, _pad_first_dim(getattr(data, attr), pad_n, value=pad_value))

    return data, valid_mask


def build_region_indices(coords, valid_mask, regions):
    coords_for_sort = coords.clone()
    coords_for_sort[~valid_mask] = float("inf")
    sorted_eta_idx = torch.argsort(coords_for_sort[..., 0], dim=-1)
    sorted_phi_idx = torch.argsort(coords_for_sort[..., 1], dim=-1)
    regions_h = rearrange(regions, "c a h -> a (c h)")
    region_indices_eta = quantile_partition(sorted_eta_idx, regions_h[0][:, None])
    region_indices_phi = quantile_partition(sorted_phi_idx, regions_h[1][:, None])
    return [region_indices_eta, region_indices_phi], regions_h
