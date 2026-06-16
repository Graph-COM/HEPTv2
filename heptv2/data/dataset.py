"""Minimal self-contained dataset wrapper for the already-processed
`trackml-large` dataset. The processed tensor lives at
`{data_root}/processed/trackml-large/data-large.pt` and is produced by the
training code — heptv2 assumes it exists and only reads it.
"""
import os

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import BaseTransform


class TrackingTransform(BaseTransform):
    def forward(self, data):
        if data.x.shape[-1] != 15:
            data.x = torch.cat([data.x, data.layer.view(-1, 1).float() / 10.0], dim=-1)
        data.coords = torch.cat([data.pos, data.x[:, :4]], dim=-1)
        data.edge_index = None
        if hasattr(data, "hit_id") and data.hit_id is not None:
            data.hit_id = data.hit_id.long()
        else:
            data.hit_id = torch.arange(data.x.shape[0], device=data.x.device, dtype=torch.long)
        return data


class TrackmlLarge(InMemoryDataset):
    def __init__(self, root, transform=None):
        self._dataset_name = "trackml-large"
        super().__init__(str(root), transform=transform)
        self.data, self.slices, _ = torch.load(self.processed_paths[0], weights_only=False)

        sorted_evtid = self.data.evtid.argsort()
        n = len(self.data.evtid)
        n_train = int(n * 0.8)
        n_train -= n_train % 10
        n_valid = int(n * 0.1)
        self.idx_split = {
            "train": sorted_evtid[:n_train],
            "valid": sorted_evtid[n_train : n_train + n_valid],
            "test": sorted_evtid[n_train + n_valid :],
        }

        self.x_dim = 15
        self.coords_dim = 6

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw", self._dataset_name)

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed", self._dataset_name)

    @property
    def raw_file_names(self):
        return ["data21575_s0.pt"]

    @property
    def processed_file_names(self):
        return ["data-large.pt"]

    def download(self):
        return

    def process(self):
        raise RuntimeError(
            f"Processed dataset not found under {self.processed_dir}. "
            "heptv2 expects the processed tensor produced by the training code."
        )
