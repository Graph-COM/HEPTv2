"""Tracking metrics: perfect / double_majority / lhc / efficiency / fake / dup.

Extracted verbatim from `gnn_tracking_prev/metrics/cluster_metrics.py` with
unused helpers inlined.
"""
from collections import Counter
from typing import Iterable, TypedDict

import numpy as np
import pandas as pd
import torch


def _zero_div_nan(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        return float("nan")


class TrackingMetrics(TypedDict, total=False):
    n_particles: int
    n_cleaned_clusters: int
    perfect: float
    double_majority: float
    lhc: float
    fake_perfect: float
    fake_double_majority: float
    fake_lhc: float
    one_way_reconstructed: float
    one_way_matched: float
    two_way_reconstructed: float
    two_way_matched: float
    ATLAS_reconstructed: float
    ATLAS_matched: float
    technical_efficiency: float
    fake_rate: float
    dup_rate: float


_NAN_RESULT: TrackingMetrics = {
    "n_particles": 0,
    "n_cleaned_clusters": 0,
    "perfect": float("nan"),
    "lhc": float("nan"),
    "double_majority": float("nan"),
    "fake_perfect": float("nan"),
    "fake_lhc": float("nan"),
    "fake_double_majority": float("nan"),
    "one_way_matched": float("nan"),
    "two_way_matched": float("nan"),
    "ATLAS_matched": float("nan"),
    "one_way_reconstructed": float("nan"),
    "two_way_reconstructed": float("nan"),
    "ATLAS_reconstructed": float("nan"),
    "technical_efficiency": float("nan"),
    "fake_rate": float("nan"),
    "dup_rate": float("nan"),
}


def _tracking_metric_df(h_df: pd.DataFrame, predicted_count_thld: int) -> pd.DataFrame:
    pid_counts = h_df[["c", "id"]].value_counts().reset_index()
    _count_key = 0 if 0 in pid_counts.columns else "count"
    grouped = pid_counts.groupby("c")
    c_df = grouped.first().rename({"id": "maj_pid", _count_key: "maj_hits"}, axis=1)
    c_df["cluster_size"] = grouped[_count_key].sum()

    unique_predicted, predicted_counts = np.unique(h_df["c"], return_counts=True)
    c_df["valid_cluster"] = (unique_predicted >= 0) & (predicted_counts >= predicted_count_thld)

    particle_props = list({"pt", "reconstructable", "eta"}.intersection(h_df.columns))
    pid_to_props = h_df[["id", *particle_props]].groupby("id")[particle_props].mean()
    c_df = c_df.merge(pid_to_props, left_on="maj_pid", right_index=True, copy=False).rename(
        columns={k: f"maj_{k}" for k in particle_props}
    )

    pid_to_count = Counter(h_df["id"])
    c_df["maj_pid_hits"] = c_df["maj_pid"].map(pid_to_count)
    c_df["maj_frac"] = (c_df["maj_hits"] / c_df["cluster_size"]).fillna(0)
    c_df["maj_pid_frac"] = (c_df["maj_hits"] / c_df["maj_pid_hits"]).fillna(0)

    c_df["perfect_match"] = (
        (c_df["maj_pid_hits"] == c_df["maj_hits"]) & (c_df["maj_frac"] > 0.99) & c_df["valid_cluster"]
    )
    c_df["double_majority"] = (
        (c_df["maj_pid_frac"] > 0.5) & (c_df["maj_frac"] > 0.5) & c_df["valid_cluster"]
    )
    c_df["lhc_match"] = (c_df["maj_frac"] > 0.75) & c_df["valid_cluster"]

    acorn = 0.5 + 1e-12
    c_df["one_way_reconstructed"] = c_df["maj_pid_frac"] >= acorn
    c_df["one_way_matched"] = c_df["maj_frac"] >= acorn
    c_df["two_way_reconstructed"] = (c_df["maj_pid_frac"] >= acorn) & (c_df["maj_frac"] >= acorn)
    c_df["two_way_matched"] = c_df["two_way_reconstructed"]
    c_df["ATLAS_matched"] = c_df["maj_frac"] >= acorn
    c_df["ATLAS_reconstructed"] = c_df["maj_pid"].isin(c_df["maj_pid"][c_df["ATLAS_matched"]])
    return c_df


def _count(c_df, h_df, c_mask, h_mask) -> TrackingMetrics:
    valid_pids = np.unique(h_df["id"][h_mask])
    n_particles = len(valid_pids)
    n_clusters = c_mask.sum().item()

    n_perfect = sum(c_df["perfect_match"][c_mask])
    n_double = sum(c_df["double_majority"][c_mask])
    n_lhc = sum(c_df["lhc_match"][c_mask])
    n_one_rec = sum(c_df["one_way_reconstructed"][c_mask])
    n_one_mat = sum(c_df["one_way_matched"][c_mask])
    n_two_mat = sum(c_df["two_way_reconstructed"][c_mask])
    n_two_rec = sum(c_df["two_way_reconstructed"][c_mask])
    n_atl_mat = sum(c_df["ATLAS_matched"][c_mask])
    n_atl_rec = sum(c_df["ATLAS_reconstructed"][c_mask])

    matched_mask = c_mask & c_df["ATLAS_matched"]
    n_matched = int(matched_mask.sum())
    if n_particles > 0:
        matched_pids = c_df.loc[matched_mask, "maj_pid"]
        matched_pids = matched_pids[matched_pids.isin(valid_pids)]
        n_reconstructed_particles = len(np.unique(matched_pids.to_numpy()))
    else:
        n_reconstructed_particles = 0
    n_dup = n_matched - n_reconstructed_particles

    return {
        "n_particles": n_particles,
        "n_cleaned_clusters": n_clusters,
        "perfect": _zero_div_nan(n_perfect, n_particles),
        "double_majority": _zero_div_nan(n_double, n_particles),
        "lhc": _zero_div_nan(n_lhc, n_clusters),
        "fake_perfect": _zero_div_nan(n_clusters - n_perfect, n_particles),
        "fake_double_majority": _zero_div_nan(n_clusters - n_double, n_particles),
        "fake_lhc": _zero_div_nan(n_clusters - n_lhc, n_clusters),
        "one_way_reconstructed": n_one_rec,
        "one_way_matched": n_one_mat,
        "two_way_reconstructed": n_two_rec,
        "two_way_matched": n_two_mat,
        "ATLAS_reconstructed": n_atl_rec,
        "ATLAS_matched": n_atl_mat,
        "technical_efficiency": _zero_div_nan(n_reconstructed_particles, n_particles),
        "fake_rate": _zero_div_nan(n_clusters - n_matched, n_clusters),
        "dup_rate": _zero_div_nan(n_dup, n_reconstructed_particles),
    }


def tracking_metrics(
    *,
    truth: np.ndarray,
    predicted: np.ndarray,
    pts: np.ndarray,
    reconstructable: np.ndarray,
    pt_thlds: Iterable[float],
    predicted_count_thld: int = 3,
    min_track_length: int = 0,
):
    assert predicted.shape == truth.shape == pts.shape
    if len(truth) == 0:
        return {pt: _NAN_RESULT for pt in pt_thlds}
    h_df = pd.DataFrame({"c": predicted, "id": truth, "pt": pts, "reconstructable": reconstructable})
    c_df = _tracking_metric_df(h_df, predicted_count_thld=predicted_count_thld)
    pid_hit_counts = h_df["id"].value_counts()
    h_df["pid_n_hits"] = h_df["id"].map(pid_hit_counts)

    result = {}
    for pt in pt_thlds:
        c_mask = (
            (c_df["maj_pt"] >= pt)
            & c_df["maj_reconstructable"]
            & c_df["valid_cluster"]
            & (c_df["maj_pid_hits"] >= min_track_length)
        )
        h_mask = (h_df["pt"] >= pt) & h_df["reconstructable"].astype(bool) & (h_df["pid_n_hits"] >= min_track_length)
        result[pt] = _count(c_df, h_df, c_mask, h_mask)
    return result


@torch.no_grad()
def compute_tracking_metrics_from_prediction(
    predicted,
    full_data,
    pt_thld: float = 0.9,
    predicted_count_thld: int = 3,
    min_track_length: int = 0,
):
    valid_mask = full_data.valid_mask.bool()
    predicted = predicted[valid_mask]
    truth = full_data.particle_id[valid_mask]
    pts = full_data.pt[valid_mask]
    reconstructable = full_data.reconstructable[valid_mask]

    res = tracking_metrics(
        truth=truth.cpu().numpy(),
        predicted=predicted.cpu().numpy(),
        pts=pts.cpu().numpy(),
        reconstructable=reconstructable.cpu().numpy(),
        pt_thlds=[pt_thld],
        predicted_count_thld=predicted_count_thld,
        min_track_length=min_track_length,
    )
    metrics = res[pt_thld]
    return {("dm" if k == "double_majority" else k): float(v) for k, v in metrics.items()}
