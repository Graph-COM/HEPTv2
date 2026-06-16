# heptv2 — minimal inference + single-GPU training

A self-contained subset of `src/` that loads a checkpoint trained by the full
pipeline and runs encoder → per-sub-event decoder → decoder_kmeans
post-processing → tracking metrics on `trackml-large`.

Also supports single-GPU training (finetune or from-scratch) with the same
loss composition as `src/tracking_trainer.py`: DETR-style set loss on masks
+ pos-weighted BCE on the encoder classifier + chunked InfoNCE on the encoder
embedding. Per-epoch validation also runs the full decoder_kmeans →
`tracking_metrics` path, so `dm` / `technical_efficiency` / `fake_rate` /
`dup_rate` appear in the valid log alongside losses.

No DDP, no wandb, no efficiency plots — use `src/` for those.

**Supports**
- HEPT attention with `encoder_serialization_types = random_lsh` (all layers)
- `decoder_serialization_type = hilbert | none`
- `use_attn_mask: true | false` (flex_attention path)
- AMP (`bfloat16` / `float16`)
- `decoder_kmeans` postprocessing

**Drops** (vs. `src/`): Muon/AdamW-only optimizer set, two schedulers, no DDP,
no wandb, no efficiency plots, no GNN reference eval, no Gumbel sampling,
no freeze/reinit helpers, no `encoder_kmeans` / `encoder_dbscan` /
`decoder_assignments` postprocessing modes, no non-`random_lsh` encoder
serializations, no `phi` decoder serialization.

**Parameter names** match `src/models/baselines/transformer.py`. The loader
strips `module.` and `_orig_mod.` prefixes, then `load_state_dict(strict=False)`
and prints missing/unexpected keys.

## Layout

```
heptv2/
├── run_inference.py      # inference CLI
├── run_train.py          # training CLI
├── model/
│   ├── transformer.py    # Transformer, Mask3DHead encoder+decoder
│   ├── attention.py      # HEPTAttention + bucketing
│   └── pos_emb.py        # PELearned + gauss_B buffer stub
├── data/
│   ├── dataset.py        # TrackmlLarge loader + TrackingTransform
│   └── preprocess.py     # eta filter, padding, sub-event split (± targets)
├── training/
│   ├── train.py          # train_one_batch, eval_one_batch, run_one_epoch
│   ├── train_utils.py    # seed/AMP/optimizer/scheduler factories
│   └── losses.py         # SetCriterion, HungarianMatcher, dice/focal
├── eval/
│   ├── postprocess.py    # decoder_kmeans assignment + k-means refinement
│   └── metrics.py        # tracking_metrics (perfect, dm, lhc, eff/fake/dup)
├── utils/
│   ├── block_size_utils.py  # block-size + sub-event length math
│   ├── hash_utils.py        # E2LSH, get_regions, quantile_partition
│   └── serialization.py     # Optional decoder ordering
├── configs/
│   ├── infer.yaml        # inference config
│   └── train.yaml        # training config
└── scripts/
    ├── infer.sh          # full inference (sbatch)
    ├── train.sh          # full training (sbatch, 24h)
    └── smoke_train.sh    # 1-epoch, 3-batch smoke (sbatch)
```

## Inference

```bash
module load anaconda3 && conda activate hept_muon
cd /storage/project/r-pli77-0/sgovil9/dec-sectorized
python -m heptv2.run_inference --config heptv2/configs/infer.yaml
```

Set `eval.limit_events: 3` in `configs/infer.yaml` for a short smoke test, or
submit `sbatch heptv2/scripts/infer.sh` for the full run.

## Training (single GPU)

```bash
python -m heptv2.run_train --config heptv2/configs/train.yaml
```

The default config finetunes from `03_07-14_20_07.6687_trans_hept_42_fp16model_newdm`.
To train from scratch, set `resume: null`. Checkpoints land under
`data_root/logs/<timestamp>_<model_name>_<seed>_<note>/`:
- `best_checkpoint.pt` — whenever the configured `best_metric_key` improves
- `last_checkpoint.pt` — every epoch

### Per-epoch eval metrics

`eval.compute_metrics: true` (default) runs the same
`decoder_kmeans_postprocess` + `compute_tracking_metrics_from_prediction`
path as `run_inference.py` on every validation pass. The per-epoch valid
log prints `dm`, `technical_efficiency`, `fake_rate`, `dup_rate`,
`n_particles`, `n_cleaned_clusters` alongside the losses.

Checkpoint selection is controlled by `best_metric_key` / `best_metric_mode`:

```yaml
# default — back-compat with the loss-only loop
best_metric_key: total_loss
best_metric_mode: min

# once you trust the metric loop, select by DM instead:
best_metric_key: dm
best_metric_mode: max
```

### Quick smoke

```bash
sbatch heptv2/scripts/smoke_train.sh
```

### Full finetune

```bash
sbatch heptv2/scripts/train.sh
```

### Differences vs `src/tracking_trainer.py`
- No DDP, no wandb, no efficiency plots, no GNN reference eval
- No freeze/reinit_queries/reinit_regions/gumbel branches
- Only `step` and `warmup_hold_cosine` schedulers
- Only `muon`, `adamw`, `adam` optimizers
