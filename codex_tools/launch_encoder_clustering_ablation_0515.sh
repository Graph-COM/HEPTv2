#!/usr/bin/env bash
set -eo pipefail

source /usr/scratch/smiao35/anaconda3/etc/profile.d/conda.sh
conda activate lse-torch-fa

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source codex_tools/env_phase48_scatter_amp_best.sh

LOGDIR="${LOGDIR:-codex_logs/decoder_ablation_0515}"
mkdir -p "$LOGDIR"

setsid python -u codex_tools/encoder_clustering_ablation.py \
  --device cuda:0 \
  --out-dir "$LOGDIR/encoder_clustering_full" \
  > "$LOGDIR/encoder_clustering_full.log" 2>&1 &
echo $! > "$LOGDIR/encoder_clustering_full.pid"
echo "encoder_clustering_full cuda:0 pid=$(cat "$LOGDIR/encoder_clustering_full.pid") log=$LOGDIR/encoder_clustering_full.log"
