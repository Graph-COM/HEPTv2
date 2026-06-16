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

CONFIG="heptv2/configs/finetune_phase48_scatter_amp_train.yaml"

COMMON_OVERRIDES=(
  --override "trainable_prefixes=[mask3d_head]"
  --override "loss_kwargs.clf_loss=0"
  --override "loss_kwargs.nce_weight=0"
  --override "resume_optimizer=false"
  --override "resume_lr_scheduler=false"
  --override "resume_start_epoch=0"
)

launch_job() {
  local tag="$1"
  local device="$2"
  local lr="$3"
  local num_sub_events="$4"
  local overlap="$5"
  local logfile="$LOGDIR/${tag}.log"
  local pidfile="$LOGDIR/${tag}.pid"

  setsid python -u -m heptv2.run_train \
    --config "$CONFIG" \
    "${COMMON_OVERRIDES[@]}" \
    --override "device=${device}" \
    --override "note=${tag}" \
    --override "optimizer_kwargs.lr=${lr}" \
    --override "num_sub_events=${num_sub_events}" \
    --override "eval.num_sub_events=${num_sub_events}" \
    --override "model_kwargs.decoder_overlap=${overlap}" \
    > "$logfile" 2>&1 &
  echo $! > "$pidfile"
  echo "${tag} ${device} lr=${lr} sub_events=${num_sub_events} overlap=${overlap} pid=$(cat "$pidfile") log=${logfile}"
}

launch_job "ablate_2A_full_decoder_nosub_lr1e-6_cuda1" "cuda:1" "1.0e-6" "1" "false"
launch_job "ablate_2A_full_decoder_nosub_lr1e-5_cuda2" "cuda:2" "1.0e-5" "1" "false"

launch_job "ablate_3_sector2_lr1e-5_cuda3" "cuda:3" "1.0e-5" "2" "false"
launch_job "ablate_3_sector4_lr1e-5_cuda4" "cuda:4" "1.0e-5" "4" "false"
launch_job "ablate_3_sector5_lr1e-5_cuda5" "cuda:5" "1.0e-5" "5" "false"

launch_job "ablate_4_sector3_overlap_lr1e-6_cuda6" "cuda:6" "1.0e-6" "3" "true"
launch_job "ablate_4_sector3_overlap_lr1e-5_cuda7" "cuda:7" "1.0e-5" "3" "true"
