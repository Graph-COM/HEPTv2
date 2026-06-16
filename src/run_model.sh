#!/bin/bash

#SBATCH -J decoderhilbert_256bs_fp16nomask
#SBATCH -A gts-pli77-ideas_l40s
#SBATCH -t 48:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem-per-gpu=100G
#SBATCH -q inferno
#SBATCH -o results/decoderhilbert_256bs_fp16nomask.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sgovil9@gatech.edu

module load anaconda3
conda activate hept_muon

# Set torch allocator to reduce fragmentation (new name; old name is deprecated).
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Optional NCCL diagnostics
# export NCCL_DEBUG=INFO

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

# Important: request all GPUs for the single launcher task.
srun --gpus-per-task=2 --gpu-bind=none torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=2 \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend="c10d" \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  tracking_trainer.py -m hept
