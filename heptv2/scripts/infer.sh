#!/bin/bash
#SBATCH -J heptv2-infer
#SBATCH -A gts-pli77-ideas_l40s
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem-per-gpu=100G
#SBATCH -q inferno
#SBATCH -o /storage/project/r-pli77-0/sgovil9/dec-sectorized/heptv2/logs/slurm_%j.out

module load anaconda3
conda activate hept_muon

export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /storage/project/r-pli77-0/sgovil9/dec-sectorized

LOG=heptv2/logs/full_inference_$(date +%Y%m%d_%H%M%S).log
mkdir -p heptv2/logs
echo "[log] writing to $LOG" | tee -a "$LOG"

python -m heptv2.run_inference --config heptv2/configs/infer.yaml 2>&1 | tee -a "$LOG"
