#!/bin/bash
# Multi-GPU training on one node with torchrun (DDP).
#   sbatch --gres=gpu:4 slurm/job_multigpu.sbatch scripts/train.py --out-dir exp/prefix ...
# Per-GPU --batch-size stays the same, so the effective batch grows with the GPU count.
#SBATCH --job-name=tsasr-ddp
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source config.sh
NGPU=$(nvidia-smi -L | wc -l)
echo "+ torchrun --nproc_per_node $NGPU $*"
torchrun --standalone --nproc_per_node "$NGPU" "$@"
