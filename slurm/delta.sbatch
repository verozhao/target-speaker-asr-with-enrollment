#!/bin/bash
# NCSA Delta, course allocation. One GPU (A40 48 GB or A100 40 GB, whichever frees up first).
# Before the first job, set your account once (see docs/delta.md):
#   echo 'export SBATCH_ACCOUNT=<account from the course run.sbatch>' >> ~/.bashrc && source ~/.bashrc
# Submit from the repository root:
#   sbatch slurm/delta.sbatch bash scripts/run.sh train_prefix
#   sbatch --partition=gpuA100x4 slurm/delta.sbatch ...      # wait for an A100 only
#   sbatch --time=12:00:00 slurm/delta.sbatch ...            # longer limit for one job
#SBATCH --job-name=tsasr
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source config.sh
echo "job $SLURM_JOB_ID on $(hostname), partition $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "+ $*"
"$@"
