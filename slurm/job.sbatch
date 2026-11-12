#!/bin/bash
# Generic GPU job: runs its arguments inside the project env.
#   sbatch slurm/job.sbatch python scripts/train.py --out-dir exp/prefix ...
# Adjust partition / GPU type for your cluster (examples in comments).
#SBATCH --job-name=tsasr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
##SBATCH --partition=GPU-shared          # e.g. PSC Bridges-2
##SBATCH --gpus=v100-32:1                # Bridges-2 style GPU request instead of --gres
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source config.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "+ $*"
"$@"
