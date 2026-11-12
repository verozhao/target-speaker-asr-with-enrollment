# Paths and environment shared by every step. Copy to config.local.sh to
# override (git-ignored), e.g. on the cluster:
#   STORAGE=/ocean/projects/<proj>/<user>/tsasr_data
STORAGE=${STORAGE:-$PWD/storage}       # raw corpora + generated LibriMix (~120 GB)
DATA=${DATA:-$PWD/data}                # manifests, speaker pools, noise/RIR lists
EXP=${EXP:-$PWD/exp}                   # checkpoints
RESULTS=${RESULTS:-$PWD/results}       # hypotheses + summaries
MODEL=${MODEL:-openai/whisper-small}
VENV=${VENV:-$PWD/.venv}
if [ -f "$PWD/config.local.sh" ]; then source "$PWD/config.local.sh"; fi
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
export STORAGE DATA EXP RESULTS MODEL
