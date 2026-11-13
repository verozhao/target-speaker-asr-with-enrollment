# Paths for NCSA Delta. Copy to config.local.sh (git-ignored) and adjust:
#   cp config.delta.example.sh config.local.sh
# Project space of the course allocation: /work/hdd/<project>/$USER (large, ~100 GB per
# student recommended) and /work/nvme/<project>/$USER (fast, used for Python envs).
PROJECT=${PROJECT:-bjdh}
STORAGE=/work/hdd/$PROJECT/$USER/tsasr_data          # corpora + generated LibriMix
DATA=/work/hdd/$PROJECT/$USER/tsasr_data/manifests   # manifests, speaker pools, noise/RIR lists
EXP=/work/hdd/$PROJECT/$USER/tsasr_exp               # checkpoints
RESULTS=$PWD/results                                 # small json files, kept with the code
VENV=/work/nvme/$PROJECT/$USER/tsasr_venv
export HF_HOME=/work/hdd/$PROJECT/$USER/hf_cache     # model downloads, keep them out of $HOME
