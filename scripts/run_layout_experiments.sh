#!/usr/bin/env bash
# Enrollment ablations (docs/layout_experiments.md). All runs train on the same mixtures
# (same seed, mixtures <= 19.48 s) and are tested on the same mixtures at 1/2/5/10 s:
#
#   random        random 1-10 s enrollment | 0.5 s separator | mixture   -> Whisper
#   fixed5        fixed 5 s enrollment     | 0.5 s separator | mixture   -> Whisper
#   wavlm         fixed 5 s enrollment | mixture -> WavLM; the enrollment frames are dropped
#                 and the mixture frames go into Whisper's encoder in place of its conv stem
#   wavlm_random  as wavlm with a random 1-10 s enrollment (optional, not part of "all")
#
# Usage (from the repository root):
#   bash scripts/run_layout_experiments.sh train random|fixed5|wavlm|wavlm_random|all
#   bash scripts/run_layout_experiments.sh eval  random|fixed5|wavlm|wavlm_random|all
#   bash scripts/run_layout_experiments.sh table
#   sbatch slurm/job.sbatch bash scripts/run_layout_experiments.sh train wavlm
#
# Options (environment variables):
#   WHISPER_FT=full|lora      full: fine-tune all of Whisper in every run (default); lora: LoRA only
#   WAVLM_TRAIN=full|frozen   fine-tune WavLM (default) or keep it frozen (less GPU memory)
#   WAVLM_MODEL=microsoft/wavlm-base-plus   or microsoft/wavlm-large
#   ENROLL_CHOICES=1,2,5,10   random runs draw only these lengths (default: uniform 1-10 s)
#   WITH_3MIX=1               also evaluate on Libri3Mix
#   EXTRA_TRAIN_ARGS="..."    passed to every train.py call (e.g. "--max-steps 200 --grad-ckpt")
#   EXTRA_EVAL_ARGS="..."     passed to every evaluate.py call (e.g. "--batch-size 32")
#   TRAIN_SOURCE=ls960|libri2mix  as in scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source config.sh
mkdir -p "$EXP" "$RESULTS" logs
MAX_MIX=19.48                       # mixtures that fit next to a 10 s enrollment + 0.5 s separator
LENGTHS="1 2 5 10"
T2=$DATA/libri2mix_test.jsonl
T3=$DATA/libri3mix_test.jsonl
TRAIN_SOURCE=${TRAIN_SOURCE:-ls960}
if [ "$TRAIN_SOURCE" = "ls960" ]; then
  TRAIN_DATA=(--train-index "$DATA/librispeech_960_index.json")
else
  TRAIN_DATA=(--train-manifest "$DATA/libri2mix_train-100.jsonl" --speaker-pool "$DATA/speaker_pool_train-100.json")
fi
FT=()
if [ "${WHISPER_FT:-full}" = "full" ]; then FT=(--full-ft); fi
# shellcheck disable=SC2206
EXTRA=(${EXTRA_TRAIN_ARGS:-})
# shellcheck disable=SC2206
EVX=(${EXTRA_EVAL_ARGS:-})
RANDOM_LEN=(--enroll-min 1 --enroll-max 10)
if [ -n "${ENROLL_CHOICES:-}" ]; then RANDOM_LEN=(--enroll-choices "$ENROLL_CHOICES"); fi
WAVLM=(--frontend ssl --separator none --ssl-model "${WAVLM_MODEL:-microsoft/wavlm-base-plus}"
       --ssl-train "${WAVLM_TRAIN:-full}")

train_one() {
  local name=$1
  local common=(python scripts/train.py "${TRAIN_DATA[@]}" --dev-manifest "$DATA/libri2mix_dev.jsonl"
                --model "$MODEL" --seed 0 --dev-enroll-dur 5 --max-mix-sec "$MAX_MIX"
                --out-dir "$EXP/layout_$name" ${FT[@]+"${FT[@]}"})
  case "$name" in
    random)       "${common[@]}" "${RANDOM_LEN[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
    fixed5)       "${common[@]}" --enroll-min 5 --enroll-max 5 ${EXTRA[@]+"${EXTRA[@]}"} ;;
    wavlm)        "${common[@]}" --enroll-min 5 --enroll-max 5 "${WAVLM[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
    wavlm_random) "${common[@]}" "${RANDOM_LEN[@]}" "${WAVLM[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
    *) echo "unknown experiment $name"; exit 1 ;;
  esac
}

eval_one() {
  local name=$1 ck="$EXP/layout_$1/best"
  local sets=("$T2")
  if [ "${WITH_3MIX:-0}" = "1" ]; then sets+=("$T3"); fi
  for m in "${sets[@]}"; do
    # same mixtures for every length and experiment: those that fit a 10 s enrollment
    for d in $LENGTHS; do
      python scripts/evaluate.py --out-dir "$RESULTS" --manifest "$m" --ckpt "$ck" --enroll-dur "$d" --fit-dur 10 \
        ${EVX[@]+"${EVX[@]}"}
    done
    # does the output depend on the enrollment? same mixtures, silence / no enrollment / other speaker
    for mode in silence none other; do
      python scripts/evaluate.py --out-dir "$RESULTS" --manifest "$m" --ckpt "$ck" --enroll-mode "$mode" --fit-dur 10 \
        ${EVX[@]+"${EVX[@]}"}
    done
    local base="$RESULTS/layout_${name}-best_$(basename "${m%.jsonl}")_clean"
    for mode in silence none other; do
      python scripts/compare_outputs.py "${base}_target_enr5s_fit10.hyps.jsonl" "${base}_${mode}_enr5s_fit10.hyps.jsonl"
    done
  done
}

each() {  # stage, name|all
  local fn=$1 which=${2:?random|fixed5|wavlm|wavlm_random|all}
  if [ "$which" = "all" ]; then
    for n in random fixed5 wavlm; do "$fn" "$n"; done
  else
    "$fn" "$which"
  fi
}

case "${1:?train|eval|table}" in
  train) each train_one "${2:-}" ;;
  eval)  each eval_one "${2:-}" ;;
  table) python scripts/layout_table.py "$RESULTS" | tee "$RESULTS/layout_table.md" ;;
  *) echo "usage: $0 train|eval NAME|all, or $0 table"; exit 1 ;;
esac
