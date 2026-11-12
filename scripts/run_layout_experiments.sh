#!/usr/bin/env bash
# Enrollment-layout experiments (docs/layout_experiments.md). All three train on the
# same mixtures (same seed, mixtures <= 19.48 s) and are tested on the same mixtures:
#
#   padding  random 1-10 s enrollment | 0.5 s separator | mixture | padding to 30 s
#            (the mixture starts later when the enrollment is longer)
#   anchor   random 1-10 s enrollment | silence up to 10.5 s | mixture
#            (the mixture always starts at 10.5 s, only the silence changes)
#   fixed5   fixed 5 s enrollment | 0.5 s separator | mixture | padding to 30 s
#
# Usage (from the repository root):
#   bash scripts/run_layout_experiments.sh train padding|anchor|fixed5|all
#   bash scripts/run_layout_experiments.sh eval  padding|anchor|fixed5|all
#   bash scripts/run_layout_experiments.sh table
#   sbatch slurm/job.sbatch bash scripts/run_layout_experiments.sh train anchor
#
# Options (environment variables):
#   ENROLL_CHOICES=1,2,5,10   train the random layouts on these lengths only (default: uniform 1-10 s)
#   MIX_START=10.5            anchor position in seconds (longest enrollment + 0.5 s separator)
#   WITH_3MIX=1               also evaluate on Libri3Mix
#   EXTRA_TRAIN_ARGS="..."    passed to every train.py call (e.g. "--max-steps 200" for a dry run)
#   EXTRA_EVAL_ARGS="..."     passed to every evaluate.py call (e.g. "--batch-size 32")
#   TRAIN_SOURCE=ls960|libri2mix  as in scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source config.sh
mkdir -p "$EXP" "$RESULTS" logs
MIX_START=${MIX_START:-10.5}
MAX_MIX=19.48                       # 30 s - 10.5 s - one 20 ms frame
LENGTHS="1 2 5 10"
T2=$DATA/libri2mix_test.jsonl
T3=$DATA/libri3mix_test.jsonl
TRAIN_SOURCE=${TRAIN_SOURCE:-ls960}
if [ "$TRAIN_SOURCE" = "ls960" ]; then
  TRAIN_DATA=(--train-index "$DATA/librispeech_960_index.json")
else
  TRAIN_DATA=(--train-manifest "$DATA/libri2mix_train-100.jsonl" --speaker-pool "$DATA/speaker_pool_train-100.json")
fi
# shellcheck disable=SC2206
EXTRA=(${EXTRA_TRAIN_ARGS:-})
# shellcheck disable=SC2206
EVX=(${EXTRA_EVAL_ARGS:-})
CHOICES=()
if [ -n "${ENROLL_CHOICES:-}" ]; then CHOICES=(--enroll-choices "$ENROLL_CHOICES"); fi

train_one() {
  local name=$1 out="$EXP/layout_$1"
  local common=(python scripts/train.py "${TRAIN_DATA[@]}" --dev-manifest "$DATA/libri2mix_dev.jsonl"
                --model "$MODEL" --seed 0 --gap 0.5 --dev-enroll-dur 5 --out-dir "$out")
  case "$name" in
    padding) "${common[@]}" --enroll-min 1 --enroll-max 10 ${CHOICES[@]+"${CHOICES[@]}"} --max-mix-sec "$MAX_MIX" ${EXTRA[@]+"${EXTRA[@]}"} ;;
    anchor)  "${common[@]}" --enroll-min 1 --enroll-max 10 ${CHOICES[@]+"${CHOICES[@]}"} --mix-start "$MIX_START" \
               --max-mix-sec "$MAX_MIX" ${EXTRA[@]+"${EXTRA[@]}"} ;;
    fixed5)  "${common[@]}" --enroll-min 5 --enroll-max 5 --max-mix-sec "$MAX_MIX" ${EXTRA[@]+"${EXTRA[@]}"} ;;
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
  local fn=$1 which=${2:?padding|anchor|fixed5|all}
  if [ "$which" = "all" ]; then
    for n in padding anchor fixed5; do "$fn" "$n"; done
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
