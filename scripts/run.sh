#!/usr/bin/env bash
# Stage driver. Run a stage on a GPU node, or submit it:
#   bash scripts/run.sh prep
#   sbatch slurm/job.sbatch bash scripts/run.sh train_prefix
#   sbatch slurm/job.sbatch bash scripts/run.sh eval exp/prefix/best
#
# Setup:     prefetch (download every model once, on a login node; jobs can then run offline)
# Data:      data, prep
# Training:  train_prefix (ours) | train_none (no conditioning) | train_spk (WavLM x-vector)
#            train_spk_titanet | train_learned_sep | train_fullft | train_len_ablation | train_wavlm
#            train_aug | train_aux
# Baselines: zeroshot, dicow
# Analyses:  eval CKPT (controls + output change, lengths, quality, Libri3Mix, AMI
#            same/other session), context, explore CKPT (position, text prompt),
#            eval_perturbed, summarize
#
# TRAIN_SOURCE=ls960 (default): LibriMix-style mixtures made on the fly from LibriSpeech 960
# TRAIN_SOURCE=libri2mix:       the pre-built Libri2Mix train-100 set
set -euo pipefail
cd "$(dirname "$0")/.."
source config.sh
mkdir -p "$DATA" "$EXP" "$RESULTS" logs
L2=$STORAGE/Libri2Mix/wav16k/max
L3=$STORAGE/Libri3Mix/wav16k/max
AUG=$DATA/aug
T2=$DATA/libri2mix_test.jsonl
T3=$DATA/libri3mix_test.jsonl
AMI_T=$DATA/ami_test.jsonl
TRAIN_SOURCE=${TRAIN_SOURCE:-ls960}
if [ "$TRAIN_SOURCE" = "ls960" ]; then
  TRAIN_DATA=(--train-index "$DATA/librispeech_960_index.json")
else
  TRAIN_DATA=(--train-manifest "$DATA/libri2mix_train-100.jsonl" --speaker-pool "$DATA/speaker_pool_train-100.json")
fi
TRAIN=(python scripts/train.py "${TRAIN_DATA[@]}" --dev-manifest "$DATA/libri2mix_dev.jsonl" --model "$MODEL")
AUG_TRAIN=(--noise-list "$AUG/noise_train.txt" --rir-list "$AUG/rir_train.txt")
AUG_TEST=(--noise-list "$AUG/noise_test.txt" --rir-list "$AUG/rir_test.txt")

ev() { python scripts/evaluate.py --out-dir "$RESULTS" "$@"; }
hyps() {  # manifest ckpt mode [extra tag parts] -> path of the hyps file evaluate.py writes
  local m=$1 ck=$2 mode=$3 extra=${4:-}
  echo "$RESULTS/$(basename "$(dirname "$ck")")-$(basename "$ck")_$(basename "${m%.jsonl}")_clean_${mode}_enr5s${extra}.hyps.jsonl"
}
compare() { python scripts/compare_outputs.py "$1" "$2"; }

eval_ckpt() {  # the required analyses for one trained model
  local ck=${1:?checkpoint dir}
  # 1. enrollment controls on identical mixtures + does the output change?
  for m in target interferer other silence none; do ev --manifest "$T2" --ckpt "$ck" --enroll-mode "$m"; done
  for m in interferer other silence none; do compare "$(hyps "$T2" "$ck" target)" "$(hyps "$T2" "$ck" "$m")"; done
  # 2. enrollment length on the subset that fits a 10 s enrollment
  for d in 1 2 5 10; do ev --manifest "$T2" --ckpt "$ck" --enroll-dur "$d" --fit-dur 10; done
  # 3. enrollment quality: noise + reverberation on the enrollment only
  for snr in 20 10 5 0; do ev --manifest "$T2" --ckpt "$ck" --condition enroll_perturbed --snr "$snr" "${AUG_TEST[@]}"; done
  # 4. unseen speaker count
  ev --manifest "$T3" --ckpt "$ck"
  ev --manifest "$T3" --ckpt "$ck" --enroll-mode interferer
  # 5. real conversations: AMI, enrollment from the same and from a different session
  if [ -f "$AMI_T" ]; then
    ev --manifest "$AMI_T" --ckpt "$ck"
    for m in interferer silence none; do ev --manifest "$AMI_T" --ckpt "$ck" --enroll-mode "$m"; done
    compare "$(hyps "$AMI_T" "$ck" target)" "$(hyps "$AMI_T" "$ck" interferer)"
    # same vs different session on identical windows, both enrollments >= 5 s
    ev --manifest "$AMI_T" --ckpt "$ck" --require-key enroll_full_both
    ev --manifest "$AMI_T" --ckpt "$ck" --require-key enroll_full_both --enroll-field enrollment_other_session
  fi
}

case "${1:?stage}" in
  prefetch)
    python scripts/prefetch_models.py --model "$MODEL" ;;
  data)
    bash scripts/get_data.sh "$STORAGE"
    bash scripts/get_ami.sh "$STORAGE" test ;;
  prep)
    for s in train-100 dev test; do
      python scripts/prepare_manifests.py --librispeech-dir "$STORAGE/LibriSpeech" --librimix-dir "$L2" \
        --n-src 2 --split "$s" --out-dir "$DATA"
    done
    python scripts/prepare_manifests.py --librispeech-dir "$STORAGE/LibriSpeech" --librimix-dir "$L3" \
      --n-src 3 --split test --out-dir "$DATA"
    python scripts/prepare_aug_lists.py --wham-dir "$STORAGE/wham_noise" --rirs-dir "$STORAGE/RIRS_NOISES" \
      --out-dir "$AUG"
    if [ "$TRAIN_SOURCE" = "ls960" ]; then
      python scripts/prepare_librispeech_index.py --librispeech-dir "$STORAGE/LibriSpeech" \
        --out "$DATA/librispeech_960_index.json"
    fi
    if [ -d "$STORAGE/ami/annotations" ]; then
      python scripts/prepare_ami.py --annotations "$STORAGE/ami/annotations" --audio-dir "$STORAGE/ami/audio" \
        --split test --out "$AMI_T"
    fi ;;
  zeroshot)
    for sys in whisper_mixture whisper_prefix whisper_source; do
      ev --manifest "$T2" --system "$sys" --model "$MODEL"
      ev --manifest "$T3" --system "$sys" --model "$MODEL"
    done
    if [ -f "$AMI_T" ]; then
      for sys in whisper_mixture whisper_prefix; do ev --manifest "$AMI_T" --system "$sys" --model "$MODEL"; done
    fi ;;
  train_prefix)       "${TRAIN[@]}" --out-dir "$EXP/prefix" ;;
  train_none)         "${TRAIN[@]}" --conditioning none --out-dir "$EXP/none" ;;
  train_spk)          "${TRAIN[@]}" --conditioning spk_emb --spk-encoder wavlm --out-dir "$EXP/spk_wavlm" ;;
  train_spk_titanet)  "${TRAIN[@]}" --conditioning spk_emb --spk-encoder titanet --out-dir "$EXP/spk_titanet" ;;
  train_learned_sep)  "${TRAIN[@]}" --separator learned --out-dir "$EXP/prefix_learnedsep" ;;
  train_fullft)       "${TRAIN[@]}" --full-ft --out-dir "$EXP/prefix_fullft" ;;
  train_wavlm)        "${TRAIN[@]}" --frontend ssl --separator none --full-ft --out-dir "$EXP/wavlm" ;;
  train_len_ablation) "${TRAIN[@]}" --enroll-min 1 --enroll-max 10 --out-dir "$EXP/prefix_len1to10" ;;
  train_aug)
    for a in mix_only both_shared both_indep; do
      "${TRAIN[@]}" "${AUG_TRAIN[@]}" --aug "$a" --out-dir "$EXP/prefix_aug_$a"
    done ;;
  train_aux)
    for x in activity contrastive; do
      "${TRAIN[@]}" "${AUG_TRAIN[@]}" --aug both_indep --aux "$x" --out-dir "$EXP/prefix_aug_both_indep_aux_$x"
    done ;;
  eval)
    eval_ckpt "${2:?usage: run.sh eval exp/<run>/best}" ;;
  eval_baselines)
    # no conditioning: one pass (the enrollment is ignored by construction)
    ev --manifest "$T2" --ckpt "$EXP/none/best"; ev --manifest "$T3" --ckpt "$EXP/none/best"
    if [ -f "$AMI_T" ]; then ev --manifest "$AMI_T" --ckpt "$EXP/none/best"; fi
    eval_ckpt "$EXP/spk_wavlm/best" ;;
  dicow)
    python scripts/eval_dicow.py --manifest "$T2" --out-dir "$RESULTS"
    python scripts/eval_dicow.py --manifest "$T3" --out-dir "$RESULTS"
    if [ -f "$AMI_T" ]; then python scripts/eval_dicow.py --manifest "$AMI_T" --out-dir "$RESULTS"; fi ;;
  context)
    python scripts/context_cost.py --librispeech-dir "$STORAGE/LibriSpeech" --model "$MODEL" --out-dir "$RESULTS" ;;
  explore)
    ck=${2:?usage: run.sh explore exp/<run>/best}
    # positional embeddings: move the whole input later, or put the enrollment after the mixture
    for lead in 2 5; do ev --manifest "$T2" --ckpt "$ck" --lead-sec "$lead" --fit-dur 10; done
    ev --manifest "$T2" --ckpt "$ck" --fit-dur 10
    ev --manifest "$T2" --ckpt "$ck" --position suffix
    # enrollment combined with a text prompt
    ev --manifest "$T2" --ckpt "$ck" --text-prompt "${TEXT_PROMPT:-The following is a conversation between two people.}" ;;
  eval_perturbed)
    for ck in "$EXP"/*/best; do
      for c in clean enroll_perturbed mix_perturbed both_perturbed; do
        ev --manifest "$T2" --ckpt "$ck" --condition "$c" "${AUG_TEST[@]}"
      done
    done
    python scripts/eval_dicow.py --manifest "$T2" --out-dir "$RESULTS" --condition mix_perturbed "${AUG_TEST[@]}" ;;
  summarize)
    python scripts/summarize.py "$RESULTS" | tee "$RESULTS/table.md" ;;
  *)
    echo "unknown stage $1"; exit 1 ;;
esac
