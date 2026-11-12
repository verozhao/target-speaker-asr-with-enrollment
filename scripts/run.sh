#!/usr/bin/env bash
# Stage driver. Run a stage directly on a GPU node, or submit it:
#   bash scripts/run.sh prep
#   sbatch slurm/job.sbatch bash scripts/run.sh train_prefix
#
# Midpoint stages, in order:
#   data          download corpora + generate Libri2Mix/Libri3Mix   (CPU, long)
#   prep          manifests, speaker pools, noise/RIR lists         (CPU)
#   zeroshot      unconditioned Whisper, zero-shot prefix, clean-source topline
#   train_prefix  LoRA Whisper with the audio enrollment prefix (no aug, no aux)
#   eval_prefix   controls (target/other/silence/none), durations, Libri3Mix
#   train_spk     speaker-embedding conditioning baseline (WavLM x-vector)
#   eval_spk
#   dicow         DiCoW v3.2 with oracle diarization
#   summarize     results/table.md
# Final-report stages (same code, more runs):
#   train_aug     mix_only, both_shared, both_indep augmentation
#   train_aux     activity and contrastive auxiliary losses (on top of both_indep)
#   eval_perturbed  every trained model under the four acoustic conditions
set -euo pipefail
cd "$(dirname "$0")/.."
source config.sh
mkdir -p "$DATA" "$EXP" "$RESULTS" logs
L2=$STORAGE/Libri2Mix/wav16k/max
L3=$STORAGE/Libri3Mix/wav16k/max
AUG=$DATA/aug
TRAIN_COMMON=(--train-manifest "$DATA/libri2mix_train-100.jsonl" --dev-manifest "$DATA/libri2mix_dev.jsonl"
              --speaker-pool "$DATA/speaker_pool_train-100.json" --model "$MODEL")
AUG_TRAIN=(--noise-list "$AUG/noise_train.txt" --rir-list "$AUG/rir_train.txt")
AUG_TEST=(--noise-list "$AUG/noise_test.txt" --rir-list "$AUG/rir_test.txt")
T2=$DATA/libri2mix_test.jsonl
T3=$DATA/libri3mix_test.jsonl

ev() { python scripts/evaluate.py --out-dir "$RESULTS" "$@"; }

eval_model() {  # ckpt  -> standard battery on clean Libri2Mix test + Libri3Mix
  local ck=$1
  for m in target other silence none; do ev --manifest "$T2" --ckpt "$ck" --enroll-mode "$m"; done
  for d in 1 2 10; do ev --manifest "$T2" --ckpt "$ck" --enroll-dur "$d"; done   # 5 s ran above
  ev --manifest "$T3" --ckpt "$ck"
}

case "${1:?stage}" in
  data)
    bash scripts/get_data.sh "$STORAGE" ;;
  prep)
    for s in train-100 dev test; do
      python scripts/prepare_manifests.py --librispeech-dir "$STORAGE/LibriSpeech" --librimix-dir "$L2" \
        --n-src 2 --split "$s" --out-dir "$DATA"
    done
    python scripts/prepare_manifests.py --librispeech-dir "$STORAGE/LibriSpeech" --librimix-dir "$L3" \
      --n-src 3 --split test --out-dir "$DATA"
    python scripts/prepare_aug_lists.py --wham-dir "$STORAGE/wham_noise" --rirs-dir "$STORAGE/RIRS_NOISES" \
      --out-dir "$AUG" ;;
  zeroshot)
    for sys in whisper_mixture whisper_prefix whisper_source; do
      ev --manifest "$T2" --system "$sys" --model "$MODEL"
      ev --manifest "$T3" --system "$sys" --model "$MODEL"
    done ;;
  train_prefix)
    python scripts/train.py "${TRAIN_COMMON[@]}" --out-dir "$EXP/prefix" ;;
  eval_prefix)
    eval_model "$EXP/prefix/best" ;;
  train_spk)
    python scripts/train.py "${TRAIN_COMMON[@]}" --conditioning spk_emb --spk-encoder wavlm --out-dir "$EXP/spk_wavlm" ;;
  eval_spk)
    for m in target other silence; do ev --manifest "$T2" --ckpt "$EXP/spk_wavlm/best" --enroll-mode "$m"; done
    for d in 1 2 10; do ev --manifest "$T2" --ckpt "$EXP/spk_wavlm/best" --enroll-dur "$d"; done
    ev --manifest "$T3" --ckpt "$EXP/spk_wavlm/best" ;;
  dicow)
    python scripts/eval_dicow.py --manifest "$T2" --out-dir "$RESULTS"
    python scripts/eval_dicow.py --manifest "$T3" --out-dir "$RESULTS" ;;
  train_aug)
    for a in mix_only both_shared both_indep; do
      python scripts/train.py "${TRAIN_COMMON[@]}" "${AUG_TRAIN[@]}" --aug "$a" --out-dir "$EXP/prefix_aug_$a"
    done ;;
  train_aux)
    for x in activity contrastive; do
      python scripts/train.py "${TRAIN_COMMON[@]}" "${AUG_TRAIN[@]}" --aug both_indep --aux "$x" \
        --out-dir "$EXP/prefix_aug_both_indep_aux_$x"
    done ;;
  eval_perturbed)
    for ck in "$EXP"/*/best; do
      for c in clean enroll_perturbed mix_perturbed both_perturbed; do
        ev --manifest "$T2" --ckpt "$ck" --condition "$c" "${AUG_TEST[@]}"
      done
    done
    for c in mix_perturbed; do python scripts/eval_dicow.py --manifest "$T2" --out-dir "$RESULTS" --condition "$c" "${AUG_TEST[@]}"; done ;;
  summarize)
    python scripts/summarize.py "$RESULTS" | tee "$RESULTS/table.md" ;;
  *)
    echo "unknown stage $1"; exit 1 ;;
esac
