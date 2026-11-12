# Audio Enrollment Prompting for Target-Speaker ASR with Whisper

11-751/18-781 course project. Whisper-small is fine-tuned with LoRA to transcribe only
the enrolled speaker in a multi-talker mixture. The enrollment audio is placed before
the mixture inside Whisper's 30 s window, and no external speaker encoder or
diarization model is used:

```
[ enrollment (1-10 s) | 0.5 s silence | mixture ]  ->  Whisper encoder (LoRA)  ->  decoder  ->  target words only
```

## Layout

| path | what it does |
|---|---|
| `tsasr/audio.py` | loading, cropping, prefix composition aligned to the 20 ms encoder frame, 50 Hz speaker activity, noise/RIR ops, STNO masks |
| `tsasr/augment.py` | WHAM! noise + OpenSLR 28 RIR augmentation (`none`, `mix_only`, `both_shared`, `both_indep`) and deterministic test perturbations |
| `tsasr/data.py` | manifest dataset, enrollment sampling, decoder prompt/labels, frame labels for the auxiliary loss, collator |
| `tsasr/model.py` | `TSWhisper` (LoRA, prefix or speaker-embedding conditioning, `activity` / `contrastive` auxiliary losses), frozen `SpeakerEncoder` (WavLM x-vector or ECAPA) |
| `tsasr/evaluation.py`, `tsasr/text.py` | batched decoding, Whisper normalizer, TS-WER, selection accuracy |
| `scripts/get_data.sh` | downloads LibriSpeech, WHAM!, RIRS_NOISES and generates Libri2Mix/Libri3Mix (16 kHz, max) with the official LibriMix script |
| `scripts/prepare_manifests.py` | target-speaker manifests with enrollment pairing, speaker pools, split checks |
| `scripts/prepare_aug_lists.py` | train/dev/test noise lists (WHAM! tr/cv/tt) and room-level RIR split |
| `scripts/train.py` | training (single GPU or `torchrun` DDP, AMP, auto-resume) |
| `scripts/evaluate.py` | zero-shot baselines and trained models, all enrollment controls and conditions |
| `scripts/eval_dicow.py` | DiCoW v3.2 baseline with oracle diarization |
| `scripts/summarize.py` | collects every `*.summary.json` into one table |
| `scripts/run.sh` | stage driver with the exact commands for every experiment |
| `slurm/*.sbatch` | generic SLURM wrappers (1 GPU, multi-GPU) |
| `tools/make_toy_librimix.py` | macOS-only toy corpus (TTS voices as speakers) for a laptop smoke test |

## Setup (GPU cluster)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # any CUDA build of torch >= 2.6
pip install -r requirements.txt
pytest -q tests                                     # 7 tests
echo 'STORAGE=/path/with/120GB/free' > config.local.sh   # see config.sh for the other paths
```

Run everything from the repository root. `config.sh` activates `.venv` if it exists.

## Midpoint run order

```bash
bash scripts/run.sh data                                   # CPU, several hours, ~120 GB
bash scripts/run.sh prep                                   # CPU, minutes
sbatch slurm/job.sbatch bash scripts/run.sh zeroshot       # 3 zero-shot systems on Libri2Mix/Libri3Mix test
sbatch slurm/job.sbatch bash scripts/run.sh train_prefix   # our model, no augmentation, no auxiliary loss
sbatch slurm/job.sbatch bash scripts/run.sh train_spk      # speaker-embedding baseline
sbatch slurm/job.sbatch bash scripts/run.sh dicow          # DiCoW baseline (downloads 3.8 GB)
# after training finishes
sbatch slurm/job.sbatch bash scripts/run.sh eval_prefix
sbatch slurm/job.sbatch bash scripts/run.sh eval_spk
bash scripts/run.sh summarize                              # results/table.md
```

For the final report: `train_aug`, `train_aux`, `eval_perturbed` (same driver).

Training resumes automatically from `<out-dir>/last` when the same command is
re-run, so a job that hits the time limit can simply be resubmitted. Multi-GPU:
`sbatch --gres=gpu:4 slurm/job_multigpu.sbatch scripts/train.py <args>`.
Defaults: batch 8 per GPU, gradient accumulation 2, lr 3e-4 with cosine decay,
4 epochs, LoRA r=32 / alpha=64 on q/k/v/out projections, bf16 autocast where
supported (fp16 + loss scaling on V100). Add `--grad-ckpt` if memory runs out.
`exp/<run>/log.jsonl` logs loss, dev TS-WER and seconds per step.

## Systems

| system | input | conditioning | trained |
|---|---|---|---|
| `whisper_mixture` | mixture | none | no |
| `whisper_prefix` | enrollment + mixture | audio prefix | no |
| `whisper_source` | clean target source | none (topline) | no |
| `exp/prefix` (ours) | enrollment + mixture | audio prefix | LoRA |
| `exp/spk_wavlm` | mixture | WavLM x-vector added to the encoder stem | LoRA + projection |
| DiCoW v3.2 | mixture | oracle STNO diarization masks | released model (large-v3-turbo) |

## Metrics

All text goes through the Whisper English normalizer. Every run writes per-utterance
hypotheses (`*.hyps.jsonl`) and a summary (`*.summary.json`) to `results/`.

- **TS-WER**: corpus WER against the enrolled speaker's transcript, with substitution, deletion and insertion rates. Insertions mostly count words of the interfering speaker.
- **Selection accuracy**: share of outputs whose closest transcript (by WER) belongs to the enrolled speaker.
- **Best-speaker WER**: WER against whichever source transcript is closest. Speaker-agnostic recognition quality, mainly for the unconditioned baseline.

Enrollment controls on identical mixtures (`--enroll-mode`): `target`, `other` (a
speaker absent from the mixture), `silence` (digital silence of the same duration), and
`none` (no prefix). Durations: `--enroll-dur 1 2 5 10`. Every system is scored on the
same subset, the mixtures that fit in 30 s with a 10 s enrollment (`--fit-dur 10`).

## Data decisions

- **Enrollment pairing.** Each (mixture, target) gets an utterance of the target speaker that is not a source of that mixture. Utterances never used as a source in the split are preferred. When none exist, the enrollment comes from another mixture's source. `data/*.stats.json` reports how many enrollments follow each policy. Utterances of at least 11 s are preferred so one enrollment can be cropped to every duration.
- **Training enrollment.** A fresh utterance of the target speaker is drawn every epoch, with a random 1-10 s crop that still fits the 30 s window.
- **Levels.** Enrollment and mixture are both normalized to -25 dBFS RMS before concatenation, so the two separately captured clips share one level inside the 30 s window.
- **Targets.** LibriSpeech transcripts are lower-cased. The `<|en|><|transcribe|><|notimestamps|>` prompt is given and excluded from the loss.
- **Splits.** LibriSpeech speakers are disjoint across train-clean-100, dev-clean and test-clean, and `prepare_manifests.py` checks this. Noise comes from WHAM! tr/cv/tt. Simulated RIR rooms are split 80/10/10 by a fixed hash of the room id.
- **Reverberation.** One RIR is applied to the summed mixture, not one per source.
- **DiCoW.** It receives oracle diarization from the source signals, which our systems never see. It uses its own larger base model (large-v3-turbo).

## Laptop smoke test (no GPU, macOS)

```bash
python tools/make_toy_librimix.py /tmp/toy
STORAGE=/tmp/toy DATA=/tmp/toydata MODEL=openai/whisper-tiny bash scripts/run.sh prep
python scripts/evaluate.py --system whisper_mixture --model openai/whisper-tiny \
    --manifest /tmp/toydata/libri2mix_test.jsonl --out-dir /tmp/toyres --num-workers 0
```

`scripts/get_data.sh` is skipped here. The toy corpus already has the LibriMix layout.
The noise and RIR lists will be tiny.

## Known gaps (planned for the final report)

- AMI evaluation is not implemented yet.
- The ECAPA speaker encoder (`--spk-encoder ecapa`) needs `pip install speechbrain` and has not been run yet. WavLM is the tested default.
