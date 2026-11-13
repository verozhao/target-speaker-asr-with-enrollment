# Audio Enrollment Prompting for Target-Speaker ASR with Whisper

11-751/18-781 course project. Whisper-small is fine-tuned (LoRA or full) to transcribe
only the enrolled speaker of a multi-talker mixture. A short enrollment clip of that
speaker sits next to the mixture inside Whisper's 30 s window, so no diarization masks
and no speaker-embedding module are needed:

```
[ enrollment ~5 s | separator: 0.5 s silence or a learned token | mixture <= 24.5 s ]  ->  Whisper  ->  target speaker's words
```

## Coverage of the project specification

| Requirement | Where |
|---|---|
| Train on [~5 s enrollment + 0.5 s silence or learned token + mixture up to 25 s], target transcript as label | defaults of `train.py` (`--enroll-min/max 5`, mixtures <= 24.48 s), `--separator silence\|learned\|none` |
| LibriMix/LibriSpeechMix-style mixtures built from LibriSpeech 960 | `--train-index` (on-the-fly mixing, `--overlap full\|partial`, `--n-src-probs`), default of `run.sh` |
| Whisper-small, full or LoRA | LoRA default, `--full-ft` |
| TS-WER on 2- and 3-speaker mixtures | Libri2Mix / Libri3Mix test |
| Real multi-party conversations, enrollment from the same and from a different session | AMI (single distant mic), `--enroll-field enrollment\|enrollment_other_session` |
| Baseline: DiCoW with diarization masks | `eval_dicow.py` (oracle masks on LibriMix and AMI) |
| Baseline: speaker embedding added to the encoder (TitaNet or WavLM x-vector) | `--conditioning spk_emb --spk-encoder wavlm\|titanet` |
| Baseline: no conditioning | zero-shot `--system whisper_mixture` and fine-tuned `--conditioning none` |
| Replace the enrollment with another speaker or silence and verify the output changes | `--enroll-mode interferer\|other\|silence\|none` + `compare_outputs.py` |
| Enrollment length 1/2/5/10 s and quality | `--enroll-dur`, `--condition enroll_perturbed --snr 20\|10\|5\|0` |
| Unseen speakers / unseen speaker counts | disjoint LibriSpeech speakers; train on 2, test on 3 speakers |
| Cost of losing 5 s of context | `context_cost.py` (30 s vs 24.5 s windows) |
| Open question: positional embeddings | `--lead-sec`, `--position suffix`, `--random-lead` |
| Open question: enrollment + text prompt | `--text-prompt` |
| Architecture ablation suggested by the TA: WavLM reads [enrollment, mixture], its mixture frames replace Whisper's front-end | `--frontend ssl` (`scripts/run_layout_experiments.sh train wavlm`) |

Not covered: OWSM v3.1 (Whisper-small is used), a streaming model, and CHiME-6 or
NOTSOFAR (AMI is the real-data set).

## Layout

| path | what it does |
|---|---|
| `tsasr/audio.py` | loading (whole files or spans), cropping, input composition aligned to the 20 ms encoder frame, 50 Hz activity, noise/RIR ops, STNO masks |
| `tsasr/data.py` | `ManifestDataset` (Libri2Mix/Libri3Mix/AMI), `OnTheFlyMixDataset` (LibriSpeech 960), labels, frame labels, collator |
| `tsasr/model.py` | `TSWhisper`: LoRA or full fine-tuning; prefix, speaker-embedding or no conditioning; silence or learned separator; activity/contrastive auxiliary losses. `SpeakerEncoder`: WavLM x-vector, TitaNet, ECAPA |
| `tsasr/augment.py` | WHAM! noise + OpenSLR 28 RIR augmentation and deterministic test perturbations |
| `tsasr/ami.py` | AMI word timings, target-speaker windows, solo enrollment regions |
| `tsasr/evaluation.py`, `tsasr/text.py` | decoding, Whisper normalizer, TS-WER, selection accuracy |
| `scripts/get_data.sh`, `scripts/get_ami.sh` | downloads; Libri2Mix/Libri3Mix generation with the official LibriMix script |
| `scripts/prepare_manifests.py` | Libri2Mix/Libri3Mix target-speaker manifests with enrollment pairing |
| `scripts/prepare_librispeech_index.py` | LibriSpeech 960 utterance index for on-the-fly mixing |
| `scripts/prepare_ami.py` | AMI windows (<= 24.5 s) with same-session, other-session, interferer and absent-speaker enrollments |
| `scripts/prepare_aug_lists.py` | noise lists (WHAM! tr/cv/tt) and room-level RIR split |
| `scripts/train.py` | training (one GPU or `torchrun`, AMP, auto-resume) |
| `scripts/evaluate.py` | all systems, enrollment controls, lengths, quality, AMI, position and text-prompt tests |
| `scripts/compare_outputs.py` | output change between two enrollment conditions on the same mixtures |
| `scripts/context_cost.py` | long-form WER and window count for 30 s vs 24.5 s windows |
| `scripts/eval_dicow.py` | DiCoW v3.2 with oracle diarization masks |
| `scripts/summarize.py` | one Markdown table of every result |
| `scripts/run.sh` | stage driver with the exact command of every experiment |
| `scripts/run_layout_experiments.sh`, `scripts/layout_table.py` | enrollment ablations (random length / fixed 5 s / WavLM front-end) and their results matrix, see `docs/layout_experiments.md` |
| `slurm/*.sbatch` | SLURM wrappers (one GPU, multi-GPU) |
| `tools/make_toy_librimix.py` | macOS toy corpus for a laptop smoke test |

## Setup (GPU cluster)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # any CUDA build, torch >= 2.6
pip install -r requirements.txt
pip install -r requirements-titanet.txt     # only for the TitaNet baseline
pytest -q tests
echo 'STORAGE=/path/with/200GB/free' > config.local.sh
```

Run from the repository root. `config.sh` activates `.venv` when it exists.
On NCSA Delta, follow [docs/delta.md](docs/delta.md) (account, paths, `slurm/delta.sbatch`, storage budget).

## Run order

```bash
bash scripts/run.sh data                                       # CPU, long: LibriSpeech 960, WHAM!, RIRs, LibriMix, AMI test
bash scripts/run.sh prep                                       # manifests, LibriSpeech index, AMI windows, noise/RIR lists
sbatch slurm/job.sbatch bash scripts/run.sh zeroshot           # Whisper without training: mixture, prefix, clean source
sbatch slurm/job.sbatch bash scripts/run.sh train_prefix       # ours
sbatch slurm/job.sbatch bash scripts/run.sh train_none         # no-conditioning baseline
sbatch slurm/job.sbatch bash scripts/run.sh train_spk          # WavLM x-vector baseline (train_spk_titanet for TitaNet)
sbatch slurm/job.sbatch bash scripts/run.sh dicow              # DiCoW baseline (downloads 3.8 GB)
sbatch slurm/job.sbatch bash scripts/run.sh eval exp/prefix/best   # all required analyses for one model
sbatch slurm/job.sbatch bash scripts/run.sh eval_baselines
sbatch slurm/job.sbatch bash scripts/run.sh context            # cost of losing 5 s of context
sbatch slurm/job.sbatch bash scripts/run.sh explore exp/prefix/best   # position and text-prompt tests
bash scripts/run.sh summarize                                  # results/table.md
```

Variants: `train_learned_sep`, `train_fullft`, `train_len_ablation` (random 1-10 s
enrollment), `train_aug`, `train_aux`, `eval_perturbed`. `TRAIN_SOURCE=libri2mix`
trains on the pre-built Libri2Mix train-100 set instead of LibriSpeech 960.

Enrollment ablations (random 1-10 s enrollment, fixed 5 s enrollment, and a WavLM
front-end that reads the enrollment and passes only the mixture frames to Whisper, all on
identical mixtures) have their own driver and instructions in
[docs/layout_experiments.md](docs/layout_experiments.md).

Training resumes from `<out-dir>/last` when the same command is re-run, so a job that
hits the time limit can be resubmitted. Multi-GPU:
`sbatch --gres=gpu:4 slurm/job_multigpu.sbatch scripts/train.py <args>`. Defaults:
batch 8 per GPU, gradient accumulation 2, cosine decay, 4 epochs of 50k on-the-fly
examples, LoRA r=32 / alpha=64 on q/k/v/out, bf16 autocast where supported. New
parameters (LoRA, projections, heads) use `--lr` 3e-4, and pretrained weights that are
fine-tuned (Whisper with `--full-ft`, WavLM) use `--lr-pretrained` 1e-5. Add
`--grad-ckpt` if memory runs out.

## Systems

| system | input | conditioning | trained |
|---|---|---|---|
| `whisper_mixture` | mixture | none | no |
| `whisper_prefix` | enrollment + mixture | audio prefix | no |
| `whisper_source` | clean target source | none (topline, LibriMix only) | no |
| `exp/none` | mixture | none | yes |
| `exp/prefix` (ours) | enrollment + separator + mixture | audio prefix | yes |
| `exp/spk_wavlm`, `exp/spk_titanet` | mixture | speaker embedding added to the encoder conv stem | yes |
| `exp/wavlm` (ablation) | enrollment + mixture into WavLM, only the mixture frames into Whisper's encoder | enrollment read by WavLM | yes |
| DiCoW v3.2 | mixture | oracle silence/target/non-target/overlap masks | released model (large-v3-turbo) |

All trained systems use the same mixtures (<= 24.48 s) and schedule.

## Metrics and analyses

Text goes through the Whisper English normalizer. Every run writes `*.hyps.jsonl` and
`*.summary.json` to `results/`.

- **TS-WER**: corpus WER against the enrolled speaker's transcript, with sub/del/ins rates.
- **Selection accuracy**: share of outputs whose closest transcript belongs to the enrolled speaker.
- **Best-speaker WER**: WER against the closest transcript in the mixture.
- **Output change** (`compare_outputs.py`): identical-output rate, WER between the two outputs, and the switch rate (output follows the target, then follows the interferer when its enrollment is given). On complete 2-speaker sets the interferer run's TS-WER equals the target run's by construction. The paired comparison is the informative number.
- **Context cost**: WER and window count on whole LibriSpeech test-clean chapters decoded in 30 s and in 24.5 s windows.

Every comparison on one test set uses identical mixtures: `--fit-dur 5` by default and
`--fit-dur 10` for the length study.

## Data decisions

- **Training mixtures.** Two speakers by default, drawn from LibriSpeech 960 for every example. Each source is scaled to an RMS level between -33 and -25 dBFS, and in `full` overlap every source starts at 0 (LibriMix max mode). `partial` delays later sources at random. Mixtures never exceed 24.48 s.
- **Enrollment.** The default is 5 s from another utterance of the target speaker. A fresh utterance and a random crop offset are drawn each epoch (`--no-random-enrollment` for a fixed pairing). Enrollment and mixture are both normalized to -25 dBFS RMS before concatenation.
- **Libri2Mix/Libri3Mix test.** Each (mixture, target) gets an utterance of the target that is not a source of that mixture. Unused utterances are preferred, and `data/*.stats.json` reports how often a source of another mixture had to be used.
- **AMI.** Windows of at most 24.5 s are cut around each speaker's words in the 16 test sessions and never split one of the target's words. The audio is the distant microphone, so everyone talking in the window is heard. Enrollments are solo stretches (nobody else talking) of the same person. The other-session enrollment comes from another session of the same series, linked by global speaker id. Some people rarely talk alone for 5 s, so `enroll_full_both` marks windows where both enrollments reach 5 s. The same-versus-other-session comparison uses those windows.
- **Splits.** LibriSpeech speakers are disjoint across train, dev and test. Noise comes from WHAM! tr/cv/tt, and RIR rooms are split 80/10/10.
- **DiCoW.** It receives oracle masks, which none of our systems see, and uses a larger backbone. Its model card lists Libri2Mix and AMI among its training data, so neither test set is out of domain for it. The paper version (DiCoW_v1) needs the code of the v1 branch of the DiCoW repository and is not wired in.

## Laptop smoke test (macOS, CPU)

```bash
python tools/make_toy_librimix.py /tmp/toy
STORAGE=/tmp/toy DATA=/tmp/toydata MODEL=openai/whisper-tiny TRAIN_SOURCE=libri2mix bash scripts/run.sh prep
python scripts/evaluate.py --system whisper_mixture --model openai/whisper-tiny \
    --manifest /tmp/toydata/libri2mix_test.jsonl --out-dir /tmp/toyres --num-workers 0
```

## Known gaps

- TitaNet and ECAPA need extra packages (`requirements-titanet.txt`, `speechbrain`) and were not run on the development laptop. WavLM is the tested speaker encoder.
- `eval_dicow.py` was not run on the development laptop, where loading DiCoW's remote code was not permitted. Its mask construction is unit-tested.
- Reverberation is applied to the summed mixture with one RIR.
