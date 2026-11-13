# Enrollment ablations

Three models are trained on identical mixtures and tested on identical mixtures. Two
use Whisper's own front-end with the enrollment in the input window. They differ only in
how long the training enrollments are. The third reads the enrollment with WavLM
instead.

## The runs

### `random`: randomized enrollment + Whisper

![random layout](figures/padding.png)

`[enrollment 1-10 s | 0.5 s silence | mixture]`, padded to 30 s by Whisper.

The enrollment length is drawn uniformly from 1-10 s for every training example
(`ENROLL_CHOICES=1,2,5,10` draws only those values). The mixture starts right after the
separator, so its position depends on the enrollment length. The total input length is
not fixed.

### `fixed5`: fixed 5 s enrollment + Whisper

`[enrollment 5 s | 0.5 s silence | mixture]`, padded to 30 s by Whisper.

This is the setup of the project specification and the reference for the other runs.

### `wavlm`: WavLM front-end

```
[enrollment 5 s | mixture] -> WavLM -> drop the enrollment frames -> mixture frames
   -> weighted sum of WavLM layers -> linear projection -> Whisper encoder (no conv stem) -> decoder
```

WavLM reads the enrollment and the mixture as one waveform, so its self-attention can
compare them. Both models run at 50 frames per second, so the frames of the mixture are
known exactly. Only those frames are kept, and they enter Whisper's encoder in place of
its log-Mel conv stem, starting at position 0. No separator is used, because the
enrollment is removed at the frame level. By default everything is fine-tuned: WavLM
(except its waveform CNN), the projection and Whisper. `WAVLM_TRAIN=frozen` keeps WavLM
frozen when GPU memory is short.

`wavlm_random` is the same with a random 1-10 s enrollment. It is optional and not part
of `all`.

## What is held fixed

- **Training mixtures.** Every run uses the same seed and the same mixture pool, capped at 19.48 s so that a 10 s enrollment always fits. The runs therefore see identical mixtures in identical order. This holds for on-the-fly LibriSpeech 960 mixtures (default) and for Libri2Mix train-100 (`TRAIN_SOURCE=libri2mix`).
- **Training recipe.** All runs fine-tune all of Whisper by default (`WHISPER_FT=full`), so the WavLM run is compared with equally adapted Whisper runs. `WHISPER_FT=lora` switches every run to LoRA. New parameters (projections, LoRA) use `--lr` (3e-4), and pretrained weights use `--lr-pretrained` (1e-5). The dev set is scored with a 5 s enrollment for checkpoint selection.
- **Test mixtures.** The test set is the Libri2Mix test mixtures of at most 19.48 s, the subset that fits a 10 s enrollment. The enrollment for each length is cropped from the start of the same utterance.
- **Controls.** On the same mixtures, the enrollment is replaced by silence, removed (`none`) or swapped for a speaker who is not in the mixture (`other`). `compare_outputs.py` measures how much the output changes against the 5 s target enrollment.

`fixed5` here uses the 19.48 s cap and full fine-tuning, so it is not the same model as
`exp/prefix` from `scripts/run.sh`.

## How to run

From the repository root, after `bash scripts/run.sh data` and `bash scripts/run.sh prep`:

```bash
# train (one GPU job each, they can run in parallel)
sbatch slurm/job.sbatch bash scripts/run_layout_experiments.sh train random
sbatch slurm/job.sbatch bash scripts/run_layout_experiments.sh train fixed5
sbatch slurm/job.sbatch bash scripts/run_layout_experiments.sh train wavlm

# evaluate once training has finished (lengths 1/2/5/10 s + controls)
sbatch slurm/job.sbatch bash scripts/run_layout_experiments.sh eval all

# results matrix
bash scripts/run_layout_experiments.sh table          # writes results/layout_table.md
```

Options, as environment variables:

| variable | effect |
|---|---|
| `WHISPER_FT=full` (default) / `lora` | fine-tune all of Whisper, or LoRA only, in every run |
| `WAVLM_TRAIN=full` (default) / `frozen` | fine-tune WavLM or keep it frozen |
| `WAVLM_MODEL=microsoft/wavlm-base-plus` | or `microsoft/wavlm-large` (1024-d, 24 layers, more memory) |
| `ENROLL_CHOICES=1,2,5,10` | random runs draw only these lengths |
| `WITH_3MIX=1` | also evaluate on Libri3Mix |
| `EXTRA_TRAIN_ARGS="--grad-ckpt --batch-size 4 --grad-accum 4"` | if a run does not fit in GPU memory |
| `EXTRA_TRAIN_ARGS="--max-steps 200 --eval-every 100"` | short dry run before the full jobs |
| `EXTRA_EVAL_ARGS="--batch-size 32"` | passed to every evaluation |
| `TRAIN_SOURCE=libri2mix` | train on Libri2Mix train-100 instead of LibriSpeech 960 |

A stopped training job resumes when the same command is submitted again.

## Outputs

- Checkpoints: `exp/layout_{random,fixed5,wavlm}/best`. The WavLM run also stores its fine-tuned WavLM in `best/ssl/`.
- Per run: `results/layout_<run>-best_libri2mix_test_clean_<mode>_enr<L>s_fit10.{summary.json,hyps.jsonl}`
- Output change: `results/layout_<run>-best_..._<mode>_enr5s_fit10__vs__..._target_enr5s_fit10.compare.json`
- Matrix: `results/layout_table.md`, with rows = run, columns = 1/2/5/10 s and the controls, cells = TS-WER % (selection accuracy %)

## Reading the matrix

- **`random` vs `fixed5`.** At 1, 2 and 10 s this shows whether training on varied lengths helps lengths that `fixed5` never saw. At 5 s it shows what the random training costs.
- **`wavlm` vs `fixed5`.** Same 5 s training enrollments, two ways of reading them: Whisper's own encoder over the enrollment-prefixed window, or WavLM followed by Whisper on the mixture frames only.
- **Controls.** If a model uses the enrollment, its outputs under `silence`, `none` and `other` differ from the 5 s target output (low identical %, high output-change WER) and lose selection accuracy.

## Code

- `tsasr/model.py`: `frontend="ssl"` (WavLM front-end, layer weights, projection), `param_groups`
- `tsasr/data.py`: `DataConfig.frontend` (the item carries the waveform), `DataConfig.enroll_choices`
- `scripts/train.py`: `--frontend ssl --ssl-model --ssl-train`, `--separator none`, `--enroll-choices`, `--max-mix-sec`, `--lr-pretrained`
- `scripts/run_layout_experiments.sh`, `scripts/layout_table.py`
