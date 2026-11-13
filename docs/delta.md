# Running on NCSA Delta

Delta is the course GPU cluster. Jobs go through SLURM: you submit from a login node,
and they run on GPU nodes. The A40 has 48 GB and the A100 has 40 GB. Login nodes are
only for editing, submitting and light setup. Never train or generate data on them.

## 1. Log in

On your laptop:

```bash
ssh <ncsa_username>@login.delta.ncsa.illinois.edu    # NCSA password, then Duo (type 1, approve the push)
```

Use your NCSA username and password. The ACCESS password does not work here.

## 2. Code and account (once)

```bash
cd /work/hdd/bjdh/$USER
git clone <URL of this repository> target-speaker-asr-with-enrollment
cd target-speaker-asr-with-enrollment

# the course allocation: copy the account from the homework job script
grep "#SBATCH" /work/hdd/bjdh/shared/hw7/run.sbatch
echo 'export SBATCH_ACCOUNT=<value of --account / -A from that output>' >> ~/.bashrc
source ~/.bashrc
```

`sbatch` reads `SBATCH_ACCOUNT`, so the job files in this repo carry no account.

## 3. Python environment (once, on the login node)

```bash
python3 --version          # 3.9 or newer is enough
python3 -m venv /work/nvme/bjdh/$USER/tsasr_venv
source /work/nvme/bjdh/$USER/tsasr_venv/bin/activate
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cp config.delta.example.sh config.local.sh     # paths below /work/hdd/bjdh/$USER and /work/nvme/bjdh/$USER
```

`config.sh` activates this environment in every job.

To check the GPU and run the tests, use a short interactive job instead of the login node:

```bash
srun --partition=gpuA40x4-interactive --gpus-per-node=1 --cpus-per-task=4 --mem=16g --time=00:30:00 --pty bash
source config.sh && python -c "import torch; print(torch.cuda.get_device_name())" && pytest -q tests
exit
```

The interactive partitions allow one queued job per user. `sinfo -s | grep interactive`
lists their exact names if `gpuA40x4-interactive` is not accepted.

## 4. Storage budget

The course project space is about 6 TB for 59 students, about 100 GB each. The full
data plan is larger than that:

| item | approx. size |
|---|---|
| LibriSpeech 960 h (`WITH_960=1`) | 60 GB (train-other-500 alone is 30 GB) |
| WHAM! noise | 20 GB |
| Libri2Mix dev/test + Libri3Mix test (with the unused `mix_both`) | 21 GB |
| Libri2Mix train-100 (only for `TRAIN_SOURCE=libri2mix`) | 33 GB, skip with `WITH_L2M_TRAIN=0` |
| RIRs, AMI test audio | 3 GB |
| model downloads (`HF_HOME`) | 7 GB |
| checkpoints (full fine-tuning about 1 GB each, best + last per run) | 15 GB |

Before downloading:

- Ask the TA whether LibriSpeech, LibriMix or WHAM! already exist somewhere on Delta (check `ls /work/hdd/bjdh/shared` too).
- Otherwise split the data across the three of you. For example, one person keeps the corpora and another the checkpoints. Make the folders readable by the group with `chmod -R g+rX <folder>`, and point each person's `config.local.sh` at them.
- `WITH_L2M_TRAIN=0` saves 33 GB when training on LibriSpeech 960 mixtures (the default).

## 5. Data

Downloading and generating mixtures takes hours of CPU time, so run it as a job and not
on the login node. Ask the TA whether a CPU partition is allowed for the course
allocation. If not, use the GPU partition with a long time limit:

```bash
WITH_L2M_TRAIN=0 sbatch --time=24:00:00 slurm/delta.sbatch bash scripts/run.sh data
sbatch --time=04:00:00 slurm/delta.sbatch bash scripts/run.sh prep
```

## 6. Experiments

Submit from the repository root. Each command is one job, and the jobs queue independently.

```bash
sbatch slurm/delta.sbatch bash scripts/run.sh zeroshot
sbatch slurm/delta.sbatch bash scripts/run.sh train_prefix
sbatch slurm/delta.sbatch bash scripts/run.sh train_none
sbatch slurm/delta.sbatch bash scripts/run.sh train_spk
sbatch slurm/delta.sbatch bash scripts/run_layout_experiments.sh train random   # also: fixed5, wavlm
```

- **Time limit.** Jobs stop after `--time` (8 hours by default in `slurm/delta.sbatch`). Training saves `last/` regularly and resumes when the same command is submitted again, so resubmit until `log.jsonl` shows `"event": "finished"`.
- **GPU type.** Jobs run on whichever of A40 or A100 frees up first. `sbatch --partition=gpuA100x4 slurm/delta.sbatch ...` waits for an A100. Do not use other partitions such as H200: the course does not allow them.
- **Memory.** If a job runs out of GPU memory (`CUDA out of memory` in the log), add `--grad-ckpt` or a smaller `--batch-size` with a larger `--grad-accum`. With the layout driver, use `EXTRA_TRAIN_ARGS`.

## 7. Watching jobs

```bash
squeue -u $USER                          # PD = waiting, R = running, gone = finished
squeue -u $USER -o "%i %S"               # estimated start time
tail -f logs/tsasr-<jobid>.out           # live output (Ctrl+C stops watching, not the job)
tail -f /work/hdd/bjdh/$USER/tsasr_exp/<run>/log.jsonl
sacct -j <jobid> --format=JobID,State,Elapsed     # COMPLETED / FAILED / TIMEOUT
scancel <jobid>                          # cancel
```

## 8. Results to your laptop

Run this on your laptop, not on Delta:

```bash
scp -r <ncsa_username>@login.delta.ncsa.illinois.edu:/work/hdd/bjdh/<ncsa_username>/target-speaker-asr-with-enrollment/results ./results_delta
```
