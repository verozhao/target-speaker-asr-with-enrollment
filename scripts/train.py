#!/usr/bin/env python
"""Fine-tune Whisper (LoRA or full) for enrollment-conditioned target-speaker ASR.

Training data, one of:
  --train-manifest data/libri2mix_train-100.jsonl          pre-built Libri2Mix
  --train-index data/librispeech_960_index.json            LibriMix-style mixtures made on the fly
Single GPU:   python scripts/train.py --train-index ... --dev-manifest ... --out-dir exp/prefix
Multi GPU:    torchrun --nproc_per_node 4 scripts/train.py ...
Re-running the same command resumes from <out-dir>/last automatically.
"""
import argparse
import datetime
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from transformers import WhisperFeatureExtractor, WhisperTokenizer, get_cosine_schedule_with_warmup

from tsasr.augment import TRAIN_MODES, Augmenter
from tsasr.data import (Collator, DataConfig, ManifestDataset, OnTheFlyMixDataset, Tokens, fits,
                        max_mixture_sec, read_manifest)
from tsasr.evaluation import autocast, decode, score, subset
from tsasr.librispeech import read_index
from tsasr.model import SpeakerEncoder, TSWhisper


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--train-manifest", help="pre-built mixtures (e.g. Libri2Mix train-100)")
    src.add_argument("--train-index", nargs="+", help="LibriSpeech index json(s): mix on the fly")
    p.add_argument("--n-src-probs", default="2:1.0", help="on-the-fly speaker counts, e.g. 2:0.7,3:0.3")
    p.add_argument("--overlap", default="full", choices=["full", "partial"], help="on-the-fly overlap")
    p.add_argument("--samples-per-epoch", type=int, default=50000, help="on-the-fly examples per epoch")
    p.add_argument("--dev-manifest", required=True)
    p.add_argument("--speaker-pool", default=None, help="train speaker pool json (random enrollment per epoch)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--conditioning", default="prefix", choices=["prefix", "spk_emb", "none"],
                   help="prefix = ours, spk_emb = embedding baseline, none = no-conditioning baseline")
    p.add_argument("--spk-encoder", default="wavlm", choices=["wavlm", "titanet", "ecapa"])
    p.add_argument("--separator", default="silence", choices=["silence", "learned"])
    p.add_argument("--position", default="prefix", choices=["prefix", "suffix"], help="enrollment before/after the mixture")
    p.add_argument("--random-lead", action="store_true", help="random start position inside the 30 s window")
    p.add_argument("--full-ft", action="store_true", help="full fine-tuning instead of LoRA (use a lower --lr)")
    p.add_argument("--aux", default="none", choices=["none", "activity", "contrastive"])
    p.add_argument("--aux-weight", type=float, default=0.1)
    # augmentation
    p.add_argument("--aug", default="none", choices=TRAIN_MODES)
    p.add_argument("--noise-list", default=None)
    p.add_argument("--rir-list", default=None)
    p.add_argument("--p-noise", type=float, default=0.5)
    p.add_argument("--p-rir", type=float, default=0.5)
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    # enrollment
    p.add_argument("--enroll-min", type=float, default=5.0, help="5/5 = fixed 5 s (spec); 1/10 = random-length ablation")
    p.add_argument("--enroll-max", type=float, default=5.0)
    p.add_argument("--gap", type=float, default=None, help="separator seconds (default 0.5 silence, 0.02 learned)")
    p.add_argument("--no-random-enrollment", action="store_true")
    # LoRA
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,out_proj")
    # optimisation
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=-1, help="optimizer steps, overrides --epochs if > 0")
    p.add_argument("--batch-size", type=int, default=8, help="per GPU")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--num-workers", type=int, default=8)
    # eval / logging
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--dev-subset", type=int, default=500)
    p.add_argument("--dev-enroll-dur", type=float, default=None, help="default: --enroll-max")
    p.add_argument("--dev-batch-size", type=int, default=16)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-examples", type=int, default=None, help="debug: cap training set")
    a = p.parse_args()
    if a.gap is None:
        a.gap = 0.02 if a.separator == "learned" else 0.5
    if a.dev_enroll_dur is None:
        a.dev_enroll_dur = a.enroll_max
    return a


class EpochSampler(Sampler):
    """Seeded per-epoch permutation, sharded over ranks, with a resumable start offset."""

    def __init__(self, n, seed, rank, world):
        self.n, self.seed, self.rank, self.world = n, seed, rank, world
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch, skip=0):
        self.epoch, self.skip = epoch, skip

    def _indices(self):
        perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
        per = math.ceil(self.n / self.world)
        perm = np.concatenate([perm, perm[: per * self.world - self.n]])
        return perm[self.rank:: self.world]

    def __iter__(self):
        return iter(self._indices()[self.skip:].tolist())

    def __len__(self):
        return max(math.ceil(self.n / self.world) - self.skip, 0)


def amp_dtype(precision, device):
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16 if precision == "bf16" else torch.float16


def main():
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        # rank 0 decodes the dev set while the others wait, so use a long timeout
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                timeout=datetime.timedelta(hours=2))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "log.jsonl")

    def log(obj):
        if is_main:
            obj = {"time": time.strftime("%H:%M:%S"), **obj}
            print(json.dumps(obj), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(obj) + "\n")

    if is_main:
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    tok = WhisperTokenizer.from_pretrained(args.model)
    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    tokens = Tokens(tok)
    collate = Collator(tokens.pad)

    # ---------------------------------------------------------------- data
    augmenter = Augmenter(args.noise_list, args.rir_list, args.aug, args.p_noise, args.p_rir,
                          (args.snr_min, args.snr_max)) if args.aug != "none" else None
    train_cfg = DataConfig(conditioning=args.conditioning, gap_sec=args.gap, position=args.position,
                           random_lead=args.random_lead, train=True,
                           enroll_min=args.enroll_min, enroll_max=args.enroll_max,
                           random_enrollment=not args.no_random_enrollment,
                           frame_labels=args.aux != "none", seed=args.seed)
    # every system trains on the same mixtures: those that fit next to the longest enrollment
    max_mix = max_mixture_sec(args.enroll_max, args.gap)
    if args.train_manifest:
        train_ex = read_manifest(args.train_manifest)
        n_all = len(train_ex)
        train_ex = [e for e in train_ex if fits(e, args.enroll_max, args.gap)]
        if args.max_train_examples:
            train_ex = train_ex[: args.max_train_examples]
        pool = None
        if args.speaker_pool and not args.no_random_enrollment:
            with open(args.speaker_pool) as f:
                pool = json.load(f)
        train_ds = ManifestDataset(train_ex, fe, tokens, train_cfg, augmenter=augmenter, speaker_pool=pool)
    else:
        utts = [u for path in args.train_index for u in read_index(path)]
        n_all = len(utts)
        probs = {int(k): float(v) for k, v in (x.split(":") for x in args.n_src_probs.split(","))}
        train_ds = OnTheFlyMixDataset(utts, fe, tokens, train_cfg, n_src_probs=probs, overlap=args.overlap,
                                      samples_per_epoch=args.max_train_examples or args.samples_per_epoch,
                                      max_mix_sec=max_mix, augmenter=augmenter)
        train_ex = train_ds
    sampler = EpochSampler(len(train_ds), args.seed, rank, world)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)

    dev_ex = [e for e in read_manifest(args.dev_manifest) if fits(e, args.dev_enroll_dur, args.gap)]
    dev_ex = subset(dev_ex, args.dev_subset, seed=0)
    dev_cfg = DataConfig(conditioning=args.conditioning, gap_sec=args.gap, position=args.position, train=False,
                         enroll_mode="target", enroll_dur=args.dev_enroll_dur)
    dev_dl = DataLoader(ManifestDataset(dev_ex, fe, tokens, dev_cfg), batch_size=args.dev_batch_size,
                        shuffle=False, collate_fn=collate, num_workers=min(args.num_workers, 4))
    log({"event": "data", "train_examples": len(train_ds), "train_pool": n_all, "max_mixture_sec": round(max_mix, 2),
         "source": "manifest" if args.train_manifest else "on-the-fly", "dev_examples": len(dev_ex), "world": world})

    # ---------------------------------------------------------------- model
    spk_enc = SpeakerEncoder(args.spk_encoder).to(device) if args.conditioning == "spk_emb" else None
    model = TSWhisper(args.model, conditioning=args.conditioning, aux=args.aux, aux_weight=args.aux_weight,
                      lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      lora_targets=args.lora_targets.split(","),
                      spk_dim=spk_enc.dim if spk_enc is not None else 512,
                      spk_encoder=args.spk_encoder, separator=args.separator, full_ft=args.full_ft).to(device)
    # data settings the evaluator must reuse
    model.cfg["train_data"] = {"gap": args.gap, "position": args.position, "enroll_len": args.enroll_max}
    model.whisper.config.use_cache = False
    if args.grad_ckpt:
        model.whisper.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = model.trainable_parameters()
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    log({"event": "model", "trainable_params": n_train, "total_params": n_total})

    steps_per_epoch = len(train_dl) // args.grad_accum
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, min(args.warmup_steps, total_steps // 10 + 1), total_steps)
    dtype = amp_dtype(args.precision, device)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)

    # ---------------------------------------------------------------- resume
    step, start_epoch, skip_batches, best = 0, 0, 0, float("inf")
    state_path = os.path.join(args.out_dir, "last", "trainer_state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu")
        missing = model.load_state_dict(st["trainable"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        step, start_epoch, skip_batches, best = st["step"], st["epoch"], st["batch_in_epoch"], st["best_wer"]
        log({"event": "resume", "step": step, "epoch": start_epoch, "batch_in_epoch": skip_batches})

    ddp = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
              find_unused_parameters=args.aux == "contrastive") if world > 1 else model
    core = model

    def save(tag, epoch, batch_in_epoch):
        if not is_main:
            return
        d = os.path.join(args.out_dir, tag)
        core.save(d)
        if tag == "last":
            torch.save({
                "trainable": {n: p.detach().cpu() for n, p in core.named_parameters() if p.requires_grad},
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch, "best_wer": best,
            }, os.path.join(d, "trainer_state.pt.tmp"))
            os.replace(os.path.join(d, "trainer_state.pt.tmp"), os.path.join(d, "trainer_state.pt"))

    def evaluate_dev():
        nonlocal best
        improved = False
        if is_main:
            core.whisper.config.use_cache = True
            hyps = decode(core, dev_dl, tok, device, dtype, spk_enc, progress=False)
            core.whisper.config.use_cache = False
            _, summ = score(dev_ex, hyps)
            wer = summ["ts"]["wer"]
            improved = wer < best
            best = min(best, wer)
            log({"event": "dev", "step": step, "ts_wer": round(wer, 2), "selection_acc": round(summ["selection_acc"], 2),
                 "ins_rate": round(summ["ts"]["ins_rate"], 2), "best": round(best, 2)})
        flag = torch.tensor([int(improved)], device=device)
        if world > 1:
            dist.broadcast(flag, 0)
        model.train()
        return bool(flag.item())

    # ---------------------------------------------------------------- train
    model.train()
    t0, run = time.time(), {"loss": 0.0, "ce": 0.0, "aux": 0.0, "n": 0}
    done = step >= total_steps
    for epoch in range(start_epoch, 10 ** 6):
        if done:
            break
        skip = skip_batches * args.batch_size if epoch == start_epoch else 0
        sampler.set_epoch(epoch, skip)
        train_ds.set_epoch(epoch)
        batch_in_epoch = skip_batches if epoch == start_epoch else 0
        micro = 0
        for batch in train_dl:
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            if spk_enc is not None:
                inputs["spk_emb"] = spk_enc(inputs.pop("enroll_wav"), inputs.pop("enroll_mask"))
            micro += 1
            batch_in_epoch += 1
            sync = micro % args.grad_accum == 0
            ctx = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with ctx:
                with autocast(device, dtype):
                    out = ddp(**inputs)
                loss = out["loss"] / args.grad_accum
                scaler.scale(loss).backward()
            run["loss"] += out["loss"].item()
            run["ce"] += out["ce"].item()
            run["aux"] += out["aux"].item() if "aux" in out else 0.0
            run["n"] += 1
            if not sync:
                continue
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, args.clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % args.log_every == 0:
                n = max(run["n"], 1)
                log({"event": "train", "step": step, "epoch": epoch, "loss": round(run["loss"] / n, 4),
                     "ce": round(run["ce"] / n, 4), "aux": round(run["aux"] / n, 4), "grad_norm": round(float(gn), 3),
                     "lr": sched.get_last_lr()[0], "sec_per_step": round((time.time() - t0) / args.log_every, 3)})
                t0, run = time.time(), {"loss": 0.0, "ce": 0.0, "aux": 0.0, "n": 0}
            if step % args.eval_every == 0 or step == total_steps:
                if evaluate_dev():
                    save("best", epoch, batch_in_epoch)
                save("last", epoch, batch_in_epoch)
                if world > 1:
                    dist.barrier()
            if step >= total_steps:
                done = True
                break
        skip_batches = 0
        opt.zero_grad(set_to_none=True)  # drop a partial accumulation at the epoch end
        if not done:
            save("last", epoch + 1, 0)
    log({"event": "finished", "step": step, "best_dev_ts_wer": best})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
