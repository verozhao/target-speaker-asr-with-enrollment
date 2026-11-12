#!/usr/bin/env python
"""Evaluate a system on a target-speaker manifest.

Zero-shot baselines (no training):
  --system whisper_mixture   unconditioned Whisper on the mixture
  --system whisper_prefix    Whisper with the enrollment prefix, no fine-tuning
  --system whisper_source    Whisper on the clean target source (topline)
Trained models:
  --ckpt exp/<run>/best      (conditioning is read from the checkpoint)

Enrollment controls on identical mixtures: --enroll-mode target|other|silence|none
Enrollment duration: --enroll-dur 1|2|5|10 (keep --fit-dur fixed so the subset is identical)
Acoustic conditions: --condition clean|enroll_perturbed|mix_perturbed|both_perturbed
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from tsasr.augment import EVAL_CONDITIONS, EvalPerturber
from tsasr.data import Collator, DataConfig, Tokens, TSASRDataset, fits, read_manifest
from tsasr.evaluation import decode, score, subset, write_jsonl
from tsasr.model import SpeakerEncoder, TSWhisper

ZERO_SHOT = {
    "whisper_mixture": ("none", "mixture"),
    "whisper_prefix": ("prefix", "mixture"),
    "whisper_source": ("none", "source"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt")
    g.add_argument("--system", choices=sorted(ZERO_SHOT))
    p.add_argument("--model", default="openai/whisper-small", help="zero-shot systems only")
    p.add_argument("--enroll-mode", default="target", choices=["target", "other", "silence", "none"])
    p.add_argument("--enroll-dur", type=float, default=5.0)
    p.add_argument("--fit-dur", type=float, default=10.0,
                   help="keep mixtures that fit with an enrollment of this length (same subset for every run)")
    p.add_argument("--gap", type=float, default=0.5)
    p.add_argument("--condition", default="clean", choices=EVAL_CONDITIONS)
    p.add_argument("--noise-list", default=None, help="test-split noise list")
    p.add_argument("--rir-list", default=None, help="test-split RIR list")
    p.add_argument("--snr", type=float, default=5.0)
    p.add_argument("--no-reverb", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if device.type == "cuda" and a.precision != "fp32":
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if a.precision == "auto" \
            else (torch.bfloat16 if a.precision == "bf16" else torch.float16)

    spk_enc = None
    if a.ckpt:
        model = TSWhisper.load(a.ckpt)
        conditioning, input_kind = model.conditioning, "mixture"
        model_name = model.cfg["model_name"]
        if conditioning == "spk_emb":
            spk_enc = SpeakerEncoder(model.cfg.get("spk_encoder", "wavlm")).to(device)
        system = os.path.normpath(a.ckpt)
    else:
        conditioning, input_kind = ZERO_SHOT[a.system]
        model = TSWhisper(a.model, conditioning=conditioning, use_lora=False)
        model_name, system = a.model, a.system
    model.to(device).eval()
    model.whisper.config.use_cache = True

    tok = WhisperTokenizer.from_pretrained(model_name)
    fe = WhisperFeatureExtractor.from_pretrained(model_name)
    tokens = Tokens(tok)

    all_ex = read_manifest(a.manifest)
    ex = [e for e in all_ex if fits(e, a.fit_dur, a.gap)]
    ex = subset(ex, a.max_examples, seed=0)
    perturber = None
    if a.condition != "clean":
        perturber = EvalPerturber(a.noise_list, a.rir_list, a.condition, a.snr,
                                  reverb=not a.no_reverb, noise=not a.no_noise)
    cfg = DataConfig(conditioning=conditioning, input_kind=input_kind, gap_sec=a.gap, train=False,
                     enroll_mode=a.enroll_mode, enroll_dur=a.enroll_dur)
    ds = TSASRDataset(ex, fe, tokens, cfg, perturber=perturber)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=Collator(tokens.pad),
                    num_workers=a.num_workers)

    t0 = time.time()
    hyps = decode(model, dl, tok, device, dtype, spk_enc, num_beams=a.num_beams)
    recs, summ = score(ex, hyps, a.enroll_mode)
    tag = a.tag or "_".join([
        os.path.basename(os.path.dirname(system.rstrip("/"))) + "-" + os.path.basename(system) if a.ckpt else system,
        os.path.splitext(os.path.basename(a.manifest))[0], a.condition, a.enroll_mode,
        f"enr{a.enroll_dur:g}s"] + ([f"beam{a.num_beams}"] if a.num_beams > 1 else []))
    summ.update({"system": system, "manifest": a.manifest, "condition": a.condition, "enroll_dur": a.enroll_dur,
                 "fit_dur": a.fit_dur, "n_manifest": len(all_ex), "n_scored": len(recs),
                 "num_beams": a.num_beams, "snr": a.snr if a.condition != "clean" else None,
                 "reverb": (not a.no_reverb) if a.condition != "clean" else None,
                 "decode_sec": round(time.time() - t0, 1)})
    os.makedirs(a.out_dir, exist_ok=True)
    write_jsonl(os.path.join(a.out_dir, tag + ".hyps.jsonl"), recs)
    with open(os.path.join(a.out_dir, tag + ".summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps({"tag": tag, "ts_wer": round(summ["ts"]["wer"], 2),
                      "ins_rate": round(summ["ts"]["ins_rate"], 2),
                      "selection_acc": round(summ["selection_acc"], 2),
                      "best_spk_wer": round(summ["best_spk"]["wer"], 2), "n": len(recs)}))


if __name__ == "__main__":
    main()
