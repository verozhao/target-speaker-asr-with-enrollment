#!/usr/bin/env python
"""Evaluate a system on a target-speaker manifest (Libri2Mix/Libri3Mix or AMI windows).

Zero-shot systems (no training):
  --system whisper_mixture   unconditioned Whisper on the mixture
  --system whisper_prefix    Whisper with the enrollment prefix, no fine-tuning
  --system whisper_source    Whisper on the clean target source (topline, LibriMix only)
Trained models:
  --ckpt exp/<run>/best      conditioning, separator, gap and position come from the checkpoint

Enrollment controls on identical mixtures (--enroll-mode):
  target      the target speaker's enrollment
  interferer  the enrollment of ANOTHER speaker in the mixture; scored against that speaker
  other       a speaker who is not in the mixture
  silence     digital silence of the same length
  none        no enrollment at all
Enrollment source (--enroll-field): enrollment (default; on AMI = same session) or
enrollment_other_session (AMI). Duration: --enroll-dur 1|2|5|10 with a fixed --fit-dur.
Quality: --condition enroll_perturbed --snr 20|10|5|0 [--no-reverb].
Position tests: --position suffix, --lead-sec 5. Text prompt: --text-prompt "...".
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
from tsasr.data import ENROLL_MODES, Collator, DataConfig, ManifestDataset, Tokens, fits, read_manifest
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
    p.add_argument("--enroll-mode", default="target", choices=ENROLL_MODES)
    p.add_argument("--enroll-field", default="enrollment", help="enrollment | enrollment_other_session")
    p.add_argument("--enroll-dur", type=float, default=5.0)
    p.add_argument("--fit-dur", type=float, default=5.0,
                   help="keep mixtures that fit with an enrollment of this length (same subset for every run)")
    p.add_argument("--gap", type=float, default=None, help="default: the checkpoint's training gap (0.5 s)")
    p.add_argument("--position", default=None, choices=["prefix", "suffix"], help="default: as trained")
    p.add_argument("--lead-sec", type=float, default=0.0, help="silence before the input (position test)")
    p.add_argument("--mix-start", type=float, default=None,
                   help="anchored layout (mixture starts at this second); default: as trained")
    p.add_argument("--text-prompt", default=None, help="Whisper previous-text prompt")
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
    p.add_argument("--require-key", action="append", default=[],
                   help="keep examples whose manifest field is true, e.g. enroll_full_both (AMI)")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--tag", default=None)
    return p.parse_args()


def attach_interferer(examples):
    """Give each example the enrollment of another speaker in the same mixture."""
    by_key = {(e["mixture_id"], e["target_index"]): e for e in examples}
    out = []
    for e in examples:
        if e.get("interferer_enrollment"):  # AMI windows carry their own
            out.append({**e, "enrolled_index": e.get("interferer_index", 1)})
            continue
        j = (e["target_index"] + 1) % e["n_src"]
        sib = by_key.get((e["mixture_id"], j))
        if sib is not None:
            out.append({**e, "interferer_enrollment": sib["enrollment"], "enrolled_index": j})
    return out


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if device.type == "cuda" and a.precision != "fp32":
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if a.precision == "auto" \
            else (torch.bfloat16 if a.precision == "bf16" else torch.float16)

    spk_enc, trained = None, {}
    if a.ckpt:
        model = TSWhisper.load(a.ckpt)
        conditioning, input_kind = model.conditioning, "mixture"
        model_name = model.cfg["model_name"]
        trained = model.cfg.get("train_data", {})
        if conditioning == "spk_emb":
            spk_enc = SpeakerEncoder(model.cfg.get("spk_encoder", "wavlm")).to(device)
        system = os.path.normpath(a.ckpt)
    else:
        conditioning, input_kind = ZERO_SHOT[a.system]
        model = TSWhisper(a.model, conditioning=conditioning, use_lora=False)
        model_name, system = a.model, a.system
    gap = a.gap if a.gap is not None else trained.get("gap", 0.5)
    position = a.position or trained.get("position", "prefix")
    mix_start = a.mix_start if a.mix_start is not None else trained.get("mix_start")
    model.to(device).eval()
    model.whisper.config.use_cache = True

    tok = WhisperTokenizer.from_pretrained(model_name)
    fe = WhisperFeatureExtractor.from_pretrained(model_name)
    tokens = Tokens(tok)

    all_ex = read_manifest(a.manifest)
    ex = [e for e in all_ex
          if fits(e, a.fit_dur, gap, a.lead_sec, mix_start) and all(e.get(k) for k in a.require_key)]
    if a.enroll_mode == "interferer":
        ex = attach_interferer(ex)
    elif a.enroll_mode in ("target", "silence"):
        ex = [e for e in ex if e.get(a.enroll_field)]
    elif a.enroll_mode == "other":
        ex = [e for e in ex if e.get("other_enrollment")]
    ex = subset(ex, a.max_examples, seed=0)
    perturber = None
    if a.condition != "clean":
        perturber = EvalPerturber(a.noise_list, a.rir_list, a.condition, a.snr,
                                  reverb=not a.no_reverb, noise=not a.no_noise)
    cfg = DataConfig(conditioning=conditioning, input_kind=input_kind, gap_sec=gap, position=position,
                     lead_sec=a.lead_sec, mix_start=mix_start, train=False, enroll_mode=a.enroll_mode, enroll_field=a.enroll_field,
                     enroll_dur=a.enroll_dur)
    ds = ManifestDataset(ex, fe, tokens, cfg, perturber=perturber)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=Collator(tokens.pad),
                    num_workers=a.num_workers)
    prompt_ids = tok.get_prompt_ids(a.text_prompt, return_tensors="pt") if a.text_prompt else None

    t0 = time.time()
    hyps = decode(model, dl, tok, device, dtype, spk_enc, num_beams=a.num_beams, prompt_ids=prompt_ids)
    recs, summ = score(ex, hyps, a.enroll_mode)
    run_name = (os.path.basename(os.path.dirname(system.rstrip("/"))) + "-" + os.path.basename(system)) \
        if a.ckpt else system
    parts = [run_name, os.path.splitext(os.path.basename(a.manifest))[0], a.condition, a.enroll_mode,
             f"enr{a.enroll_dur:g}s"]
    if a.enroll_field != "enrollment":
        parts.append(a.enroll_field.replace("enrollment_", ""))
    if a.condition != "clean":
        parts.append(f"snr{a.snr:g}" + ("" if not a.no_reverb else "-norev") + ("" if not a.no_noise else "-nonoise"))
    if a.fit_dur != 5.0:
        parts.append(f"fit{a.fit_dur:g}")
    parts += [k.replace("_", "") for k in a.require_key]
    if a.position and a.position != trained.get("position", "prefix"):
        parts.append(a.position)
    if a.lead_sec:
        parts.append(f"lead{a.lead_sec:g}")
    if a.mix_start is not None and a.mix_start != trained.get("mix_start"):
        parts.append(f"start{a.mix_start:g}")
    if a.text_prompt:
        parts.append("textprompt")
    if a.num_beams > 1:
        parts.append(f"beam{a.num_beams}")
    tag = a.tag or "_".join(parts)
    summ.update({"system": system, "manifest": a.manifest, "condition": a.condition, "enroll_dur": a.enroll_dur,
                 "enroll_field": a.enroll_field, "fit_dur": a.fit_dur, "gap": gap, "position": position,
                 "lead_sec": a.lead_sec, "mix_start": mix_start, "text_prompt": a.text_prompt,
                 "require_key": a.require_key,
                 "n_manifest": len(all_ex),
                 "n_scored": len(recs), "num_beams": a.num_beams, "snr": a.snr if a.condition != "clean" else None,
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
