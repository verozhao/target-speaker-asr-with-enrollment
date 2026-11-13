#!/usr/bin/env python
"""Cost of losing 5 s of context.

A 5 s enrollment + 0.5 s separator leaves 24.5 s of Whisper's 30 s window for audio.
This script decodes the same long-form audio with non-overlapping windows of each
length in --windows and reports WER, the number of windows (compute) and decode time.
The difference between 30 s and 24.5 s windows is the context cost of the prompt.

Long-form audio: whole LibriSpeech chapters (--subset, default test-clean), utterances
concatenated in order with 0.3 s pauses and cut to --max-minutes. Plain Whisper is the
default model, so the number isolates the window-length effect from the conditioning.

  python scripts/context_cost.py --librispeech-dir $S/LibriSpeech --out-dir results
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from tsasr import SAMPLE_RATE
from tsasr.audio import load_audio
from tsasr.evaluation import autocast
from tsasr.librispeech import index_subset
from tsasr.model import TSWhisper
from tsasr.text import corpus_scores, normalize, word_errors


def chapters(root, subset, max_minutes, max_chapters):
    utts = index_subset(root, subset)
    by = defaultdict(list)
    for u in utts.values():
        by[(u["speaker"], u["chapter"])].append(u)
    out = []
    for key in sorted(by)[:max_chapters]:
        us = sorted(by[key], key=lambda u: u["utt_id"])
        pieces, texts, total = [], [], 0.0
        for u in us:
            if total + u["duration"] > max_minutes * 60:
                break
            pieces += [load_audio(u["path"]), np.zeros(int(0.3 * SAMPLE_RATE), np.float32)]
            texts.append(u["text"])
            total += u["duration"] + 0.3
        out.append({"id": f"{key[0]}-{key[1]}", "audio": np.concatenate(pieces), "text": " ".join(texts)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-dir", required=True)
    p.add_argument("--subset", default="test-clean")
    p.add_argument("--max-minutes", type=float, default=5.0)
    p.add_argument("--max-chapters", type=int, default=87, help="test-clean has 87 chapters")
    p.add_argument("--windows", type=float, nargs="+", default=[30.0, 24.5])
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--ckpt", default=None, help="optional trained model, decoded without enrollment")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else None
    if a.ckpt:
        model = TSWhisper.load(a.ckpt)
        name = model.cfg["model_name"]
        if model.frontend == "ssl":
            sys.exit("context_cost.py measures Whisper's 30 s window; use a Whisper-frontend model")
    else:
        model = TSWhisper(a.model, conditioning="none", use_lora=False)
        name = a.model
    model.to(device).eval()
    tok, fe = WhisperTokenizer.from_pretrained(name), WhisperFeatureExtractor.from_pretrained(name)
    docs = chapters(a.librispeech_dir, a.subset, a.max_minutes, a.max_chapters)

    results = {"model": a.ckpt or a.model, "subset": a.subset, "n_docs": len(docs),
               "audio_minutes": round(sum(len(d["audio"]) for d in docs) / SAMPLE_RATE / 60, 1), "windows": {}}
    for w in a.windows:
        n = int(w * SAMPLE_RATE)
        chunks = [(di, d["audio"][s: s + n]) for di, d in enumerate(docs) for s in range(0, len(d["audio"]), n)]
        hyp = defaultdict(list)
        t0 = time.time()
        for b in range(0, len(chunks), a.batch_size):
            batch = chunks[b: b + a.batch_size]
            feats = fe([c for _, c in batch], sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
            with autocast(device, dtype):
                seqs = model.generate(feats)
            for (di, _), t in zip(batch, tok.batch_decode(seqs, skip_special_tokens=True)):
                hyp[di].append(t.strip())
        errs = [word_errors(normalize(d["text"]), normalize(" ".join(hyp[i]))) for i, d in enumerate(docs)]
        results["windows"][f"{w:g}"] = {"n_windows": len(chunks), "decode_sec": round(time.time() - t0, 1),
                                        **corpus_scores(errs)}
        print(json.dumps({"window_sec": w, **results["windows"][f"{w:g}"]}))
    ws = [f"{w:g}" for w in a.windows]
    if len(ws) >= 2:
        r0, r1 = results["windows"][ws[0]], results["windows"][ws[1]]
        results["cost"] = {"wer_abs_increase": round(r1["wer"] - r0["wer"], 2),
                           "window_ratio": round(r1["n_windows"] / max(r0["n_windows"], 1), 3)}
    os.makedirs(a.out_dir, exist_ok=True)
    tag = "context_cost_" + (os.path.basename(os.path.dirname(a.ckpt.rstrip("/"))) if a.ckpt else a.model.split("/")[-1])
    with open(os.path.join(a.out_dir, tag + ".json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results.get("cost", {})))


if __name__ == "__main__":
    main()
