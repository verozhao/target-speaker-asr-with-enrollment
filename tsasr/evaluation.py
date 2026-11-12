"""Batched decoding and target-speaker scoring."""
from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .text import closest_speaker, corpus_scores, normalize, word_errors


def autocast(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=dtype)


@torch.no_grad()
def decode(model, loader, tokenizer, device, amp_dtype=None, spk_encoder=None, num_beams: int = 1,
           max_new_tokens: int = 220, progress: bool = True) -> Dict[str, str]:
    model.eval()
    hyps: Dict[str, str] = {}
    for batch in tqdm(loader, desc="decode", disable=not progress):
        feats = batch["input_features"].to(device)
        spk = None
        if spk_encoder is not None:
            spk = spk_encoder(batch["enroll_wav"].to(device), batch["enroll_mask"].to(device))
        with autocast(device, amp_dtype):
            seqs = model.generate(feats, spk_emb=spk, num_beams=num_beams, max_new_tokens=max_new_tokens)
        texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        for i, t in zip(batch["ids"], texts):
            hyps[i] = t.strip()
    return hyps


def score(examples: List[dict], hyps: Dict[str, str], enroll_mode: str = "target"):
    """Per-utterance records and a corpus summary.

    ts_wer           WER against the enrolled (target) speaker's transcript
    selection_acc    share of outputs closest (by WER) to the target transcript
    best_spk_wer     WER against whichever source transcript is closest (speaker-agnostic)
    """
    recs, errs_t, errs_best = [], [], []
    sel, n_hyp_words, n_tgt_words, n_empty = 0, 0, 0, 0
    for ex in examples:
        if ex["id"] not in hyps:
            continue
        hyp = hyps[ex["id"]]
        k = ex["target_index"]
        refs = [normalize(t) for t in ex["texts"]]
        nh = normalize(hyp)
        e_all = [word_errors(r, nh) for r in refs]
        best = closest_speaker(nh, refs)
        errs_t.append(e_all[k])
        errs_best.append(e_all[best])
        sel += int(best == k)
        n_hyp_words += len(nh.split())
        n_tgt_words += len(refs[k].split())
        n_empty += int(len(nh.split()) == 0)
        recs.append({
            "id": ex["id"], "mixture_id": ex["mixture_id"], "target_index": k,
            "enroll_mode": enroll_mode, "hyp": hyp, "hyp_norm": nh, "ref_norm": refs[k],
            "errors_target": e_all[k], "closest_speaker": best,
            "wer_per_speaker": [round((e["sub"] + e["del"] + e["ins"]) / max(e["ref_words"], 1), 4)
                                for e in e_all],
        })
    n = max(len(recs), 1)
    summary = {
        "enroll_mode": enroll_mode,
        "ts": corpus_scores(errs_t),
        "best_spk": corpus_scores(errs_best),
        "selection_acc": 100.0 * sel / n,
        "hyp_to_target_word_ratio": n_hyp_words / max(n_tgt_words, 1),
        "empty_output_rate": 100.0 * n_empty / n,
    }
    return recs, summary


def write_jsonl(path: str, rows: List[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def subset(examples: List[dict], n: Optional[int], seed: int = 0) -> List[dict]:
    if n is None or n >= len(examples):
        return examples
    idx = np.random.default_rng(seed).permutation(len(examples))[:n]
    return [examples[i] for i in sorted(idx)]
