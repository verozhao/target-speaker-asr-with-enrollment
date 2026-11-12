"""Text normalization and WER scoring.

All systems are scored with the Whisper English text normalizer so that casing,
punctuation and spelling variants do not count as errors.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence

import jiwer

_SPECIAL = re.compile(r"<\|[^|]*\|>")


@lru_cache(maxsize=1)
def _normalizer():
    from transformers import WhisperTokenizer

    tok = WhisperTokenizer.from_pretrained("openai/whisper-small")
    return tok.normalize


def normalize(text: str) -> str:
    text = _SPECIAL.sub(" ", text or "")
    out = _normalizer()(text)
    return " ".join(out.split())


def training_text(librispeech_text: str) -> str:
    """Target text used for training: LibriSpeech transcripts lower-cased."""
    return " ".join(librispeech_text.lower().split())


def word_errors(ref: str, hyp: str) -> Dict[str, int]:
    """Edit counts on normalized text. ref/hyp must already be normalized."""
    if not ref.strip():
        return {"sub": 0, "del": 0, "ins": len(hyp.split()), "hits": 0, "ref_words": 0}
    if not hyp.strip():
        n_ref = len(ref.split())
        return {"sub": 0, "del": n_ref, "ins": 0, "hits": 0, "ref_words": n_ref}
    o = jiwer.process_words(ref, hyp)
    return {
        "sub": o.substitutions,
        "del": o.deletions,
        "ins": o.insertions,
        "hits": o.hits,
        "ref_words": o.substitutions + o.deletions + o.hits,
    }


def utt_wer(ref: str, hyp: str) -> float:
    e = word_errors(ref, hyp)
    return (e["sub"] + e["del"] + e["ins"]) / max(e["ref_words"], 1)


def corpus_scores(errs: Iterable[Dict[str, int]]) -> Dict[str, float]:
    tot = {"sub": 0, "del": 0, "ins": 0, "hits": 0, "ref_words": 0}
    n = 0
    for e in errs:
        n += 1
        for k in tot:
            tot[k] += e[k]
    rw = max(tot["ref_words"], 1)
    return {
        "n_utts": n,
        "ref_words": tot["ref_words"],
        "wer": 100.0 * (tot["sub"] + tot["del"] + tot["ins"]) / rw,
        "sub_rate": 100.0 * tot["sub"] / rw,
        "del_rate": 100.0 * tot["del"] / rw,
        "ins_rate": 100.0 * tot["ins"] / rw,
    }


def closest_speaker(hyp: str, refs: Sequence[str]) -> int:
    """Index of the reference with the lowest WER against hyp (ties -> lowest index)."""
    wers: List[float] = [utt_wer(r, hyp) for r in refs]
    return min(range(len(refs)), key=lambda i: (wers[i], i))
