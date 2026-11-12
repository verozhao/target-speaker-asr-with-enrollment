"""LibriSpeech utterance index (speaker, chapter, transcript, path, duration)."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import soundfile as sf


def index_subset(root: str, subset: str) -> Dict[str, dict]:
    """root = .../LibriSpeech, subset = e.g. train-clean-100. Returns utt_id -> record."""
    utts = {}
    base = os.path.join(root, subset)
    for spk in sorted(os.listdir(base)):
        sd = os.path.join(base, spk)
        if not os.path.isdir(sd):
            continue
        for ch in sorted(os.listdir(sd)):
            cd = os.path.join(sd, ch)
            tf = os.path.join(cd, f"{spk}-{ch}.trans.txt")
            if not os.path.exists(tf):
                continue
            with open(tf) as f:
                for ln in f:
                    uid, text = ln.strip().split(" ", 1)
                    utts[uid] = {"utt_id": uid, "speaker": spk, "chapter": ch, "subset": subset, "text": text,
                                 "path": os.path.abspath(os.path.join(cd, uid + ".flac"))}
    with ThreadPoolExecutor(16) as ex:
        durs = list(ex.map(lambda u: sf.info(u["path"]).duration, utts.values()))
    for u, d in zip(utts.values(), durs):
        u["duration"] = round(d, 3)
    return utts


def load_or_build(root: str, subset: str, cache: str) -> Dict[str, dict]:
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    utts = index_subset(root, subset)
    with open(cache, "w") as f:
        json.dump(utts, f)
    return utts


def read_index(path: str) -> List[dict]:
    with open(path) as f:
        d = json.load(f)
    return list(d.values()) if isinstance(d, dict) else d
