"""AMI corpus helpers: word timings, target-speaker windows and enrollment regions.

Inputs are the public manual annotations (ami_public_manual_1.6.2) and the single
distant microphone audio (<session>.Array1-01.wav). Session ids look like ES2004a.
Speakers are identified by their corpus-wide global id (e.g. FEE013), which links the
same person across the sessions of a series (ES2004a-d).
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

Word = Tuple[float, float, str]

# full-corpus ASR partition
AMI_TEST = [f"{s}{x}" for s in ("EN2002", "ES2004", "IS1009", "TS3003") for x in "abcd"]
AMI_DEV = [f"{s}{x}" for s in ("ES2011", "IS1008", "TS3004") for x in "abcd"] + \
          ["IB4001", "IB4002", "IB4003", "IB4004", "IB4005", "IB4010", "IB4011"]


def _h(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def read_speaker_map(annotation_dir: str) -> Dict[str, Dict[str, str]]:
    """session id -> {agent letter: global speaker id}, from corpusResources/*.xml."""
    out: Dict[str, Dict[str, str]] = {}
    for f in sorted(glob.glob(os.path.join(annotation_dir, "corpusResources", "*.xml"))):
        for el in ET.parse(f).getroot().iter():
            sess = el.get("observation")
            if not sess:
                continue
            agents = {c.get("nxt_agent"): c.get("global_name") for c in el
                      if c.get("global_name") and c.get("nxt_agent")}
            if agents:
                out[sess] = agents
    return out


def clean_token(tok: str) -> str:
    # spelled letters are written X_M_L_ in the annotation
    return tok.replace("_", "") if "_" in tok else tok


def read_words(annotation_dir: str, session: str, speaker_map: Dict[str, Dict[str, str]]) -> Dict[str, List[Word]]:
    """global speaker id -> time-sorted (start, end, token); punctuation dropped."""
    out: Dict[str, List[Word]] = {}
    agents = speaker_map.get(session, {})
    for f in sorted(glob.glob(os.path.join(annotation_dir, "words", f"{session}.*.words.xml"))):
        agent = os.path.basename(f).split(".")[1]
        spk = agents.get(agent, f"{session}_{agent}")
        ws = []
        for el in ET.parse(f).getroot():
            if el.tag != "w" or el.get("punc") == "true" or not (el.text or "").strip():
                continue
            st, en = el.get("starttime"), el.get("endtime")
            if st is None or en is None:
                continue
            ws.append((float(st), float(en), clean_token(el.text.strip())))
        out[spk] = sorted(ws)
    return out


def merge(intervals: Sequence[Tuple[float, float]], max_pause: float) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= max_pause:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _overlaps(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def target_windows(words: List[Word], max_len: float = 24.5, pad: float = 0.1,
                   min_words: int = 3) -> List[Tuple[float, float, int, int]]:
    """Greedy windows over one speaker's words: (start, end, first_word, last_word).
    A window never cuts one of this speaker's words."""
    out, i = [], 0
    while i < len(words):
        s = words[i][0]
        j = i
        while j + 1 < len(words) and words[j + 1][1] - s <= max_len - 2 * pad:
            j += 1
        if words[j][1] - s > max_len - 2 * pad:  # a single over-long token
            i = j + 1
            continue
        prev_end = words[i - 1][1] if i > 0 else 0.0
        next_start = words[j + 1][0] if j + 1 < len(words) else float("inf")
        start = max(s - pad, prev_end, 0.0)
        end = min(words[j][1] + pad, next_start)
        if j - i + 1 >= min_words:
            out.append((round(start, 3), round(end, 3), i, j))
        i = j + 1
    return out


def solo_regions(words_by_spk: Dict[str, List[Word]], spk: str, min_len: float = 2.0,
                 max_pause: float = 0.5, guard: float = 0.2) -> List[Tuple[float, float]]:
    """Stretches of `spk` speech with nobody else speaking within `guard` seconds."""
    regions = merge([(s, e) for s, e, _ in words_by_spk.get(spk, [])], max_pause)
    others = merge([(s, e) for o, ws in words_by_spk.items() if o != spk for s, e, _ in ws], 0.0)
    keep, k = [], 0
    for s, e in regions:
        while k < len(others) and others[k][1] <= s - guard:
            k += 1
        clash = any(_overlaps((s - guard, e + guard), others[m]) for m in range(k, min(k + 50, len(others)))
                    if others[m][0] < e + guard)
        if not clash and e - s >= min_len:
            keep.append((round(s, 3), round(e, 3)))
    return keep


def pick_region(regions: List[Tuple[float, float]], want: float, key: str,
                avoid: Optional[Tuple[float, float]] = None, margin: float = 1.0) -> Optional[Tuple[float, float]]:
    cands = [r for r in regions if avoid is None or not _overlaps((r[0] - margin, r[1] + margin), avoid)]
    if not cands:
        return None
    long_enough = [r for r in cands if r[1] - r[0] >= want + 0.5]
    if long_enough:
        return sorted(long_enough, key=lambda r: _h(key, r))[0]
    return max(cands, key=lambda r: (r[1] - r[0], -_h(key, r)))


def words_in(words: List[Word], start: float, end: float) -> List[Word]:
    return [w for w in words if w[0] >= start and w[1] <= end]


def text_of(words: List[Word]) -> str:
    return re.sub(r"\s+", " ", " ".join(w[2] for w in words)).strip()
