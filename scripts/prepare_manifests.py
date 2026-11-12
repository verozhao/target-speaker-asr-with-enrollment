#!/usr/bin/env python
"""Build target-speaker manifests from LibriMix output metadata + LibriSpeech.

One line per (mixture, target speaker). Each target gets an enrollment utterance
of the same speaker that is not a source of that mixture. Utterances that are not
used as a source anywhere in the split are preferred ("unused"); when every
utterance of a speaker is used (typical for test-clean, where 3000 mixtures reuse
2620 utterances) the enrollment comes from another mixture ("other_mixture").
Long utterances (>= --prefer-min-dur) are preferred so that the same enrollment
can be cropped to 1, 2, 5 and 10 s.

Example:
  python scripts/prepare_manifests.py --librispeech-dir $S/LibriSpeech \
      --librimix-dir $S/Libri2Mix/wav16k/max --n-src 2 --split test --out-dir data
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

SUBSET = {"train-100": "train-clean-100", "train-360": "train-clean-360", "dev": "dev-clean",
          "test": "test-clean"}


def h(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def index_librispeech(root: str, subset: str, cache: str):
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
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
                    utts[uid] = {"utt_id": uid, "speaker": spk, "text": text,
                                 "path": os.path.abspath(os.path.join(cd, uid + ".flac"))}
    with ThreadPoolExecutor(16) as ex:
        durs = list(ex.map(lambda u: sf.info(u["path"]).duration, utts.values()))
    for u, d in zip(utts.values(), durs):
        u["duration"] = round(d, 3)
    with open(cache, "w") as f:
        json.dump(utts, f)
    return utts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-dir", required=True, help=".../LibriSpeech")
    p.add_argument("--librimix-dir", required=True, help=".../Libri2Mix/wav16k/max")
    p.add_argument("--n-src", type=int, required=True)
    p.add_argument("--split", required=True, choices=sorted(SUBSET))
    p.add_argument("--mix-type", default="mix_clean")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefer-min-dur", type=float, default=11.0)
    p.add_argument("--gap", type=float, default=0.5)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    subset = SUBSET[a.split]
    utts = index_librispeech(a.librispeech_dir, subset,
                             os.path.join(a.out_dir, f"librispeech_{subset}_index.json"))
    by_spk = defaultdict(list)
    for u in utts.values():
        by_spk[u["speaker"]].append(u)

    md = os.path.join(a.librimix_dir, "metadata", f"mixture_{a.split}_{a.mix_type}.csv")
    with open(md) as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} mixtures in {md}")

    def local(path, sub, mid):
        if path and os.path.exists(path):
            return path
        alt = os.path.join(a.librimix_dir, a.split, sub, mid + ".wav")
        if not os.path.exists(alt):
            sys.exit(f"missing audio for {mid}: {path} / {alt}")
        return alt

    mixtures = []
    for r in rows:
        mid = r["mixture_ID"]
        ids = mid.split("_")
        if len(ids) != a.n_src or any(i not in utts for i in ids):
            sys.exit(f"cannot map mixture_ID {mid} to {a.n_src} {subset} utterances")
        mixtures.append({
            "mixture_id": mid, "source_utts": ids,
            "mixture": local(r.get("mixture_path"), a.mix_type, mid),
            "sources": [local(r.get(f"source_{k + 1}_path"), f"s{k + 1}", mid) for k in range(a.n_src)],
            "duration": round(int(float(r["length"])) / 16000, 3),
        })
    used = {u for m in mixtures for u in m["source_utts"]}
    split_speakers = sorted({utts[u]["speaker"] for u in used})

    def pick(spk, exclude, key):
        cands = [u for u in by_spk[spk] if u["utt_id"] not in exclude]
        if not cands:
            return None
        best = min(cands, key=lambda u: (u["utt_id"] in used, u["duration"] < a.prefer_min_dur,
                                         h(key, u["utt_id"])))
        return {"utt_id": best["utt_id"], "path": best["path"], "duration": best["duration"],
                "policy": "other_mixture" if best["utt_id"] in used else "unused"}

    out, stats = [], Counter()
    for m in mixtures:
        spks = [utts[u]["speaker"] for u in m["source_utts"]]
        if len(set(spks)) != len(spks):
            stats["skipped_same_speaker"] += 1
            continue
        texts = [utts[u]["text"] for u in m["source_utts"]]
        for k in range(a.n_src):
            ex_id = f"{m['mixture_id']}__s{k + 1}"
            enr = pick(spks[k], set(m["source_utts"]), ex_id)
            if enr is None:
                stats["skipped_no_enrollment"] += 1
                continue
            others = [s for s in split_speakers if s not in spks]
            o_spk = others[h(ex_id, "other") % len(others)]
            o_enr = pick(o_spk, set(), ex_id + "|other")
            o_enr["speaker"] = o_spk
            stats[f"policy_{enr['policy']}"] += 1
            stats["long_enrollment"] += int(enr["duration"] >= a.prefer_min_dur)
            out.append({"id": ex_id, "mixture_id": m["mixture_id"], "split": a.split, "n_src": a.n_src,
                        "duration": m["duration"], "mixture": m["mixture"], "sources": m["sources"],
                        "source_utts": m["source_utts"], "speakers": spks, "texts": texts,
                        "target_index": k, "enrollment": enr, "other_enrollment": o_enr})

    name = f"libri{a.n_src}mix_{a.split}"
    with open(os.path.join(a.out_dir, name + ".jsonl"), "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")
    pool = {s: [{"utt_id": u["utt_id"], "path": u["path"], "duration": u["duration"]}
                for u in by_spk[s] if u["duration"] >= 1.0] for s in split_speakers}
    with open(os.path.join(a.out_dir, f"speaker_pool_{a.split}.json"), "w") as f:
        json.dump(pool, f)

    d = np.array([e["duration"] for e in out]) if out else np.zeros(1)
    ed = np.array([e["enrollment"]["duration"] for e in out]) if out else np.zeros(1)
    summary = {"manifest": name, "mixtures": len(mixtures), "examples": len(out),
               "speakers": len(split_speakers), **stats,
               "mixture_sec": {"mean": round(float(d.mean()), 2), "max": round(float(d.max()), 2)},
               "enrollment_sec": {"mean": round(float(ed.mean()), 2), "min": round(float(ed.min()), 2)},
               "fits_30s_with_enrollment": {f"{t:g}s": int(np.sum(d + a.gap + t + 0.02 <= 30.0))
                                            for t in (1, 2, 5, 10)}}
    # speaker overlap with manifests of other splits already in out_dir
    overlaps = {}
    for fn in os.listdir(a.out_dir):
        if fn.endswith(".jsonl") and fn.startswith("libri") and fn != name + ".jsonl":
            with open(os.path.join(a.out_dir, fn)) as f:
                other = {s for ln in f for s in json.loads(ln)["speakers"]}
            if not fn.endswith(f"_{a.split}.jsonl"):
                overlaps[fn] = len(other & set(split_speakers))
    summary["speaker_overlap_with_other_splits"] = overlaps
    with open(os.path.join(a.out_dir, name + ".stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if any(overlaps.values()):
        sys.exit("ERROR: speakers overlap across splits")


if __name__ == "__main__":
    main()
