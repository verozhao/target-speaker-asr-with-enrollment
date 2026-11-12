#!/usr/bin/env python
"""Index LibriSpeech subsets for on-the-fly mixing (train on LibriSpeech 960).

  python scripts/prepare_librispeech_index.py --librispeech-dir $S/LibriSpeech \
      --subsets train-clean-100 train-clean-360 train-other-500 --out data/librispeech_960_index.json
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsasr.librispeech import index_subset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-dir", required=True)
    p.add_argument("--subsets", nargs="+", default=["train-clean-100", "train-clean-360", "train-other-500"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    utts = {}
    for s in a.subsets:
        u = index_subset(a.librispeech_dir, s)
        print(f"{s}: {len(u)} utterances")
        utts.update(u)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(list(utts.values()), f)
    spk = Counter(u["speaker"] for u in utts.values())
    print(json.dumps({"utterances": len(utts), "speakers": len(spk),
                      "hours": round(sum(u["duration"] for u in utts.values()) / 3600, 1)}))


if __name__ == "__main__":
    main()
