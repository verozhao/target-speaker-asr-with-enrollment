#!/usr/bin/env python
"""Noise and RIR file lists with disjoint train/dev/test sets.

Noise: WHAM! tr -> train, cv -> dev, tt -> test (disjoint noise files by design).
RIRs:  OpenSLR 28 simulated RIRs, split at the ROOM level (size/RoomNNN), 80/10/10,
       by a stable hash so the split never changes between runs.

  python scripts/prepare_aug_lists.py --wham-dir $S/wham_noise --rirs-dir $S/RIRS_NOISES --out-dir data/aug
"""
import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict


def room_split(room: str, dev_pct: int, test_pct: int) -> str:
    b = int(hashlib.sha1(room.encode()).hexdigest()[:8], 16) % 100
    return "test" if b < test_pct else "dev" if b < test_pct + dev_pct else "train"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wham-dir", required=True)
    p.add_argument("--rirs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dev-pct", type=int, default=10)
    p.add_argument("--test-pct", type=int, default=10)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    stats = {}

    for split, sub in (("train", "tr"), ("dev", "cv"), ("test", "tt")):
        files = sorted(glob.glob(os.path.join(a.wham_dir, sub, "*.wav")))
        if not files:
            raise SystemExit(f"no WHAM! files in {os.path.join(a.wham_dir, sub)}")
        with open(os.path.join(a.out_dir, f"noise_{split}.txt"), "w") as f:
            f.write("\n".join(os.path.abspath(x) for x in files) + "\n")
        stats[f"noise_{split}"] = len(files)

    rirs = sorted(glob.glob(os.path.join(a.rirs_dir, "simulated_rirs", "*room", "Room*", "*.wav")))
    if not rirs:
        raise SystemExit(f"no simulated RIRs under {a.rirs_dir}/simulated_rirs")
    by_split, rooms = defaultdict(list), defaultdict(set)
    for r in rirs:
        size = os.path.basename(os.path.dirname(os.path.dirname(r)))
        room = f"{size}/{os.path.basename(os.path.dirname(r))}"
        s = room_split(room, a.dev_pct, a.test_pct)
        by_split[s].append(f"{room}\t{os.path.abspath(r)}")
        rooms[s].add(room)
    for s in ("train", "dev", "test"):
        with open(os.path.join(a.out_dir, f"rir_{s}.txt"), "w") as f:
            f.write("\n".join(by_split[s]) + "\n")
        stats[f"rir_{s}"] = {"files": len(by_split[s]), "rooms": len(rooms[s])}
    assert not (rooms["train"] & rooms["test"]) and not (rooms["train"] & rooms["dev"])
    with open(os.path.join(a.out_dir, "aug_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
