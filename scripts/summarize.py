#!/usr/bin/env python
"""Collect *.summary.json files into one Markdown table (and LaTeX rows).

  python scripts/summarize.py results/ > results/table.md
"""
import glob
import json
import os
import sys


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    rows = []
    for fn in sorted(glob.glob(os.path.join(root, "**", "*.summary.json"), recursive=True)):
        with open(fn) as f:
            s = json.load(f)
        rows.append({
            "file": os.path.basename(fn).replace(".summary.json", ""),
            "system": s.get("system", ""),
            "set": os.path.splitext(os.path.basename(s.get("manifest", "")))[0],
            "cond": s.get("condition", ""),
            "enroll": s.get("enroll_mode", ""),
            "dur": s.get("enroll_dur", ""),
            "n": s.get("n_scored", s["ts"]["n_utts"]),
            "wer": s["ts"]["wer"], "sub": s["ts"]["sub_rate"], "del": s["ts"]["del_rate"],
            "ins": s["ts"]["ins_rate"], "sel": s["selection_acc"], "best": s["best_spk"]["wer"],
        })
    hdr = "| run | set | condition | enrollment | dur (s) | n | TS-WER | sub | del | ins | sel. acc | best-spk WER |"
    print(hdr)
    print("|" + "|".join(["---"] * 12) + "|")
    for r in rows:
        print(f"| {r['file']} | {r['set']} | {r['cond']} | {r['enroll']} | {r['dur']} | {r['n']} | "
              f"{r['wer']:.2f} | {r['sub']:.2f} | {r['del']:.2f} | {r['ins']:.2f} | {r['sel']:.1f} | {r['best']:.2f} |")
    print("\n% LaTeX rows: run & TS-WER & ins & sel. acc")
    for r in rows:
        name = r["file"].replace("_", r"\_")
        print(f"{name} & {r['wer']:.1f} & {r['ins']:.1f} & {r['sel']:.1f} \\\\")


if __name__ == "__main__":
    main()
