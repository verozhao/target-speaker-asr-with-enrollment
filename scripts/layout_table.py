#!/usr/bin/env python
"""Results matrix of the enrollment-layout experiments.

Rows: experiment (padding, anchor, fixed5). Columns: test enrollment length and the
controls. Cell: TS-WER % (selection accuracy %). One table per test set.

  python scripts/layout_table.py results/ > results/layout_table.md
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

EXPERIMENTS = ("padding", "anchor", "fixed5")
COLUMNS = ["1s", "2s", "5s", "10s", "silence", "none", "other"]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    cells = defaultdict(dict)  # (test set, experiment) -> column -> text
    pat = re.compile(r"^layout_(\w+?)-best_(.+?)_clean_(\w+?)_enr([\d.]+)s_fit10$")
    for fn in glob.glob(os.path.join(root, "layout_*.summary.json")):
        m = pat.match(os.path.basename(fn).replace(".summary.json", ""))
        if not m:
            continue
        exp, test_set, mode, dur = m.groups()
        with open(fn) as f:
            s = json.load(f)
        col = f"{float(dur):g}s" if mode == "target" else mode
        cells[(test_set, exp)][col] = f"{s['ts']['wer']:.1f} ({s['selection_acc']:.0f})"
    comps = {}
    for fn in glob.glob(os.path.join(root, "layout_*.compare.json")):
        with open(fn) as f:
            c = json.load(f)
        m = re.match(r"^layout_(\w+?)-best_(.+?)_clean_", os.path.basename(c["run_a"]))
        if m:
            comps[(m.group(2), m.group(1), c["mode_b"])] = c
    for test_set in sorted({t for t, _ in cells}):
        print(f"## {test_set}: TS-WER % (selection accuracy %), identical test mixtures\n")
        print("| experiment | " + " | ".join(COLUMNS) + " |")
        print("|" + "---|" * (len(COLUMNS) + 1))
        for exp in EXPERIMENTS:
            row = cells.get((test_set, exp))
            if row:
                print(f"| {exp} | " + " | ".join(row.get(c, "-") for c in COLUMNS) + " |")
        rows = [(exp, mode, comps[(test_set, exp, mode)]) for exp in EXPERIMENTS for mode in ("silence", "none", "other")
                if (test_set, exp, mode) in comps]
        if rows:
            print("\nOutput change against the 5 s target enrollment (identical % / output-change WER):\n")
            print("| experiment | silence | none | other |")
            print("|---|---|---|---|")
            for exp in EXPERIMENTS:
                vals = [comps.get((test_set, exp, mode)) for mode in ("silence", "none", "other")]
                if any(vals):
                    print(f"| {exp} | " + " | ".join(
                        f"{v['identical_rate']:.0f} / {v['output_change_wer']:.1f}" if v else "-" for v in vals) + " |")
        print()


if __name__ == "__main__":
    main()
