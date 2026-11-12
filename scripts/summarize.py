#!/usr/bin/env python
"""Collect results into Markdown tables: every *.summary.json (WER runs), every
*.compare.json (output-change analysis) and context_cost_*.json.

  python scripts/summarize.py results/ > results/table.md
"""
import glob
import json
import os
import sys


def rows(root, pattern):
    for fn in sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True)):
        with open(fn) as f:
            yield fn, json.load(f)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    print("## Target-speaker WER\n")
    hdr = ["run", "set", "condition", "enrollment", "source", "dur (s)", "n", "TS-WER", "sub", "del", "ins",
           "sel. acc", "best-spk WER"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    latex = []
    for fn, s in rows(root, "*.summary.json"):
        name = os.path.basename(fn).replace(".summary.json", "")
        ts = s["ts"]
        cells = [name, os.path.splitext(os.path.basename(s.get("manifest", "")))[0], s.get("condition", ""),
                 s.get("enroll_mode", ""), s.get("enroll_field", "enrollment").replace("enrollment_", "") or "-",
                 s.get("enroll_dur", ""), s.get("n_scored", ts["n_utts"]), f"{ts['wer']:.2f}", f"{ts['sub_rate']:.2f}",
                 f"{ts['del_rate']:.2f}", f"{ts['ins_rate']:.2f}", f"{s['selection_acc']:.1f}",
                 f"{s['best_spk']['wer']:.2f}"]
        print("| " + " | ".join(map(str, cells)) + " |")
        latex.append(f"{name.replace('_', chr(92) + '_')} & {ts['wer']:.1f} & {ts['ins_rate']:.1f} & "
                     f"{s['selection_acc']:.1f} \\\\")

    comps = list(rows(root, "*.compare.json"))
    if comps:
        print("\n## Does the output change with the enrollment?\n")
        print("| run B vs run A | modes | n | identical % | output-change WER | follows A % | follows B % | switch % |")
        print("|---|---|---|---|---|---|---|---|")
        for fn, c in comps:
            print(f"| {os.path.basename(fn).replace('.compare.json', '')} | {c['mode_a']} vs {c['mode_b']} | "
                  f"{c['n_paired']} | {c['identical_rate']} | {c['output_change_wer']} | {c['follows_A']} | "
                  f"{c['follows_B']} | {c['switch_rate']} |")

    ctx = list(rows(root, "context_cost_*.json"))
    if ctx:
        print("\n## Cost of losing 5 s of context (long-form, fixed windows)\n")
        print("| model | window (s) | windows | WER | decode s |")
        print("|---|---|---|---|---|")
        for _, c in ctx:
            for w, r in c["windows"].items():
                print(f"| {c['model']} | {w} | {r['n_windows']} | {r['wer']:.2f} | {r['decode_sec']} |")
            if "cost" in c:
                print(f"| {c['model']} | cost | x{c['cost']['window_ratio']} windows | "
                      f"+{c['cost']['wer_abs_increase']} WER | |")

    print("\n% LaTeX rows: run & TS-WER & ins & sel. acc")
    print("\n".join(latex))


if __name__ == "__main__":
    main()
