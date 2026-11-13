#!/usr/bin/env python
"""Does the output change when the enrollment changes?

Pairs two evaluation runs of the SAME model on the SAME mixtures (by example id),
e.g. --enroll-mode target vs interferer / other / silence / none, and reports:
  identical_rate     % of examples whose normalized output is unchanged
  output_change_wer  WER between the two outputs (run A as reference), 0 = unchanged
  follows_A / B      % of outputs closest to the speaker that run enrolled
  switch_rate        % of examples where A follows its enrolled speaker AND B follows a
                     different enrolled speaker (target vs interferer swap)

  python scripts/compare_outputs.py results/X_target.hyps.jsonl results/X_interferer.hyps.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsasr.text import corpus_scores, word_errors  # noqa: E402


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in (json.loads(ln) for ln in f if ln.strip())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--out", default=None, help="json path (default: next to run_b)")
    a = p.parse_args()
    A, B = load(a.run_a), load(a.run_b)
    ids = sorted(set(A) & set(B))
    if not ids:
        sys.exit("no common example ids")
    same = fol_a = fol_b = sw = 0
    errs = []
    for i in ids:
        ra, rb = A[i], B[i]
        same += int(ra["hyp_norm"] == rb["hyp_norm"])
        errs.append(word_errors(ra["hyp_norm"], rb["hyp_norm"]))
        fa = bool(ra["follows_enrolled"]) if "follows_enrolled" in ra else ra["closest_speaker"] == ra["enrolled_index"]
        fb = bool(rb["follows_enrolled"]) if "follows_enrolled" in rb else rb["closest_speaker"] == rb["enrolled_index"]
        fol_a += fa
        fol_b += fb
        sw += int(fa and fb and ra["enrolled_index"] != rb["enrolled_index"])
    n = len(ids)
    res = {
        "run_a": a.run_a, "run_b": a.run_b, "n_paired": n,
        "mode_a": A[ids[0]]["enroll_mode"], "mode_b": B[ids[0]]["enroll_mode"],
        "identical_rate": round(100 * same / n, 2),
        "output_change_wer": round(corpus_scores(errs)["wer"], 2),
        "follows_A": round(100 * fol_a / n, 2), "follows_B": round(100 * fol_b / n, 2),
        "switch_rate": round(100 * sw / n, 2),
    }
    out = a.out or a.run_b.replace(".hyps.jsonl", "") + "__vs__" + \
        os.path.basename(a.run_a).replace(".hyps.jsonl", "") + ".compare.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
