#!/usr/bin/env python
"""Build an AMI target-speaker manifest (single distant microphone, Array1-01).

For every speaker of every session, the speaker's words are grouped into windows of
at most --max-len seconds (default 24.5, so 5 s enrollment + 0.5 s separator fit in
30 s). The window audio is the distant-microphone signal, so it contains everyone
who talks inside it. Reference = the target's words inside the window.

Enrollment (all from the same microphone, nobody else talking):
  enrollment                 same session, outside the window
  enrollment_other_session   another session of the same person (global speaker id)
  interferer_enrollment      the most active other speaker of the window (at least
                             --min-words words inside it), same session
  other_enrollment           a person who is not in this session

  python scripts/prepare_ami.py --annotations $S/ami/annotations --audio-dir $S/ami/audio \
      --split test --out data/ami_test.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsasr.ami import (AMI_DEV, AMI_TEST, _h, merge, pick_region, read_speaker_map, read_words,  # noqa: E402
                       solo_regions, target_windows, text_of, words_in)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True, help="unzipped ami_public_manual_1.6.2")
    p.add_argument("--audio-dir", required=True, help="holds <session>.Array1-01.wav")
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument("--sessions", nargs="*", default=None, help="override the session list")
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=float, default=24.5)
    p.add_argument("--enroll-len", type=float, default=5.0)
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--min-enroll", type=float, default=2.0, help="shortest usable solo stretch")
    a = p.parse_args()

    sessions = a.sessions or (AMI_TEST if a.split == "test" else AMI_DEV)
    audio = {s: os.path.abspath(os.path.join(a.audio_dir, f"{s}.Array1-01.wav")) for s in sessions}
    missing = [s for s, f in audio.items() if not os.path.exists(f)]
    if missing:
        print(f"skipping sessions without audio: {missing}")
    sessions = [s for s in sessions if s not in missing]
    smap = read_speaker_map(a.annotations)
    words = {s: read_words(a.annotations, s, smap) for s in sessions}
    solos = {s: {spk: solo_regions(words[s], spk, a.min_enroll) for spk in words[s]} for s in sessions}
    where = {}  # global speaker id -> sessions
    for s in sessions:
        for spk in words[s]:
            where.setdefault(spk, []).append(s)

    def enrollment(sess, spk, key, avoid=None):
        r = pick_region(solos[sess].get(spk, []), a.enroll_len, key, avoid)
        if r is None:
            return None
        return {"path": audio[sess], "start": r[0], "end": r[1], "duration": round(r[1] - r[0], 3),
                "session": sess, "speaker": spk}

    out, stats = [], Counter()
    for sess in sessions:
        ws = words[sess]
        for spk, tw in ws.items():
            for (st, en, i, j) in target_windows(tw, a.max_len, min_words=a.min_words):
                stats["windows"] += 1
                key = f"{sess}_{st}_{en}__{spk}"
                enr = enrollment(sess, spk, key, (st, en))
                if enr is None:
                    stats["skipped_no_same_session_enrollment"] += 1
                    continue
                # everyone with at least one word fully inside the window, target first
                inside = {o: words_in(ow, st, en) for o, ow in ws.items() if o != spk}
                others = sorted([o for o in inside if inside[o]], key=lambda o: -len(inside[o]))
                speakers = [spk] + others
                texts = [text_of(tw[i: j + 1])] + [text_of(inside[o]) for o in others]
                # activity of everyone overlapping the window (oracle diarization for DiCoW)
                act_spk = [spk] + [o for o, ow in ws.items() if o != spk
                                   and any(w[0] < en and w[1] > st for w in ow)]
                activity = [[(round(max(s0, st) - st, 3), round(min(e0, en) - st, 3))
                             for s0, e0 in merge([(w[0], w[1]) for w in ws[o] if w[0] < en and w[1] > st], 0.3)]
                            for o in act_spk]
                tgt_time = sum(e - s for s, e in activity[0])
                ovl = sum(max(0.0, min(e, e2) - max(s, s2)) for s, e in activity[0]
                          for oth in activity[1:] for s2, e2 in merge(oth, 0.0)) if len(activity) > 1 else 0.0
                rec = {
                    "id": key, "mixture_id": f"{sess}_{st}_{en}", "split": f"ami_{a.split}", "session": sess,
                    "n_src": len(speakers), "duration": round(en - st, 3), "mixture": audio[sess],
                    "mixture_span": [st, en], "sources": [], "speakers": speakers, "texts": texts,
                    "target_index": 0, "enrollment": enr, "activity": activity, "activity_speakers": act_spk,
                    "overlap_ratio": round(min(ovl / max(tgt_time, 1e-6), 1.0), 3),
                }
                other_sess = [s for s in where.get(spk, []) if s != sess]
                if other_sess:
                    s2 = sorted(other_sess, key=lambda s: _h(key, s))[0]
                    rec["enrollment_other_session"] = enrollment(s2, spk, key + "|xs")
                # swap test only when the interferer says enough inside the window to be scored
                if others and len(inside[others[0]]) >= a.min_words:
                    rec["interferer_enrollment"] = enrollment(sess, others[0], key + "|int", (st, en))
                    if rec["interferer_enrollment"] is not None:
                        rec["interferer_index"] = 1
                absent = sorted({(s, o) for s in sessions if s != sess for o in words[s]
                                 if o not in ws and solos[s].get(o)}, key=lambda t: _h(key, *t))
                if absent:
                    rec["other_enrollment"] = enrollment(absent[0][0], absent[0][1], key + "|other")
                # same- and other-session enrollment both at least --enroll-len long: compare
                # the two conditions on identical windows with full-length enrollments
                xs = rec.get("enrollment_other_session")
                rec["enroll_full_both"] = bool(xs) and min(enr["duration"], xs["duration"]) >= a.enroll_len
                stats["enroll_full_both"] += int(rec["enroll_full_both"])
                stats["examples"] += 1
                stats["with_other_session"] += int(rec.get("enrollment_other_session") is not None)
                stats["with_interferer"] += int(rec.get("interferer_enrollment") is not None)
                stats["multi_speaker"] += int(len(act_spk) > 1)
                out.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    summ = {"sessions": sessions, **stats,
            "mean_window_sec": round(sum(r["duration"] for r in out) / max(len(out), 1), 2),
            "mean_overlap_ratio": round(sum(r["overlap_ratio"] for r in out) / max(len(out), 1), 3)}
    with open(os.path.splitext(a.out)[0] + ".stats.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
