#!/usr/bin/env python
"""DiCoW baseline (BUT-FIT/DiCoW_v3_2, Whisper large-v3-turbo) with ORACLE diarization.

DiCoW conditions on a silence/target/non-target/overlap (STNO) mask instead of an
enrollment. Oracle masks come from the clean sources on LibriMix (frame energy at
50 Hz, pauses < 200 ms closed) and from the word timings on AMI (the manifest's
"activity" field), so this baseline gets perfect diarization, which none of our
systems use. Note: the v3.2 model card lists Libri2Mix and AMI among its training
data, so neither test set is out of domain for it. The paper version is DiCoW_v1,
which needs the code from the v1 branch of the DiCoW GitHub repo.

The STNO construction copies get_stno_mask from BUTSpeechFIT/DiCoW pipeline.py
(class order 0 silence, 1 target, 2 non-target, 3 overlap). DiCoW's remote code
is pinned to a fixed revision. transformers==4.55.0 is what the DiCoW repo pins.

  python scripts/eval_dicow.py --manifest data/libri2mix_test.jsonl --out-dir results
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer

from tsasr import SAMPLE_RATE, WHISPER_ENC_FRAMES, WHISPER_WINDOW_SAMPLES
from tsasr.audio import load_audio, smooth_activity, speech_activity, stno_from_activity
from tsasr.augment import EvalPerturber
from tsasr.data import fits, read_manifest
from tsasr.evaluation import score, subset, write_jsonl

DICOW_REVISION = "00a09c622bf58c34fdc76c3b5676f4da706f2a8e"
_TS = re.compile(r"<\|[^|]*\|>")


def oracle_stno(sources, target):
    acts = [smooth_activity(speech_activity(s)) for s in sources]
    n = min(min(len(a) for a in acts), WHISPER_ENC_FRAMES)
    full = []
    for a in acts:
        f = np.zeros(WHISPER_ENC_FRAMES, dtype=bool)
        f[:n] = a[:n]
        full.append(f)
    return stno_from_activity(full, target)  # frames after the mixture end -> silence


def stno_from_segments(activity, target):
    """activity: per speaker list of (start, end) seconds relative to the window."""
    full = []
    for segs in activity:
        f = np.zeros(WHISPER_ENC_FRAMES, dtype=bool)
        for s, e in segs:
            f[max(int(round(s * 50)), 0): min(int(round(e * 50)), WHISPER_ENC_FRAMES)] = True
        full.append(f)
    if len(full) == 1:
        full.append(np.zeros(WHISPER_ENC_FRAMES, dtype=bool))
    return stno_from_activity(full, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="BUT-FIT/DiCoW_v3_2")
    p.add_argument("--revision", default=None, help="default: pinned commit for v3_2, main otherwise")
    p.add_argument("--fit-dur", type=float, default=5.0, help="same subset as evaluate.py")
    p.add_argument("--gap", type=float, default=0.5)
    p.add_argument("--condition", default="clean", choices=["clean", "mix_perturbed"])
    p.add_argument("--noise-list", default=None)
    p.add_argument("--rir-list", default=None)
    p.add_argument("--snr", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--language", default=None, help="e.g. en, passed to generate if set")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    if a.revision is None:
        a.revision = DICOW_REVISION if a.model == "BUT-FIT/DiCoW_v3_2" else "main"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (a.fp16 and device.type == "cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(a.model, revision=a.revision, trust_remote_code=True,
                                                      torch_dtype=dtype).to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model.set_tokenizer(tok)

    all_ex = read_manifest(a.manifest)
    ex = subset([e for e in all_ex if fits(e, a.fit_dur, a.gap)], a.max_examples, seed=0)
    pert = EvalPerturber(a.noise_list, a.rir_list, a.condition, a.snr) if a.condition != "clean" else None

    gen_kw = {"num_beams": a.num_beams, "return_timestamps": True}
    if a.language:
        gen_kw["language"] = a.language
    hyps, t0 = {}, time.time()
    for i in tqdm(range(0, len(ex), a.batch_size), desc="dicow"):
        chunk = ex[i: i + a.batch_size]
        feats, masks = [], []
        for e in chunk:
            start, end = e.get("mixture_span") or (None, None)
            mix = load_audio(e["mixture"], start=start, end=end)
            if pert is not None:
                _, mix = pert(None, mix, e["id"])
            mix = mix[:WHISPER_WINDOW_SAMPLES]
            feats.append(fe(mix, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features[0])
            if e.get("sources"):
                srcs = [load_audio(s) for s in e["sources"]]
                masks.append(torch.from_numpy(oracle_stno(srcs, e["target_index"])))
            else:  # AMI: activity[0] is the target
                masks.append(torch.from_numpy(stno_from_segments(e["activity"], 0)))
        feats = torch.stack(feats).to(device, dtype)
        stno = torch.stack(masks).to(device, dtype)
        assert feats.shape[-1] == 2 * stno.shape[-1] == 3000, (feats.shape, stno.shape)
        with torch.no_grad():
            seqs = model.generate(input_features=feats, stno_mask=stno, **gen_kw)
        if not torch.is_tensor(seqs):
            seqs = seqs["sequences"] if isinstance(seqs, dict) else seqs.sequences
        for e, t in zip(chunk, tok.batch_decode(seqs, skip_special_tokens=True)):
            hyps[e["id"]] = " ".join(_TS.sub(" ", t).split())

    recs, summ = score(ex, hyps, "oracle_diarization")
    tag = a.tag or f"dicow_{os.path.splitext(os.path.basename(a.manifest))[0]}_{a.condition}"
    summ.update({"system": f"{a.model}@{a.revision[:8]} (oracle diarization)", "manifest": a.manifest,
                 "condition": a.condition, "fit_dur": a.fit_dur, "n_manifest": len(all_ex),
                 "n_scored": len(recs), "num_beams": a.num_beams, "decode_sec": round(time.time() - t0, 1)})
    os.makedirs(a.out_dir, exist_ok=True)
    write_jsonl(os.path.join(a.out_dir, tag + ".hyps.jsonl"), recs)
    with open(os.path.join(a.out_dir, tag + ".summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps({"tag": tag, "ts_wer": round(summ["ts"]["wer"], 2), "n": len(recs)}))


if __name__ == "__main__":
    main()
