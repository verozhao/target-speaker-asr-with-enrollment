#!/usr/bin/env python
"""macOS only: synthesize a tiny LibriSpeech / LibriMix / WHAM! / RIRS_NOISES look-alike
with the `say` voices as speakers, in the exact directory layouts the real
pipeline expects. Used only to smoke-test the code end to end on a laptop.

  python tools/make_toy_librimix.py /tmp/toy
"""
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
SPLITS = {  # LibriSpeech subset -> {speaker id: voice}
    "train-clean-100": {"1001": "Daniel", "1002": "Karen", "1003": "Moira", "1004": "Fred"},
    "dev-clean": {"2001": "Kathy", "2002": "Ralph", "2003": "Rishi"},
    "test-clean": {"3001": "Albert", "3002": "Junior", "3003": "Flo (English (US))", "3004": "Reed (English (US))"},
}
MIXSPLIT = {"train-clean-100": "train-100", "dev-clean": "dev", "test-clean": "test"}
WORDS = ("the river ran past an old mill where seven children played with a red ball while their "
         "mother baked bread and the dog slept under a wooden table near the open window in "
         "summer light every morning brings new letters from distant cities across green hills").split()


def tts(text, voice, rng):
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = os.path.join(d, "a.aiff"), os.path.join(d, "a.wav")
        subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", aiff, wav], check=True)
        x, sr = sf.read(wav, dtype="float32")
    assert sr == SR
    lead = np.zeros(int(rng.uniform(0.1, 0.4) * SR), np.float32)
    return np.concatenate([lead, x, lead])


def main():
    root = os.path.abspath(sys.argv[1])
    rng = np.random.default_rng(0)
    utts = {}
    for subset, spks in SPLITS.items():
        for spk, voice in spks.items():
            ch = "100"
            d = os.path.join(root, "LibriSpeech", subset, spk, ch)
            os.makedirs(d, exist_ok=True)
            lines = []
            n_utt = 6 if subset == "train-clean-100" else 4
            for u in range(n_utt):
                uid = f"{spk}-{ch}-{u:04d}"
                n_w = int(rng.integers(5, 12)) if u else 22  # utterance 0 is long (enrollment)
                text = " ".join(rng.choice(WORDS, n_w)).upper()
                x = tts(text.lower(), voice, rng)
                sf.write(os.path.join(d, uid + ".flac"), x, SR)
                lines.append(f"{uid} {text}")
                utts[uid] = (subset, spk, x)
            with open(os.path.join(d, f"{spk}-{ch}.trans.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")

    for n_src in (2, 3):
        base = os.path.join(root, f"Libri{n_src}Mix", "wav16k", "max")
        os.makedirs(os.path.join(base, "metadata"), exist_ok=True)
        for subset, spks in SPLITS.items():
            if n_src == 3 and subset != "test-clean":
                continue
            split = MIXSPLIT[subset]
            subs = [f"s{k + 1}" for k in range(n_src)] + ["mix_clean"]
            for s in subs:
                os.makedirs(os.path.join(base, split, s), exist_ok=True)
            ids = sorted(u for u in utts if utts[u][0] == subset)
            rows = []
            # use only utterances 1.. as sources so utterance 0 stays "unused" for enrollment
            pool = [u for u in ids if not u.endswith("-0000")]
            n_mix = 12 if subset == "train-clean-100" else 6
            for m in range(n_mix):
                while True:
                    pick = list(rng.choice(pool, n_src, replace=False))
                    if len({utts[u][1] for u in pick}) == n_src:
                        break
                mid = "_".join(pick)
                srcs = [utts[u][2] * rng.uniform(0.4, 0.9) for u in pick]
                L = max(len(s) for s in srcs)
                srcs = [np.pad(s, (0, L - len(s))) for s in srcs]
                mix = np.sum(srcs, 0)
                peak = np.abs(mix).max()
                if peak > 0.99:
                    srcs = [s * 0.99 / peak for s in srcs]
                    mix = mix * 0.99 / peak
                row = {"mixture_ID": mid}
                p = os.path.join(base, split, "mix_clean", mid + ".wav")
                sf.write(p, mix, SR)
                row["mixture_path"] = p
                for k, s in enumerate(srcs):
                    p = os.path.join(base, split, f"s{k + 1}", mid + ".wav")
                    sf.write(p, s, SR)
                    row[f"source_{k + 1}_path"] = p
                row["length"] = L
                rows.append(row)
            with open(os.path.join(base, "metadata", f"mixture_{split}_mix_clean.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    for sub in ("tr", "cv", "tt"):
        d = os.path.join(root, "wham_noise", sub)
        os.makedirs(d, exist_ok=True)
        for i in range(4):
            n = rng.standard_normal(SR * 8).astype(np.float32)
            n = np.convolve(n, np.ones(8) / 8, mode="same") * 0.1  # low-passed noise
            sf.write(os.path.join(d, f"{sub}{i:03d}.wav"), np.stack([n, n], 1), SR)  # stereo like WHAM!
    for size in ("smallroom", "mediumroom"):
        for r in range(1, 16):
            d = os.path.join(root, "RIRS_NOISES", "simulated_rirs", size, f"Room{r:03d}")
            os.makedirs(d, exist_ok=True)
            for j in range(1, 3):
                t60 = rng.uniform(0.2, 0.8)
                t = np.arange(int(SR * t60)) / SR
                rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / t60)
                rir[: int(0.002 * SR)] = 0
                rir[int(0.002 * SR)] = 1.0
                sf.write(os.path.join(d, f"Room{r:03d}-{j:05d}.wav"), rir.astype(np.float32), SR)
    print("toy data at", root)


if __name__ == "__main__":
    main()
