"""Waveform utilities: loading, cropping, enrollment-prefix composition,
frame-level speaker activity, noise and reverberation."""
from __future__ import annotations

from math import gcd
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from . import ENC_HOP, SAMPLE_RATE

EPS = 1e-8


def load_audio(path: str, sr: int = SAMPLE_RATE, channel: int = 0,
               start: Optional[float] = None, end: Optional[float] = None) -> np.ndarray:
    """Mono float32 at `sr`. start/end (seconds) read only that segment."""
    if start is None and end is None:
        x, fs = sf.read(path, dtype="float32", always_2d=True)
    else:
        fs = sf.info(path).samplerate
        a = int(round((start or 0.0) * fs))
        b = None if end is None else int(round(end * fs))
        x, fs = sf.read(path, dtype="float32", always_2d=True, start=a, stop=b)
    x = x[:, min(channel, x.shape[1] - 1)]
    if fs != sr:
        g = gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g).astype(np.float32)
    return x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS))


def rms_normalize(x: np.ndarray, target_dbfs: float = -25.0) -> np.ndarray:
    """Scale to a fixed RMS level so that enrollment and mixture enter the shared
    30 s window at the same level (level differences between separately captured
    clips are not a usable cue). A peak limiter avoids clipping."""
    if len(x) == 0:
        return x
    g = 10 ** (target_dbfs / 20) / rms(x)
    y = x * g
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def frame_energy_db(x: np.ndarray, hop: int = ENC_HOP) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    fr = x[: n * hop].reshape(n, hop).astype(np.float64)
    return (10 * np.log10(np.mean(fr**2, axis=1) + 1e-10)).astype(np.float32)


def speech_activity(x: np.ndarray, rel_db: float = 30.0, abs_floor_db: float = -60.0,
                    hop: int = ENC_HOP) -> np.ndarray:
    """Boolean activity per 20 ms frame (Whisper encoder rate). A frame is active
    if its energy is within rel_db of the loudest frame and above abs_floor_db."""
    e = frame_energy_db(x, hop)
    if len(e) == 0:
        return e.astype(bool)
    return (e > e.max() - rel_db) & (e > abs_floor_db)


def smooth_activity(act: np.ndarray, max_gap: int = 10, min_len: int = 5) -> np.ndarray:
    """Segment-like activity: close pauses shorter than max_gap frames, then drop
    segments shorter than min_len frames (defaults: 200 ms and 100 ms at 50 Hz)."""
    a = act.astype(bool).copy()
    idx = np.nonzero(a)[0]
    for s, e in zip(idx[:-1], idx[1:]):
        if 1 < e - s <= max_gap + 1:
            a[s:e] = True
    out = np.zeros_like(a)
    i = 0
    while i < len(a):
        if a[i]:
            j = i
            while j < len(a) and a[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def trim_silence(x: np.ndarray, rel_db: float = 35.0) -> np.ndarray:
    act = speech_activity(x, rel_db=rel_db)
    if not act.any():
        return x
    idx = np.nonzero(act)[0]
    s, e = idx[0] * ENC_HOP, (idx[-1] + 1) * ENC_HOP
    return x[s:e]


def crop(x: np.ndarray, dur_sec: Optional[float], rng: Optional[np.random.Generator] = None,
         sr: int = SAMPLE_RATE) -> np.ndarray:
    """Crop to dur_sec. Deterministic (from the start) when rng is None, random
    offset otherwise. Shorter inputs are returned unchanged."""
    if dur_sec is None:
        return x
    n = int(round(dur_sec * sr))
    if len(x) <= n:
        return x
    off = 0 if rng is None else int(rng.integers(0, len(x) - n + 1))
    return x[off: off + n]


def _pad_to_frame(n: int, hop: int = ENC_HOP) -> int:
    return n + (-n) % hop


def compose_input(enroll: Optional[np.ndarray], mixture: np.ndarray, gap_sec: float = 0.5,
                  position: str = "prefix", lead_sec: float = 0.0, sr: int = SAMPLE_RATE,
                  hop: int = ENC_HOP):
    """Build the encoder input and the encoder-frame spans of its parts.

    prefix: [lead | enrollment | separator | mixture]
    suffix: [lead | mixture | separator | enrollment]
    Every part after the first starts on an encoder frame boundary (20 ms); the
    separator (silence of gap_sec) absorbs the alignment padding.

    Returns (waveform, spans) with spans = {"enroll": (s, e), "sep": (s, e), "mix": (s, e)}
    in encoder frames. Without an enrollment the input is [lead | mixture]."""
    lead = _pad_to_frame(int(round(lead_sec * sr)), hop) if lead_sec > 0 else 0
    z = lambda n: np.zeros(n, dtype=np.float32)  # noqa: E731
    if enroll is None or len(enroll) == 0:
        wav = np.concatenate([z(lead), mixture]).astype(np.float32)
        m0 = lead // hop
        return wav, {"enroll": (0, 0), "sep": (0, 0), "mix": (m0, m0 + len(mixture) // hop)}
    gap = int(round(gap_sec * sr))
    first = enroll if position == "prefix" else mixture
    second = mixture if position == "prefix" else enroll
    gap = _pad_to_frame(len(first) + gap, hop) - len(first)
    wav = np.concatenate([z(lead), first, z(gap), second]).astype(np.float32)
    f0 = lead // hop
    first_full = f0 + len(first) // hop          # frames fully inside the first part
    s0 = (lead + len(first) + gap) // hop        # start of the second part
    first_span, second_span = (f0, first_full), (s0, s0 + len(second) // hop)
    spans = {"sep": (first_full, s0)}
    if position == "prefix":
        spans.update(enroll=first_span, mix=second_span)
    else:
        spans.update(mix=(f0, f0 + len(first) // hop), enroll=second_span)
    return wav, spans


def compose_prefix(enroll: Optional[np.ndarray], mixture: np.ndarray, gap_sec: float = 0.5,
                   sr: int = SAMPLE_RATE, hop: int = ENC_HOP) -> Tuple[np.ndarray, int, int]:
    """[enrollment | silence gap | mixture]. Returns (waveform, n_enroll_frames, mix_offset_frames)."""
    wav, sp = compose_input(enroll, mixture, gap_sec, "prefix", 0.0, sr, hop)
    return wav, sp["enroll"][1] - sp["enroll"][0], sp["mix"][0]


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add a noise segment (tiled or randomly cropped to len(x)) at snr_db."""
    if len(noise) < len(x):
        reps = int(np.ceil(len(x) / max(len(noise), 1)))
        noise = np.tile(noise, reps)
    off = 0 if rng is None else int(rng.integers(0, len(noise) - len(x) + 1))
    n = noise[off: off + len(x)].astype(np.float32)
    ps, pn = rms(x) ** 2, rms(n) ** 2
    g = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return (x + g * n).astype(np.float32)


def apply_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with an RIR, aligned to the direct path, same length as x, same RMS."""
    rir = rir.astype(np.float64)
    rir = rir / (np.max(np.abs(rir)) + EPS)
    d = int(np.argmax(np.abs(rir)))
    y = fftconvolve(x.astype(np.float64), rir)[d: d + len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    y = y * (rms(x) / (rms(y) + EPS))
    return y.astype(np.float32)


def stno_from_activity(acts: List[np.ndarray], target: int) -> np.ndarray:
    """Silence/target/non-target/overlap mask (4, T) from per-speaker 0/1 activity,
    matching DiCoW's get_stno_mask (pipeline.py of BUTSpeechFIT/DiCoW)."""
    d = np.stack([a.astype(np.float32) for a in acts], 0)
    others = np.ones(len(acts), dtype=bool)
    others[target] = False
    sil = np.prod(1 - d, axis=0)
    no_other = np.prod(1 - d[others], axis=0)
    tgt = d[target] * no_other
    non_tgt = (1 - d[target]) * (1 - no_other)
    ovl = d[target] - tgt
    return np.stack([sil, tgt, non_tgt, ovl], 0)
