import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsasr import ENC_HOP, SAMPLE_RATE  # noqa: E402
from tsasr.audio import (add_noise, apply_rir, compose_prefix, crop, rms, smooth_activity,  # noqa: E402
                         speech_activity, stno_from_activity)
from tsasr.text import closest_speaker, corpus_scores, word_errors  # noqa: E402


def tone(sec, on=None, sr=SAMPLE_RATE):
    t = np.arange(int(sec * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    if on is not None:
        m = np.zeros_like(x)
        for s, e in on:
            m[int(s * sr): int(e * sr)] = 1
        x = x * m
    return x


def test_compose_prefix_alignment():
    enr, mix = tone(2.013), tone(3.0)
    wav, n_enr, off = compose_prefix(enr, mix, gap_sec=0.5)
    assert len(wav) == off * ENC_HOP + len(mix)
    assert n_enr == len(enr) // ENC_HOP
    assert np.allclose(wav[off * ENC_HOP:], mix)
    assert np.all(wav[len(enr): off * ENC_HOP] == 0)
    wav2, n2, off2 = compose_prefix(None, mix)
    assert n2 == 0 and off2 == 0 and len(wav2) == len(mix)


def test_activity_and_stno():
    a = tone(4, [(0.0, 2.0)])
    b = tone(4, [(1.0, 3.0)])
    acts = [speech_activity(a), speech_activity(b)]
    assert len(acts[0]) == 4 * SAMPLE_RATE // ENC_HOP
    st = stno_from_activity(acts, 0)
    assert st.shape == (4, len(acts[0]))
    assert np.allclose(st.sum(0), 1.0)
    f = lambda sec: int(sec * 50)  # noqa: E731
    assert st[1, f(0.5)] == 1  # target only
    assert st[3, f(1.5)] == 1  # overlap
    assert st[2, f(2.5)] == 1  # non-target only
    assert st[0, f(3.5)] == 1  # silence


def test_smooth_activity():
    act = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], bool)
    out = smooth_activity(act, max_gap=2, min_len=3)
    assert out[:9].all()          # 2-frame pause closed
    assert not out[9:].any()      # isolated single frame dropped


def test_crop_and_noise_rir():
    x = tone(5)
    assert len(crop(x, 2.0)) == 2 * SAMPLE_RATE
    rng = np.random.default_rng(0)
    assert len(crop(x, 2.0, rng)) == 2 * SAMPLE_RATE
    assert len(crop(x, 9.0)) == len(x)
    n = np.random.default_rng(1).standard_normal(SAMPLE_RATE).astype(np.float32)
    y = add_noise(x, n, snr_db=10.0, rng=rng)
    snr = 10 * np.log10(rms(x) ** 2 / rms(y - x) ** 2)
    assert abs(snr - 10.0) < 0.5
    rir = np.zeros(800, np.float32)
    rir[40] = 1.0
    rir[400] = 0.3
    z = apply_rir(x, rir)
    assert len(z) == len(x) and abs(rms(z) - rms(x)) < 1e-3


def test_wer_helpers():
    e = word_errors("a b c d", "a x c d e")
    assert (e["sub"], e["del"], e["ins"], e["ref_words"]) == (1, 0, 1, 4)
    assert word_errors("a b", "")["del"] == 2
    s = corpus_scores([e, word_errors("a b", "a b")])
    assert s["wer"] == pytest.approx(100 * 2 / 6)
    assert closest_speaker("hello world", ["foo bar", "hello world"]) == 1


def test_epoch_sampler_shards_and_resume():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from train import EpochSampler

    n, world = 11, 3
    seen = []
    for r in range(world):
        s = EpochSampler(n, seed=0, rank=r, world=world)
        s.set_epoch(2)
        seen += list(s)
    assert set(seen) == set(range(n))
    s = EpochSampler(n, 0, 0, 1)
    s.set_epoch(1)
    full = list(s)
    s.set_epoch(1, skip=4)
    assert list(s) == full[4:] and len(s) == n - 4


def test_tokens_build():
    tr = pytest.importorskip("transformers")
    from tsasr.data import Tokens

    tok = tr.WhisperTokenizer.from_pretrained("openai/whisper-tiny")
    t = Tokens(tok)
    dec_in, labels = t.build("hello world")
    assert len(dec_in) == len(labels)
    assert dec_in[0] == t.sot and dec_in[1:4] == t.prompt
    assert labels[:3] == [-100] * 3 and labels[-1] == t.eot
    assert labels[3:-1] == dec_in[4:]
    assert tok.decode(labels[3:-1]).strip() == "hello world"
