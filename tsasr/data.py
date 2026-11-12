"""Datasets for enrollment-conditioned target-speaker ASR.

Encoder input (default): [enrollment ~5 s | separator 0.5 s | mixture <= 24.5 s] within
Whisper's 30 s window, label = the target speaker's transcript.

ManifestDataset reads pre-built examples (Libri2Mix/Libri3Mix, AMI windows). One
JSON object per line and per (mixture, target speaker):
  id                "<mixture_id>__<target>"
  mixture_id, split, n_src, duration (mixture seconds)
  mixture           wav path; optional mixture_span [start, end] in seconds
  sources           per-speaker source wavs (empty for real audio)
  source_utts       LibriSpeech utterance ids of the sources (LibriMix only)
  speakers, texts   speaker ids and transcripts of everyone in the mixture
  target_index      index of the target in speakers/texts
  enrollment        {path, duration[, start, end, utt_id]} of the target, not taken from this mixture
  other_enrollment  a speaker absent from the mixture
  optional: enrollment_other_session, interferer_enrollment, activity

OnTheFlyMixDataset mixes LibriSpeech utterances on the fly (LibriMix-style, any
number of speakers, full or partial overlap) with a fresh enrollment per example.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from . import ENC_HOP, SAMPLE_RATE, WHISPER_ENC_FRAMES, WHISPER_WINDOW_SAMPLES, WHISPER_WINDOW_SEC
from .audio import compose_input, crop, load_audio, rms, rms_normalize, speech_activity, trim_silence
from .text import training_text

# frame label codes (encoder rate, 50 Hz)
IGNORE, SIL, TGT, OTHER, OVL = -1, 0, 1, 2, 3
ENROLL_MODES = ("target", "interferer", "other", "silence", "none")
SLACK = ENC_HOP / SAMPLE_RATE  # frame-alignment slack per part (20 ms)


def read_manifest(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def max_mixture_sec(enroll_sec: float, gap_sec: float, lead_sec: float = 0.0) -> float:
    """Longest mixture that fits in 30 s next to the enrollment, separator and lead."""
    slack = SLACK * (2 if lead_sec > 0 else 1)
    return WHISPER_WINDOW_SEC - enroll_sec - gap_sec - lead_sec - slack


def fits(ex: dict, enroll_sec: float, gap_sec: float, lead_sec: float = 0.0) -> bool:
    return ex["duration"] <= max_mixture_sec(enroll_sec, gap_sec, lead_sec) + 1e-9


class Tokens:
    """Whisper decoder prompt + target construction (multilingual checkpoints)."""

    def __init__(self, tokenizer, max_len: int = 440):
        t = tokenizer
        self.tok = t
        self.sot = t.convert_tokens_to_ids("<|startoftranscript|>")
        self.prompt = [t.convert_tokens_to_ids(x) for x in ("<|en|>", "<|transcribe|>", "<|notimestamps|>")]
        self.eot = t.convert_tokens_to_ids("<|endoftext|>")
        self.pad = t.pad_token_id if t.pad_token_id is not None else self.eot
        self.max_len = max_len

    def build(self, text: str):
        ids = self.tok.encode(" " + text, add_special_tokens=False)[: self.max_len]
        dec_in = [self.sot] + self.prompt + ids
        labels = [-100] * len(self.prompt) + ids + [self.eot]
        return dec_in, labels


@dataclass
class DataConfig:
    conditioning: str = "prefix"        # prefix | spk_emb | none
    input_kind: str = "mixture"         # mixture | source (clean target, topline)
    gap_sec: float = 0.5                # separator length (silence, or frames of a learned token)
    position: str = "prefix"            # prefix | suffix (enrollment before / after the mixture)
    lead_sec: float = 0.0               # silence before everything (positional-embedding tests)
    random_lead: bool = False           # train: random lead within the free part of the window
    # training enrollment
    train: bool = False
    enroll_min: float = 5.0
    enroll_max: float = 5.0
    random_enrollment: bool = True      # draw a new utterance of the target speaker every epoch
    # evaluation enrollment
    enroll_mode: str = "target"         # target | interferer | other | silence | none
    enroll_field: str = "enrollment"    # e.g. enrollment_other_session on AMI
    enroll_dur: Optional[float] = 5.0
    # frame labels for the auxiliary losses
    frame_labels: bool = False
    act_rel_db: float = 30.0
    seed: int = 0


class _Base(Dataset):
    def __init__(self, feature_extractor, tokens: Tokens, cfg: DataConfig, augmenter=None, perturber=None):
        self.fe = feature_extractor
        self.tokens = tokens
        self.cfg = cfg
        self.augmenter = augmenter
        self.perturber = perturber
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _rng(self, i):
        return np.random.default_rng([self.cfg.seed, self.epoch, i])

    def _enroll_len(self, mix_sec: float, rng) -> Optional[float]:
        c = self.cfg
        if not c.train:
            return c.enroll_dur
        hi = c.enroll_max
        if c.conditioning == "prefix":
            hi = min(hi, WHISPER_WINDOW_SEC - mix_sec - c.gap_sec - 2 * SLACK)
        lo = min(c.enroll_min, hi)
        return float(rng.uniform(lo, hi)) if hi > lo else hi

    def _finish(self, ex_id, mixture, sources, k, texts, enroll, rng, enroll_is_silence=False):
        """sources: per-speaker signals aligned with `mixture` (or None)."""
        c = self.cfg
        if c.train and self.augmenter is not None:
            enroll, mixture = self.augmenter(enroll, mixture, rng)
        elif self.perturber is not None:
            # a silence control stays digital silence, only the mixture is perturbed
            new_enroll, mixture = self.perturber(None if enroll_is_silence else enroll, mixture, ex_id)
            if not enroll_is_silence:
                enroll = new_enroll
        mixture = rms_normalize(mixture)
        if enroll is not None and not enroll_is_silence:
            enroll = rms_normalize(enroll)

        use_enroll = c.conditioning == "prefix" and enroll is not None
        lead = c.lead_sec
        if c.train and c.random_lead:
            used = len(mixture) / SAMPLE_RATE + ((len(enroll) / SAMPLE_RATE + c.gap_sec) if use_enroll else 0.0)
            lead = float(rng.uniform(0.0, max(WHISPER_WINDOW_SEC - used - 3 * SLACK, 0.0)))
        wav, spans = compose_input(enroll if use_enroll else None, mixture, c.gap_sec, c.position, lead)
        if len(wav) > WHISPER_WINDOW_SAMPLES:
            wav = wav[:WHISPER_WINDOW_SAMPLES]
        feats = self.fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]

        dec_in, labels = self.tokens.build(training_text(texts[k]))
        item = {
            "id": ex_id,
            "input_features": feats.astype(np.float32),
            "dec_in": dec_in,
            "labels": labels,
            "enroll_span": spans["enroll"],
            "sep_span": spans["sep"],
            "mix_offset": spans["mix"][0],
        }
        if c.conditioning == "spk_emb":
            item["enroll_wav"] = enroll if enroll is not None else np.zeros(SAMPLE_RATE, np.float32)
        if c.frame_labels:
            item["frame_labels"] = self._frame_labels(sources, k, spans["mix"][0])
        return item

    def _frame_labels(self, sources, k, off):
        fl = np.full(WHISPER_ENC_FRAMES, IGNORE, dtype=np.int8)
        if not sources:
            return fl
        acts = [speech_activity(s, rel_db=self.cfg.act_rel_db) for s in sources]
        n = max(min(min(len(a) for a in acts), WHISPER_ENC_FRAMES - off), 0)
        acts = [a[:n] for a in acts]
        tgt = acts[k]
        others = np.zeros(n, dtype=bool)
        for j, a in enumerate(acts):
            if j != k:
                others |= a
        code = np.full(n, SIL, dtype=np.int8)
        code[tgt & ~others] = TGT
        code[~tgt & others] = OTHER
        code[tgt & others] = OVL
        fl[off: off + n] = code
        return fl

    def _load_enroll(self, src: dict, dur, rng) -> np.ndarray:
        wav = load_audio(src["path"], start=src.get("start"), end=src.get("end"))
        return crop(trim_silence(wav), dur, rng if self.cfg.train else None)


class ManifestDataset(_Base):
    def __init__(self, examples: List[dict], feature_extractor, tokens: Tokens, cfg: DataConfig,
                 augmenter=None, perturber=None, speaker_pool: Optional[Dict[str, List[dict]]] = None):
        super().__init__(feature_extractor, tokens, cfg, augmenter, perturber)
        self.ex = examples
        self.pool = speaker_pool

    def __len__(self):
        return len(self.ex)

    def _enroll_source(self, ex, rng) -> Optional[dict]:
        c = self.cfg
        if c.train:
            if c.random_enrollment and self.pool:
                spk = ex["speakers"][ex["target_index"]]
                excl = set(ex.get("source_utts", []))
                cands = [u for u in self.pool.get(spk, []) if u["utt_id"] not in excl]
                long_enough = [u for u in cands if u["duration"] >= c.enroll_max + 0.5]
                cands = long_enough or cands
                if cands:
                    return cands[int(rng.integers(len(cands)))]
            return ex[c.enroll_field]
        if c.enroll_mode in ("target", "silence"):
            return ex.get(c.enroll_field)
        if c.enroll_mode == "interferer":
            return ex.get("interferer_enrollment")
        if c.enroll_mode == "other":
            return ex.get("other_enrollment")
        return None

    def __getitem__(self, i):
        ex = self.ex[i]
        c = self.cfg
        rng = self._rng(i)
        k = ex["target_index"]
        start, end = ex.get("mixture_span") or (None, None)
        has_src = bool(ex.get("sources"))
        need_sources = has_src and (c.frame_labels or c.input_kind == "source")
        sources = [load_audio(p) for p in ex["sources"]] if need_sources else None
        if c.input_kind == "source":
            if sources is None:
                raise ValueError(f"{ex['id']}: no clean sources for input_kind=source")
            mixture = sources[k]
        else:
            mixture = load_audio(ex["mixture"], start=start, end=end)

        enroll, silence = None, False
        if c.conditioning != "none" and (c.train or c.enroll_mode != "none"):
            src = self._enroll_source(ex, rng)
            if src is None:
                raise ValueError(f"{ex['id']}: no enrollment for mode {c.enroll_mode}/{c.enroll_field}")
            enroll = self._load_enroll(src, self._enroll_len(len(mixture) / SAMPLE_RATE, rng), rng)
            if not c.train and c.enroll_mode == "silence":
                enroll, silence = np.zeros_like(enroll), True
        return self._finish(ex["id"], mixture, sources, k, ex["texts"], enroll, rng, silence)


# kept for code that still imports the old name
TSASRDataset = ManifestDataset


class OnTheFlyMixDataset(_Base):
    """LibriMix-style mixtures drawn on the fly from a LibriSpeech utterance index.

    Each source is scaled to an RMS level drawn from level_db; overlap="full" starts
    every source at 0 (LibriMix max mode), overlap="partial" delays each further source
    by a random offset inside the current mixture. Mixtures never exceed max_mix_sec.
    """

    def __init__(self, utterances: List[dict], feature_extractor, tokens: Tokens, cfg: DataConfig,
                 n_src_probs: Optional[Dict[int, float]] = None, overlap: str = "full",
                 samples_per_epoch: int = 50000, max_mix_sec: Optional[float] = None,
                 level_db=(-33.0, -25.0), augmenter=None):
        super().__init__(feature_extractor, tokens, cfg, augmenter, None)
        assert overlap in ("full", "partial")
        self.n_src_probs = n_src_probs or {2: 1.0}
        self.overlap, self.samples, self.level_db = overlap, samples_per_epoch, level_db
        self.max_mix = max_mix_sec if max_mix_sec is not None else max_mixture_sec(cfg.enroll_max, cfg.gap_sec)
        self.mix_pool, self.enroll_pool = defaultdict(list), defaultdict(list)
        for u in utterances:
            if u["duration"] >= 1.0:
                self.enroll_pool[u["speaker"]].append(u)
                if u["duration"] <= self.max_mix:
                    self.mix_pool[u["speaker"]].append(u)
        need_enroll = cfg.conditioning != "none"
        self.speakers = sorted(s for s in self.mix_pool if not need_enroll or len(self.enroll_pool[s]) >= 2)
        if len(self.speakers) < max(self.n_src_probs):
            raise ValueError("not enough speakers for the requested speaker count")

    def __len__(self):
        return self.samples

    def __getitem__(self, i):
        c = self.cfg
        rng = self._rng(i)
        counts = sorted(self.n_src_probs)
        p = np.array([self.n_src_probs[n] for n in counts], dtype=float)
        n = int(rng.choice(counts, p=p / p.sum()))
        spks = list(rng.choice(self.speakers, n, replace=False))
        utts = [self.mix_pool[s][int(rng.integers(len(self.mix_pool[s])))] for s in spks]
        srcs = []
        for u in utts:
            x = load_audio(u["path"])
            srcs.append(x * (10 ** (float(rng.uniform(*self.level_db)) / 20) / rms(x)))
        max_len = int(self.max_mix * SAMPLE_RATE)
        offs, cur = [], 0
        for j, x in enumerate(srcs):
            o = 0
            if self.overlap == "partial" and j > 0:
                o = int(rng.integers(0, max(min(cur, max_len - len(x)), 0) + 1))
            offs.append(o)
            cur = max(cur, o + len(x))
        positioned = [np.pad(x, (o, cur - o - len(x))).astype(np.float32) for x, o in zip(srcs, offs)]
        mix = np.sum(positioned, axis=0)
        peak = float(np.max(np.abs(mix)))
        if peak > 0.9:
            positioned = [x * (0.9 / peak) for x in positioned]
            mix = mix * (0.9 / peak)
        k = int(rng.integers(n))
        enroll = None
        if c.conditioning != "none":
            cands = [u for u in self.enroll_pool[spks[k]] if u["utt_id"] != utts[k]["utt_id"]]
            long_enough = [u for u in cands if u["duration"] >= c.enroll_max + 0.5]
            cands = long_enough or cands
            src = cands[int(rng.integers(len(cands)))]
            enroll = self._load_enroll(src, self._enroll_len(len(mix) / SAMPLE_RATE, rng), rng)
        return self._finish(f"otf-e{self.epoch}-{i}__s{k + 1}", mix.astype(np.float32),
                            positioned if c.frame_labels else None, k, [u["text"] for u in utts], enroll, rng)


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        L = max(len(b["dec_in"]) for b in batch)
        dec = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            dec[i, : len(b["dec_in"])] = torch.tensor(b["dec_in"])
            lab[i, : len(b["labels"])] = torch.tensor(b["labels"])
        out = {
            "ids": [b["id"] for b in batch],
            "input_features": torch.from_numpy(np.stack([b["input_features"] for b in batch])),
            "decoder_input_ids": dec,
            "labels": lab,
            "enroll_span": torch.tensor([b["enroll_span"] for b in batch], dtype=torch.long),
            "sep_span": torch.tensor([b["sep_span"] for b in batch], dtype=torch.long),
            "mix_offset": torch.tensor([b["mix_offset"] for b in batch]),
        }
        if "frame_labels" in batch[0]:
            out["frame_labels"] = torch.from_numpy(np.stack([b["frame_labels"] for b in batch])).long()
        if "enroll_wav" in batch[0]:
            n = max(len(b["enroll_wav"]) for b in batch)
            w = torch.zeros(len(batch), n)
            m = torch.zeros(len(batch), n, dtype=torch.long)
            for i, b in enumerate(batch):
                w[i, : len(b["enroll_wav"])] = torch.from_numpy(np.asarray(b["enroll_wav"], dtype=np.float32))
                m[i, : len(b["enroll_wav"])] = 1
            out["enroll_wav"], out["enroll_mask"] = w, m
        return out
