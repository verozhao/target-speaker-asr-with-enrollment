"""Manifest-driven dataset for enrollment-prefixed target-speaker ASR.

Manifest: one JSON object per line and per (mixture, target speaker):
  id               "<mixture_id>__s<k>"
  mixture_id, split, n_src, duration (mixture seconds)
  mixture          path of the mixture wav
  sources          paths of the n_src source wavs (s1, s2, ...)
  source_utts      LibriSpeech utterance ids of the sources
  speakers         speaker ids of the sources
  texts            LibriSpeech transcripts of the sources
  target_index     k-1
  enrollment       {utt_id, path, duration, policy}  target speaker, not in this mixture
  other_enrollment {utt_id, path, duration, speaker} a speaker absent from the mixture
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from . import ENC_HOP, SAMPLE_RATE, WHISPER_ENC_FRAMES, WHISPER_WINDOW_SAMPLES, WHISPER_WINDOW_SEC
from .audio import compose_prefix, crop, load_audio, rms_normalize, speech_activity, trim_silence
from .text import training_text

# frame label codes (encoder rate, 50 Hz)
IGNORE, SIL, TGT, OTHER, OVL = -1, 0, 1, 2, 3


def read_manifest(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def fits(ex: dict, enroll_sec: float, gap_sec: float) -> bool:
    # one encoder hop of slack for the frame alignment of the gap
    return ex["duration"] + gap_sec + enroll_sec + ENC_HOP / SAMPLE_RATE <= WHISPER_WINDOW_SEC


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
    input_kind: str = "auto"            # auto | mixture | source (oracle topline)
    gap_sec: float = 0.5
    # training enrollment
    train: bool = False
    enroll_min: float = 1.0
    enroll_max: float = 10.0
    random_enrollment: bool = True      # resample the target's enrollment utterance every epoch
    # evaluation enrollment
    enroll_mode: str = "target"         # target | other | silence | none
    enroll_dur: Optional[float] = 5.0
    # labels for the auxiliary loss
    frame_labels: bool = False
    act_rel_db: float = 30.0
    seed: int = 0
    extra: Dict = field(default_factory=dict)


class TSASRDataset(Dataset):
    def __init__(self, examples: List[dict], feature_extractor, tokens: Tokens, cfg: DataConfig,
                 augmenter=None, perturber=None, speaker_pool: Optional[Dict[str, List[dict]]] = None):
        self.ex = examples
        self.fe = feature_extractor
        self.tokens = tokens
        self.cfg = cfg
        self.augmenter = augmenter
        self.perturber = perturber
        self.pool = speaker_pool
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.ex)

    # ------------------------------------------------------------------ enrollment
    def _enroll_source(self, ex, rng) -> Optional[dict]:
        c = self.cfg
        if c.train:
            if c.random_enrollment and self.pool:
                spk = ex["speakers"][ex["target_index"]]
                cands = [u for u in self.pool.get(spk, []) if u["utt_id"] not in set(ex["source_utts"])]
                if cands:
                    return cands[int(rng.integers(len(cands)))]
            return ex["enrollment"]
        if c.enroll_mode in ("target", "silence"):
            return ex["enrollment"]
        if c.enroll_mode == "other":
            return ex["other_enrollment"]
        return None  # none

    def _enrollment(self, ex, mix_sec, rng) -> Optional[np.ndarray]:
        c = self.cfg
        if c.conditioning == "none" or (not c.train and c.enroll_mode == "none"):
            return None
        src = self._enroll_source(ex, rng)
        wav = trim_silence(load_audio(src["path"]))
        if c.train:
            room = WHISPER_WINDOW_SEC - mix_sec - c.gap_sec - 2 * ENC_HOP / SAMPLE_RATE
            hi = min(c.enroll_max, room) if c.conditioning == "prefix" else c.enroll_max
            lo = min(c.enroll_min, hi)
            wav = crop(wav, float(rng.uniform(lo, hi)), rng)
        else:
            wav = crop(wav, c.enroll_dur, None)
        if not c.train and c.enroll_mode == "silence":
            wav = np.zeros_like(wav)
        return wav

    # ------------------------------------------------------------------ item
    def __getitem__(self, i):
        ex = self.ex[i]
        c = self.cfg
        rng = np.random.default_rng([c.seed, self.epoch, i])
        k = ex["target_index"]

        need_sources = c.frame_labels or c.input_kind == "source"
        sources = [load_audio(p) for p in ex["sources"]] if need_sources else None
        if c.input_kind == "source":
            mixture = sources[k]
        else:
            mixture = load_audio(ex["mixture"])
        mix_sec = len(mixture) / SAMPLE_RATE

        enroll = self._enrollment(ex, mix_sec, rng)
        enroll_is_silence = enroll is not None and not np.any(enroll)

        if c.train and self.augmenter is not None:
            enroll, mixture = self.augmenter(enroll, mixture, rng)
        elif self.perturber is not None:
            # a silence control stays digital silence, only the mixture is perturbed
            new_enroll, mixture = self.perturber(None if enroll_is_silence else enroll, mixture, ex["id"])
            if not enroll_is_silence:
                enroll = new_enroll

        mixture = rms_normalize(mixture)
        if enroll is not None and not enroll_is_silence:
            enroll = rms_normalize(enroll)

        if c.conditioning == "prefix" and enroll is not None:
            wav, n_enr, off = compose_prefix(enroll, mixture, c.gap_sec)
        else:
            wav, n_enr, off = mixture, 0, 0
        if len(wav) > WHISPER_WINDOW_SAMPLES:
            wav = wav[:WHISPER_WINDOW_SAMPLES]

        feats = self.fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]

        text = training_text(ex["texts"][k])
        dec_in, labels = self.tokens.build(text)

        item = {
            "id": ex["id"],
            "input_features": feats.astype(np.float32),
            "dec_in": dec_in,
            "labels": labels,
            "n_enroll_frames": n_enr,
            "mix_offset": off,
        }
        if c.conditioning == "spk_emb":
            item["enroll_wav"] = enroll if enroll is not None else np.zeros(SAMPLE_RATE, np.float32)
        if c.frame_labels:
            acts = [speech_activity(s, rel_db=c.act_rel_db) for s in sources]
            n = min(min(len(a) for a in acts), WHISPER_ENC_FRAMES - off)
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
            fl = np.full(WHISPER_ENC_FRAMES, IGNORE, dtype=np.int8)
            fl[off: off + n] = code
            item["frame_labels"] = fl
        return item


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
            "n_enroll_frames": torch.tensor([b["n_enroll_frames"] for b in batch]),
            "mix_offset": torch.tensor([b["mix_offset"] for b in batch]),
        }
        if "frame_labels" in batch[0]:
            out["frame_labels"] = torch.from_numpy(np.stack([b["frame_labels"] for b in batch])).long()
        if "enroll_wav" in batch[0]:
            n = max(len(b["enroll_wav"]) for b in batch)
            w = torch.zeros(len(batch), n)
            m = torch.zeros(len(batch), n, dtype=torch.long)
            for i, b in enumerate(batch):
                w[i, : len(b["enroll_wav"])] = torch.from_numpy(b["enroll_wav"])
                m[i, : len(b["enroll_wav"])] = 1
            out["enroll_wav"], out["enroll_mask"] = w, m
        return out
