"""Noise (WHAM!) and reverberation (OpenSLR 28 simulated RIRs) augmentation.

Training modes (marginal per-segment probabilities are identical in every mode):
  none         no augmentation
  mix_only     mixture gets noise / RIR, enrollment stays clean
  both_shared  both get independent noise segments, and the SAME RIR
               (one reverb decision shared by enrollment and mixture)
  both_indep   both get independent noise segments and independently sampled RIRs

Evaluation perturbations are deterministic per example (seeded by example id).
"""
from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

import numpy as np

from .audio import add_noise, apply_rir, load_audio

TRAIN_MODES = ("none", "mix_only", "both_shared", "both_indep")
EVAL_CONDITIONS = ("clean", "enroll_perturbed", "mix_perturbed", "both_perturbed")


def read_list(path: Optional[str]) -> List[str]:
    if not path:
        return []
    with open(path) as f:
        # rir lists are "room_id<TAB>path", noise lists are "path"
        return [ln.rstrip("\n").split("\t")[-1] for ln in f if ln.strip()]


def stable_seed(*parts) -> int:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


class Augmenter:
    def __init__(self, noise_list: Optional[str], rir_list: Optional[str], mode: str = "none",
                 p_noise: float = 0.5, p_rir: float = 0.5, snr_range=(0.0, 20.0)):
        assert mode in TRAIN_MODES, mode
        self.mode = mode
        self.noises = read_list(noise_list)
        self.rirs = read_list(rir_list)
        self.p_noise, self.p_rir = p_noise, p_rir
        self.snr_range = snr_range
        if mode != "none":
            if not self.noises and p_noise > 0:
                raise ValueError("noise augmentation requested but noise list is empty")
            if not self.rirs and p_rir > 0:
                raise ValueError("RIR augmentation requested but RIR list is empty")

    def _noise(self, x, rng, snr=None):
        n = load_audio(self.noises[int(rng.integers(len(self.noises)))])
        snr = float(rng.uniform(*self.snr_range)) if snr is None else snr
        return add_noise(x, n, snr, rng)

    def _rir(self, rng) -> np.ndarray:
        return load_audio(self.rirs[int(rng.integers(len(self.rirs)))])

    def segment(self, x, rng, rir=None, noise=True, snr=None):
        if rir is not None:
            x = apply_rir(x, rir)
        if noise:
            x = self._noise(x, rng, snr)
        return x

    def __call__(self, enroll: Optional[np.ndarray], mixture: np.ndarray,
                 rng: np.random.Generator) -> Tuple[Optional[np.ndarray], np.ndarray]:
        if self.mode == "none":
            return enroll, mixture
        mix_rev = rng.random() < self.p_rir
        mix_rir = self._rir(rng) if mix_rev else None
        mixture = self.segment(mixture, rng, mix_rir, noise=rng.random() < self.p_noise)
        if enroll is None or self.mode == "mix_only":
            return enroll, mixture
        if self.mode == "both_shared":
            enr_rir = mix_rir
        else:  # both_indep
            enr_rir = self._rir(rng) if rng.random() < self.p_rir else None
        enroll = self.segment(enroll, rng, enr_rir, noise=rng.random() < self.p_noise)
        return enroll, mixture


class EvalPerturber:
    """Deterministic test-time perturbation with held-out noise and rooms."""

    def __init__(self, noise_list: Optional[str], rir_list: Optional[str], condition: str = "clean",
                 snr_db: float = 5.0, reverb: bool = True, noise: bool = True):
        assert condition in EVAL_CONDITIONS, condition
        self.condition = condition
        self.aug = Augmenter(noise_list, rir_list, "none")
        self.snr_db, self.reverb, self.noise = snr_db, reverb, noise
        if condition != "clean":
            if noise and not self.aug.noises:
                raise ValueError("eval noise list is empty")
            if reverb and not self.aug.rirs:
                raise ValueError("eval RIR list is empty")

    def _apply(self, x, key):
        rng = np.random.default_rng(stable_seed(key))
        rir = self.aug._rir(rng) if self.reverb else None
        return self.aug.segment(x, rng, rir, noise=self.noise, snr=self.snr_db)

    def __call__(self, enroll, mixture, ex_id: str):
        c = self.condition
        if enroll is not None and c in ("enroll_perturbed", "both_perturbed"):
            enroll = self._apply(enroll, f"{ex_id}|enroll")
        if c in ("mix_perturbed", "both_perturbed"):
            # keyed on the mixture id so every target of a mixture sees the same input
            mixture = self._apply(mixture, f"{ex_id.split('__')[0]}|mix")
        return enroll, mixture
