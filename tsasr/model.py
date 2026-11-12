"""Whisper for target-speaker ASR.

conditioning
  prefix   enrollment audio placed in the 30 s window next to the mixture; the encoder
           relates the two through self-attention (our method, no extra modules)
  spk_emb  an external speaker embedding (WavLM x-vector, TitaNet or ECAPA) projected to
           d_model and added to the encoder conv-stem output (baseline)
  none     mixture only (no-conditioning baseline)

separator  silence (the gap frames stay as they are) or learned (a trained vector
           replaces the encoder-stem frames of the gap)
adaptation LoRA (default) or full fine-tuning
aux        optional auxiliary loss on the final encoder states:
           activity     frame-level target-speaker activity (BCE) on mixture frames
           contrastive  pooled enrollment frames closer to target-only than to
                        interferer-only frames of the same mixture (2-way InfoNCE)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration

from .data import IGNORE, OTHER, OVL, TGT

CONFIG_NAME = "tsasr_config.json"
HEADS_NAME = "extra_heads.pt"
HEAD_NAMES = ("spk_proj", "act_head", "c_proj", "sep_emb")


class TSWhisper(nn.Module):
    def __init__(self, model_name: str = "openai/whisper-small", conditioning: str = "prefix",
                 aux: str = "none", aux_weight: float = 0.1, lora_r: int = 32, lora_alpha: int = 64,
                 lora_dropout: float = 0.05,
                 lora_targets: List[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
                 spk_dim: int = 512, contrastive_dim: int = 256, tau: float = 0.1,
                 spk_encoder: str = "wavlm", separator: str = "silence", full_ft: bool = False,
                 adapter_dir: Optional[str] = None, weights_dir: Optional[str] = None,
                 use_lora: bool = True):
        super().__init__()
        assert conditioning in ("prefix", "spk_emb", "none")
        assert aux in ("none", "activity", "contrastive")
        assert separator in ("silence", "learned")
        if aux == "contrastive" and conditioning != "prefix":
            raise ValueError("contrastive aux loss needs enrollment frames in the encoder (prefix)")
        if model_name.endswith(".en"):
            raise ValueError("use a multilingual Whisper checkpoint (prompt uses <|en|><|transcribe|>)")
        self.cfg = dict(model_name=model_name, conditioning=conditioning, aux=aux, aux_weight=aux_weight,
                        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                        lora_targets=list(lora_targets), spk_dim=spk_dim, contrastive_dim=contrastive_dim,
                        tau=tau, spk_encoder=spk_encoder, separator=separator, full_ft=full_ft,
                        use_lora=use_lora and not full_ft)
        base = WhisperForConditionalGeneration.from_pretrained(weights_dir or model_name)
        base.config.forced_decoder_ids = None
        base.generation_config.forced_decoder_ids = None
        for p in base.parameters():
            p.requires_grad_(full_ft)
        if full_ft or not use_lora:
            self.peft, self.whisper = None, base
        elif adapter_dir is not None:
            self.peft = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
            self.whisper = self.peft.base_model.model
        else:
            lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              target_modules=list(lora_targets), bias="none")
            self.peft = get_peft_model(base, lcfg)
            self.whisper = self.peft.base_model.model
        d = base.config.d_model
        self.conditioning, self.aux, self.aux_weight, self.tau = conditioning, aux, aux_weight, tau
        self.separator = separator

        self._spk_bias = None
        self._sep_mask = None
        if conditioning == "spk_emb":
            self.spk_proj = nn.Linear(spk_dim, d)
            nn.init.zeros_(self.spk_proj.weight)  # starts as plain Whisper
            nn.init.zeros_(self.spk_proj.bias)
        if separator == "learned":
            self.sep_emb = nn.Parameter(torch.randn(d) * 0.02)
        if aux == "activity":
            self.act_head = nn.Linear(d, 1)
        if aux == "contrastive":
            self.c_proj = nn.Linear(d, contrastive_dim)
        self.whisper.model.encoder.conv2.register_forward_hook(self._stem_hook)

    # ----------------------------------------------------------------- conditioning
    def _stem_hook(self, module, inputs, output):
        """output: (B, d_model, 1500) conv-stem features before GELU and positions."""
        if self._spk_bias is not None:
            output = output + self._spk_bias.to(output.dtype)[:, :, None]
        if self._sep_mask is not None:
            sep = self.sep_emb.to(output.dtype)[None, :, None]
            output = torch.where(self._sep_mask[:, None, :], sep, output)
        return output

    def _set_cond(self, spk_emb=None, sep_span=None, n_frames: int = 1500):
        self._spk_bias = self.spk_proj(spk_emb) if (self.conditioning == "spk_emb" and spk_emb is not None) else None
        self._sep_mask = None
        if self.separator == "learned" and sep_span is not None:
            t = torch.arange(n_frames, device=sep_span.device)[None, :]
            self._sep_mask = (t >= sep_span[:, :1]) & (t < sep_span[:, 1:2])

    def _clear_cond(self):
        self._spk_bias, self._sep_mask = None, None

    # ----------------------------------------------------------------- training
    def forward(self, input_features, decoder_input_ids, labels, spk_emb=None, frame_labels=None,
                enroll_span=None, sep_span=None, **_) -> Dict[str, torch.Tensor]:
        self._set_cond(spk_emb, sep_span, input_features.shape[-1] // 2)
        try:
            enc = self.whisper.model.encoder(input_features).last_hidden_state
        finally:
            self._clear_cond()
        out = self.whisper(encoder_outputs=(enc,), decoder_input_ids=decoder_input_ids, labels=labels)
        res = {"ce": out.loss, "loss": out.loss}
        if self.aux == "activity":
            res["aux"] = self._activity_loss(enc, frame_labels)
        elif self.aux == "contrastive":
            res["aux"] = self._contrastive_loss(enc, frame_labels, enroll_span)
        if "aux" in res:
            res["loss"] = out.loss + self.aux_weight * res["aux"]
        return res

    def _activity_loss(self, enc, fl):
        logits = self.act_head(enc).squeeze(-1).float()
        valid = fl != IGNORE
        if not valid.any():
            return logits.sum() * 0.0
        target = ((fl == TGT) | (fl == OVL)).float()
        return F.binary_cross_entropy_with_logits(logits[valid], target[valid])

    def _contrastive_loss(self, enc, fl, enroll_span, min_frames: int = 5):
        losses = []
        for b in range(enc.shape[0]):
            s, e = int(enroll_span[b, 0]), int(enroll_span[b, 1])
            t_mask, o_mask = fl[b] == TGT, fl[b] == OTHER
            if e - s < min_frames or t_mask.sum() < min_frames or o_mask.sum() < min_frames:
                continue
            z = torch.stack([enc[b, s:e].mean(0), enc[b][t_mask].mean(0), enc[b][o_mask].mean(0)])
            z = F.normalize(self.c_proj(z).float(), dim=-1)
            logits = torch.stack([z[0] @ z[1], z[0] @ z[2]]) / self.tau
            losses.append(F.cross_entropy(logits[None], torch.zeros(1, dtype=torch.long, device=enc.device)))
        if not losses:
            return enc.sum() * 0.0 + sum(p.sum() for p in self.c_proj.parameters()) * 0.0
        return torch.stack(losses).mean()

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, input_features, spk_emb=None, sep_span=None, num_beams: int = 1,
                 max_new_tokens: int = 220, prompt_ids: Optional[torch.Tensor] = None):
        self._set_cond(spk_emb, sep_span, input_features.shape[-1] // 2)
        kw = {"prompt_ids": prompt_ids} if prompt_ids is not None else {}
        try:
            return self.whisper.generate(input_features=input_features, language="en", task="transcribe",
                                         num_beams=num_beams, max_new_tokens=max_new_tokens, **kw)
        finally:
            self._clear_cond()

    # ----------------------------------------------------------------- io
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        if self.cfg["full_ft"]:
            self.whisper.save_pretrained(out_dir)
        elif self.peft is not None:
            self.peft.save_pretrained(out_dir)
        heads = {}
        for n in HEAD_NAMES:
            m = getattr(self, n, None)
            if m is not None:
                heads[n] = m.detach().cpu() if isinstance(m, nn.Parameter) else m.state_dict()
        torch.save(heads, os.path.join(out_dir, HEADS_NAME))
        with open(os.path.join(out_dir, CONFIG_NAME), "w") as f:
            json.dump(self.cfg, f, indent=2)

    @classmethod
    def load(cls, ckpt_dir: str, map_location="cpu") -> "TSWhisper":
        with open(os.path.join(ckpt_dir, CONFIG_NAME)) as f:
            cfg = json.load(f)
        train_data = cfg.pop("train_data", {})
        if cfg.get("full_ft"):
            m = cls(**cfg, weights_dir=ckpt_dir)
        else:
            m = cls(**cfg, adapter_dir=ckpt_dir if cfg.get("use_lora", True) else None)
        heads = torch.load(os.path.join(ckpt_dir, HEADS_NAME), map_location=map_location)
        for n, v in heads.items():
            target = getattr(m, n)
            if isinstance(target, nn.Parameter):
                target.data.copy_(v)
            else:
                target.load_state_dict(v)
        m.cfg["train_data"] = train_data
        return m


class SpeakerEncoder(nn.Module):
    """Frozen external speaker encoder for the speaker-embedding baseline.

    wavlm    microsoft/wavlm-base-plus-sv x-vector (512-d), transformers only
    titanet  nvidia/speakerverification_en_titanet_large (192-d), needs nemo_toolkit[asr]
    ecapa    speechbrain/spkrec-ecapa-voxceleb (192-d), needs speechbrain
    """

    def __init__(self, kind: str = "wavlm"):
        super().__init__()
        self.kind = kind
        if kind == "wavlm":
            from transformers import WavLMForXVector

            # TSASR_WAVLM_SV may point to a local copy (the hub repo only has a .bin,
            # which transformers>=4.50 loads only with torch>=2.6)
            self.net = WavLMForXVector.from_pretrained(os.environ.get("TSASR_WAVLM_SV", "microsoft/wavlm-base-plus-sv"))
            self.dim = self.net.config.xvector_output_dim
        elif kind == "titanet":
            from nemo.collections.asr.models import EncDecSpeakerLabelModel

            self.net = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
            self.dim = 192
        elif kind == "ecapa":
            from speechbrain.inference.speaker import EncoderClassifier

            self.net = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                      savedir=os.path.expanduser("~/.cache/speechbrain_ecapa"))
            self.dim = 192
        else:
            raise ValueError(kind)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # always frozen / eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embs = []
        for b in range(wav.shape[0]):  # unpadded, one utterance at a time
            n = int(mask[b].sum())
            x = wav[b, : max(n, 1600)][None].float()  # at least 0.1 s
            if not torch.any(x):  # silence control -> zero embedding
                embs.append(torch.zeros(self.dim, device=wav.device))
                continue
            if self.kind == "wavlm":
                x = (x - x.mean()) / (x.std() + 1e-7)
                e = self.net(input_values=x).embeddings[0]
            elif self.kind == "titanet":
                _, e = self.net.forward(input_signal=x, input_signal_length=torch.tensor([x.shape[1]], device=x.device))
                e = e.reshape(-1)
            else:
                if next(self.net.mods.parameters()).device != x.device:
                    self.net.to(x.device)
                    self.net.device = x.device
                e = self.net.encode_batch(x).reshape(-1)
            embs.append(F.normalize(e.float(), dim=-1))
        return torch.stack(embs)
