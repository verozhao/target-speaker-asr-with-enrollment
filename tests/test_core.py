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


# --------------------------------------------------------------------- new features
def test_compose_input_spans():
    from tsasr.audio import compose_input

    enr, mix = tone(5.0), tone(10.0)
    wav, sp = compose_input(enr, mix, gap_sec=0.5, position="prefix")
    assert sp["enroll"] == (0, 250) and sp["sep"][0] == 250 and sp["mix"][0] == sp["sep"][1]
    assert np.allclose(wav[sp["mix"][0] * ENC_HOP: sp["mix"][0] * ENC_HOP + len(mix)], mix)
    wav, sp = compose_input(enr, mix, gap_sec=0.5, position="suffix", lead_sec=2.0)
    assert sp["mix"] == (100, 600) and sp["enroll"][0] == sp["sep"][1]
    assert np.allclose(wav[sp["enroll"][0] * ENC_HOP: sp["enroll"][0] * ENC_HOP + len(enr)], enr)
    assert np.all(wav[: 100 * ENC_HOP] == 0)
    wav, sp = compose_input(enr, mix, gap_sec=0.02, position="prefix")  # learned-token separator
    assert 1 <= sp["sep"][1] - sp["sep"][0] <= 2
    wav, sp = compose_input(None, mix, lead_sec=1.0)
    assert sp["enroll"] == (0, 0) and sp["mix"][0] == 50


def test_fits_matches_spec():
    from tsasr.data import fits, max_mixture_sec

    assert abs(max_mixture_sec(5.0, 0.5) - 24.48) < 1e-6
    assert fits({"duration": 24.4}, 5.0, 0.5) and not fits({"duration": 24.6}, 5.0, 0.5)


def _write_ami(tmp_path):
    ann = tmp_path / "ann"
    (ann / "corpusResources").mkdir(parents=True)
    (ann / "words").mkdir()
    sess_xml = "".join(
        f'<sess observation="{s}">'
        + "".join(f'<speaker nxt_agent="{ag}" global_name="{g}"/>' for ag, g in zip("AB", ("SPK1", "SPK2")))
        + "</sess>" for s in ("S001a", "S001b"))
    (ann / "corpusResources" / "sessions.xml").write_text(f"<root>{sess_xml}</root>")

    def words(items):
        body = "".join(f'<w starttime="{s}" endtime="{e}">{t}</w><w starttime="{e}" endtime="{e}" punc="true">.</w>'
                       for s, e, t in items)
        return f"<root>{body}<vocalsound starttime='1' endtime='2'/></root>"

    a = [(1.0 + i, 1.8 + i, w) for i, w in enumerate("one two three four five six seven X_M_L_".split())]
    a += [(40.0 + 0.5 * i, 40.4 + 0.5 * i, "solo") for i in range(14)]  # solo stretch for enrollment
    b = [(3.2, 3.9, "hello"), (4.1, 4.6, "there"), (60.0, 66.5, "alone")]
    for s in ("S001a", "S001b"):
        (ann / "words" / f"{s}.A.words.xml").write_text(words(a))
        (ann / "words" / f"{s}.B.words.xml").write_text(words(b))
    return str(ann)


def test_ami_parsing_and_windows(tmp_path):
    from tsasr.ami import read_speaker_map, read_words, solo_regions, target_windows, text_of, words_in

    ann = _write_ami(tmp_path)
    smap = read_speaker_map(ann)
    assert smap["S001a"] == {"A": "SPK1", "B": "SPK2"}
    ws = read_words(ann, "S001a", smap)
    assert set(ws) == {"SPK1", "SPK2"} and ws["SPK1"][7][2] == "XML"
    assert all(t != "." for _, _, t in ws["SPK1"])  # punctuation dropped
    wins = target_windows(ws["SPK1"], max_len=24.5)
    for st, en, i, j in wins:
        assert en - st <= 24.5 + 1e-9
        assert all(st <= w[0] and w[1] <= en for w in ws["SPK1"][i: j + 1])
    st, en, i, j = wins[0]
    assert text_of(words_in(ws["SPK2"], st, en)) == "hello there"
    solo = solo_regions(ws, "SPK1", min_len=2.0)
    assert solo and solo[0][0] >= 39.0  # the overlapped start is not a solo region


def test_attach_interferer():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from evaluate import attach_interferer

    ex = [{"id": f"m__s{k + 1}", "mixture_id": "m", "target_index": k, "n_src": 2, "sources": ["a", "b"],
           "enrollment": {"path": f"e{k}"}} for k in range(2)]
    ami_solo = {"id": "w__A", "mixture_id": "w", "target_index": 0, "n_src": 1, "sources": [],
                "enrollment": {"path": "own"}}  # nobody else speaks: must not pair with itself
    ami = {"id": "v__A", "mixture_id": "v", "target_index": 0, "n_src": 2, "sources": [],
           "enrollment": {"path": "own"}, "interferer_enrollment": {"path": "other"}, "interferer_index": 1}
    out = attach_interferer(ex + [{"id": "x__s1", "mixture_id": "x", "target_index": 0, "n_src": 2,
                                   "sources": ["a", "b"], "enrollment": {"path": "lonely"}}, ami_solo, ami])
    assert len(out) == 3  # x has no sibling line, the solo AMI window is dropped
    assert out[2]["id"] == "v__A" and out[2]["enrolled_index"] == 1
    assert out[0]["interferer_enrollment"]["path"] == "e1" and out[0]["enrolled_index"] == 1
    assert out[1]["interferer_enrollment"]["path"] == "e0" and out[1]["enrolled_index"] == 0


def test_compare_outputs(tmp_path):
    import json
    import subprocess

    def rec(i, hyp, closest, enrolled, mode):
        return {"id": i, "hyp_norm": hyp, "closest_speaker": closest, "enrolled_index": enrolled, "enroll_mode": mode}

    a = [rec("1", "a b c", 0, 0, "target"), rec("2", "x y", 0, 0, "target")]
    b = [rec("1", "d e f", 1, 1, "interferer"), rec("2", "x y", 0, 1, "interferer")]
    pa, pb = tmp_path / "a.hyps.jsonl", tmp_path / "b.hyps.jsonl"
    pa.write_text("\n".join(map(json.dumps, a)))
    pb.write_text("\n".join(map(json.dumps, b)))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, os.path.join(root, "scripts", "compare_outputs.py"), str(pa), str(pb)],
                         capture_output=True, text=True, check=True).stdout
    res = json.loads(out.strip().splitlines()[-1])
    assert res["identical_rate"] == 50.0 and res["switch_rate"] == 50.0 and res["follows_B"] == 50.0


def test_model_features_tiny(tmp_path):
    torch = pytest.importorskip("torch")
    from tsasr.model import TSWhisper

    feats = torch.randn(2, 80, 3000)
    dec = torch.tensor([[50258, 50259, 50360, 50364, 400], [50258, 50259, 50360, 50364, 401]])
    lab = torch.tensor([[-100, -100, -100, 400, 50257], [-100, -100, -100, 401, 50257]])
    sep = torch.tensor([[250, 276], [100, 126]])
    m = TSWhisper("openai/whisper-tiny", separator="learned", aux="activity")
    fl = torch.full((2, 1500), -1)
    fl[:, 300:400] = 1
    out = m(feats, dec, lab, frame_labels=fl, enroll_span=torch.tensor([[0, 250], [0, 100]]), sep_span=sep)
    out["loss"].backward()
    assert m.sep_emb.grad is not None and m.sep_emb.grad.abs().sum() > 0
    m.save(str(tmp_path / "lora"))
    m2 = TSWhisper.load(str(tmp_path / "lora"))
    assert torch.allclose(m2.sep_emb, m.sep_emb)
    f = TSWhisper("openai/whisper-tiny", conditioning="none", full_ft=True)
    enc_pos = f.whisper.model.encoder.embed_positions.weight
    assert not enc_pos.requires_grad  # fixed sinusoids stay frozen, as in Whisper
    assert all(p.requires_grad for p in f.whisper.parameters() if p is not enc_pos)
    with torch.no_grad():
        f.whisper.model.decoder.layers[0].fc1.bias.add_(1.0)
    f.save(str(tmp_path / "full"))
    f2 = TSWhisper.load(str(tmp_path / "full"))
    assert torch.allclose(f2.whisper.model.decoder.layers[0].fc1.bias, f.whisper.model.decoder.layers[0].fc1.bias)


def test_dicow_masks_from_segments():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from eval_dicow import stno_from_segments

    st = stno_from_segments([[(0.0, 2.0)], [(1.0, 3.0)]], 0)
    assert st.shape == (4, 1500) and np.allclose(st.sum(0), 1.0)
    assert st[1, 25] == 1 and st[3, 75] == 1 and st[2, 125] == 1 and st[0, 175] == 1
    solo = stno_from_segments([[(0.0, 1.0)]], 0)  # target alone in the window
    assert solo[1, 10] == 1 and solo[2].sum() == 0


class _StubFE:
    def __call__(self, wav, sampling_rate, return_tensors):
        class R:
            input_features = np.zeros((1, 80, 3000), np.float32)
        self.last_len = len(wav)
        return R()


class _StubTokens:
    def build(self, text):
        return [1, 2], [3, 4]


def _layout_item(enroll_sec, frontend="whisper", gap=0.5, mode_none=False):
    from tsasr.data import DataConfig, _Base

    ds = _Base(_StubFE(), _StubTokens(), DataConfig(gap_sec=gap, frontend=frontend))
    enr = None if mode_none else tone(enroll_sec)
    item = ds._finish("m__s1", tone(9.5), None, 0, ["a"], enr, np.random.default_rng(0))
    return item, getattr(ds.fe, "last_len", None)


def test_padding_layout_moves_mixture():
    for L in (1, 2, 5, 10):
        item, n = _layout_item(L)
        assert item["mix_offset"] == L * 50 + 25  # enrollment + 0.5 s separator
        assert n == int((L + 0.5 + 9.5) * SAMPLE_RATE)
        assert item["mix_frames"] == 475


def test_ssl_item_carries_waveform():
    item, n = _layout_item(5, frontend="ssl", gap=0.0)
    assert n is None and "input_features" not in item  # no log-Mel for the WavLM front-end
    assert len(item["wav"]) == int(14.5 * SAMPLE_RATE) and item["mix_offset"] == 250
    from tsasr.data import Collator

    b = Collator(0)([{**item, "dec_in": [1, 2], "labels": [3, 4]}, {**_layout_item(2, "ssl", 0.0)[0],
                                                                     "dec_in": [1], "labels": [3]}])
    assert b["wav"].shape == (2, int(14.5 * SAMPLE_RATE)) and b["wav_mask"][1].sum() == int(11.5 * SAMPLE_RATE)
    assert b["mix_offset"].tolist() == [250, 100] and "input_features" not in b


def test_enroll_choices_respect_room():
    from tsasr.data import DataConfig, _Base

    rng = np.random.default_rng(0)
    ds = _Base(None, None, DataConfig(train=True, enroll_choices=(1, 2, 5, 10)))
    assert {ds._enroll_len(19.0, rng) for _ in range(200)} == {1.0, 2.0, 5.0, 10.0}
    assert {ds._enroll_len(25.0, rng) for _ in range(200)} <= {1.0, 2.0}  # 30 - 25 - 0.5 leaves < 5 s
    ds = _Base(None, None, DataConfig(train=True, enroll_min=1, enroll_max=10))
    assert all(1.0 <= ds._enroll_len(19.0, rng) <= 10.0 for _ in range(200))


def test_ssl_frontend_tiny(tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

    from tsasr.model import TSWhisper

    ssl_dir = tmp_path / "tiny_wavlm"
    cfg = WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                      conv_dim=(32,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
    WavLMModel(cfg).save_pretrained(ssl_dir)
    Wav2Vec2FeatureExtractor(do_normalize=False, return_attention_mask=True).save_pretrained(ssl_dir)
    m = TSWhisper("openai/whisper-tiny", frontend="ssl", ssl_model=str(ssl_dir), ssl_train="full", full_ft=True)
    wav = torch.randn(2, int(9.5 * SAMPLE_RATE)) * 0.1
    mask = torch.ones_like(wav, dtype=torch.long)
    mask[1, int(7.0 * SAMPLE_RATE):] = 0
    dec = torch.tensor([[50258, 50259, 50360, 50364, 400], [50258, 50259, 50360, 50364, 401]])
    lab = torch.tensor([[-100, -100, -100, 400, 50257], [-100, -100, -100, 401, 50257]])
    kw = dict(wav=wav, wav_mask=mask, mix_offset=torch.tensor([250, 100]), mix_frames=torch.tensor([225, 250]))
    out = m(decoder_input_ids=dec, labels=lab, **kw)
    out["loss"].backward()
    assert m.ssl_proj.weight.grad.abs().sum() > 0 and m.ssl_layer_w.grad is not None
    assert m.ssl.encoder.layers[0].attention.q_proj.weight.grad is not None      # WavLM is fine-tuned
    assert all(p.grad is None for p in m.ssl.feature_extractor.parameters())    # its CNN stays frozen
    assert m.whisper.model.encoder.conv1.weight.grad is None                     # Whisper's stem is bypassed
    groups = m.param_groups(3e-4, 1e-5)
    assert any(p is m.ssl_proj.weight for p in groups[0]["params"]) and groups[1]["lr"] == 1e-5
    seqs = m.generate(max_new_tokens=3, **kw)
    assert seqs.shape[0] == 2
    m.save(str(tmp_path / "ck"))
    m2 = TSWhisper.load(str(tmp_path / "ck"))
    k = "encoder.layers.0.attention.q_proj.weight"
    assert torch.allclose(m2.ssl.state_dict()[k], m.ssl.state_dict()[k])
    assert torch.allclose(m2.ssl_proj.weight, m.ssl_proj.weight)


def test_selection_is_strict():
    from tsasr.evaluation import score

    ex = [{"id": f"{i}", "mixture_id": f"{i}", "target_index": 0, "texts": t}
          for i, t in enumerate([["red ball", "blue sky"], ["red ball", "blue sky"], ["red ball", "red ball"],
                                 ["red ball"]])]
    hyps = {"0": "red ball", "1": "", "2": "red ball", "3": "red ball"}  # right, empty, tie, single speaker
    recs, summ = score(ex, hyps)
    assert [r["follows_enrolled"] for r in recs] == [True, False, False, None]
    assert summ["n_selection"] == 3 and abs(summ["selection_acc"] - 100 / 3) < 1e-6


def _tiny_ssl_model(tmp_path, ssl_train):
    from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

    from tsasr.model import TSWhisper

    d = tmp_path / f"tiny_wavlm_{ssl_train}"
    cfg = WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                      conv_dim=(32,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
    assert cfg.feat_extract_norm == "group"
    WavLMModel(cfg).save_pretrained(d)
    Wav2Vec2FeatureExtractor(do_normalize=False, return_attention_mask=True).save_pretrained(d)
    return TSWhisper("openai/whisper-tiny", frontend="ssl", ssl_model=str(d), ssl_train=ssl_train)


def test_ssl_features_do_not_depend_on_batch_padding(tmp_path):
    torch = pytest.importorskip("torch")
    m = _tiny_ssl_model(tmp_path, "frozen").eval()
    short, long = torch.randn(1, 4 * SAMPLE_RATE) * 0.1, torch.randn(1, 9 * SAMPLE_RATE) * 0.1
    alone = m._ssl_frames(short, torch.ones_like(short, dtype=torch.long), torch.tensor([50]), torch.tensor([100]))
    wav = torch.zeros(2, 9 * SAMPLE_RATE)
    wav[0, : short.shape[1]], wav[1] = short[0], long[0]
    mask = (wav != 0).long()
    mask[0, : short.shape[1]] = 1
    mask[1] = 1
    batched = m._ssl_frames(wav, mask, torch.tensor([50, 50]), torch.tensor([100, 300]))
    assert torch.allclose(alone[0], batched[0], atol=1e-5)


def test_frozen_ssl_stays_in_eval_mode(tmp_path):
    m = _tiny_ssl_model(tmp_path, "frozen")
    m.train()
    assert m.training and not m.ssl.training and m.ssl.config.layerdrop == 0.0
