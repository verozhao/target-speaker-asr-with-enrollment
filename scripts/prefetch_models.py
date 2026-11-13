#!/usr/bin/env python
"""Download every model the experiments use into HF_HOME, once, on a machine with internet
(e.g. a login node). Only files are downloaded; no model code is executed. Jobs on compute
nodes can then run with HF_HUB_OFFLINE=1.

  python scripts/prefetch_models.py                 # whisper-small, whisper-tiny (tests), WavLM, DiCoW
  python scripts/prefetch_models.py --skip-dicow    # without the 3.8 GB DiCoW checkpoint
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

from tsasr import DICOW_MODEL, DICOW_REVISION  # noqa: E402

SKIP = ("*.h5", "*.msgpack", "*.onnx", "*.ot", "*.tflite", "*.mlmodel", "*.pt", "*.ckpt")


def fetch(repo: str, revision=None):
    files = HfApi().list_repo_files(repo, revision=revision)
    weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
    path = snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.txt", "*.py", "*.model",
                                                                      "*.tiktoken", weights],
                             ignore_patterns=list(SKIP))
    print(f"{repo}@{revision or 'main'} -> {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--skip-dicow", action="store_true")
    p.add_argument("--extra", nargs="*", default=[], help="more repo ids, e.g. microsoft/wavlm-large")
    a = p.parse_args()
    repos = [(a.model, None), ("openai/whisper-tiny", None), ("microsoft/wavlm-base-plus", None),
             ("microsoft/wavlm-base-plus-sv", None)] + [(r, None) for r in a.extra]
    if not a.skip_dicow:
        repos.append((DICOW_MODEL, DICOW_REVISION))
    print(f"HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    for repo, rev in repos:
        fetch(repo, rev)
    print("done. Jobs can now run with HF_HUB_OFFLINE=1")


if __name__ == "__main__":
    main()
