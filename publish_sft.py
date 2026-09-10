# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""Publish the SFT artifacts missing from HF: 2 checkpoints + the SFT corpus.

The base/bilingual/named checkpoints are already on HF. This adds:
  - gpt-11m-sft-en.pt, gpt-11m-sft-en-v3.pt       -> model repo
  - sft-stream-en.bin(+labels), qp-pairs, heldout -> dataset repo

FILES are the big blobs that the git repo only pointers at; they live in a
source directory of your choosing (default: the local GPT lab). Auth via
HF_TOKEN env var — never written to disk:

    HF_TOKEN=... uv run publish_sft.py --from /path/to/gpt-from-scratch
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

MODEL_REPO = "NathanielArfin/gpt-hansard-11m"
SFT_REPO = "NathanielArfin/gpt-hansard-11m-sft"

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="src_root", default=Path(__file__).resolve().parent)
args = ap.parse_args()

SRC = Path(args.src_root)
api = HfApi(token=os.environ["HF_TOKEN"])

REPO_FILES = [  # (source path, destination path, repo, repo_type)
    ("gpt-11m-sft-en.pt", "gpt-11m-sft-en.pt", MODEL_REPO, "model"),
    ("gpt-11m-sft-en-v3.pt", "gpt-11m-sft-en-v3.pt", MODEL_REPO, "model"),
    ("sft-stream-en.bin", "sft-stream-en.bin", SFT_REPO, "dataset"),
    ("sft-stream-en.bin.labels", "sft-stream-en.bin.labels", SFT_REPO, "dataset"),
    ("sft/qp-pairs.jsonl", "qp-pairs.jsonl", SFT_REPO, "dataset"),
    ("sft/qp-heldout.jsonl", "qp-heldout.jsonl", SFT_REPO, "dataset"),
]


def up(src: Path, dst: str, repo: str, repo_type: str):
    mb = src.stat().st_size / 1e6
    print(f"uploading {src.name} ({mb:,.1f} MB) -> {repo}:{dst}", flush=True)
    api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                    repo_id=repo, repo_type=repo_type)


api.create_repo(SFT_REPO, repo_type="dataset", exist_ok=True, private=False)
for rel, dst, repo, repo_type in REPO_FILES:
    src = SRC / rel
    if not src.exists():
        print(f"MISSING {src} — skipped (upload the rest anyway)", flush=True)
        continue
    up(src, dst, repo, repo_type)
print("ALL UPLOADED", flush=True)