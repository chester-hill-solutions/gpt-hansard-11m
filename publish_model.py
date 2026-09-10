# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""Publish GPT-Hansard-11M to the Hub: model repo + interactive Space.

Usage: HF_TOKEN=... uv run publish_model.py [--stage base|sft|space|all]
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
MODEL = "NathanielArfin/gpt-hansard-11m"
SPACE = "NathanielArfin/hansard-chat"

api = HfApi(token=os.environ.get("HF_TOKEN"))  # None -> use stored/active token
stage = sys.argv[1] if len(sys.argv) > 1 else "all"


def up(src: str, dst: str, repo: str, repo_type: str):
    mb = (ROOT / src).stat().st_size / 1e6
    print(f"uploading {src} ({mb:,.1f} MB) -> {dst}", flush=True)
    api.upload_file(path_or_fileobj=str(ROOT / src), path_in_repo=dst,
                    repo_id=repo, repo_type=repo_type)


if stage in ("base", "all"):
    api.create_repo(MODEL, repo_type="model", exist_ok=True, private=False)
    up("hf-model-card.md", "README.md", MODEL, "model")
    up("gpt-11m-base.pt", "gpt-11m-base.pt", MODEL, "model")
    print(f"base published: https://huggingface.co/{MODEL}", flush=True)

if stage in ("sft", "all"):
    src = "gpt-11m-sft-bilingual.pt" if (ROOT / "gpt-11m-sft-bilingual.pt").exists() \
        else "gpt_tokens_sft.pt"
    up(src, "gpt-11m-sft-bilingual.pt", MODEL, "model")
    print(f"SFT checkpoint published (from {src})", flush=True)

if stage in ("space", "all"):
    api.create_repo(SPACE, repo_type="space", space_sdk="gradio",
                    exist_ok=True, private=False)
    up("space-app.py", "app.py", SPACE, "space")
    up("space-requirements.txt", "requirements.txt", SPACE, "space")
    print(f"Space published: https://huggingface.co/spaces/{SPACE}", flush=True)

print("DONE", flush=True)
