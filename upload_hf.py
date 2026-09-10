# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""Publish the 2006–now corpus to Hugging Face: NathanielArfin/canadian-hansard-2006-now.

Order matters: small files first (repo exists + instantly useful), then
tokens.bin (what Colab needs), then the big raw text (plan-B / completeness).
Auth via HF_TOKEN env var — never written to disk.
"""
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
REPO = "NathanielArfin/canadian-hansard-2006-now"

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=REPO, repo_type="dataset", exist_ok=True, private=False)
print(f"repo ready: https://huggingface.co/datasets/{REPO}", flush=True)

FILES = [("hf-card.md", "README.md"),
         ("merges.json", "merges.json"),
         ("tokens-info.json", "tokens-info.json"),
         ("tokens.bin", "tokens.bin"),
         ("hansard-2006-now.txt", "hansard-2006-now.txt")]

for src, dst in FILES:
    mb = (ROOT / src).stat().st_size / 1e6
    t0 = time.time()
    print(f"uploading {src} ({mb:,.0f} MB) -> {dst} ...", flush=True)
    api.upload_file(path_or_fileobj=str(ROOT / src), path_in_repo=dst,
                    repo_id=REPO, repo_type="dataset")
    print(f"  done in {time.time() - t0:,.0f}s", flush=True)

print("ALL UPLOADED", flush=True)
