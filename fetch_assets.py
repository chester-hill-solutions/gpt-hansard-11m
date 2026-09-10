# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""Fetch every model/data artifact from Hugging Face into this repo.

Reads ASSETS.md (one line per asset):  <local path>\t<HF resolve URL>
Skips anything already present. Idempotent — safe to re-run.

Usage:  uv run fetch_assets.py            (from the repo dir)
"""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "ASSETS.md"
LOCAL = "local"
HF = "hf"


def parse_manifest() -> list[tuple[str, str]]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"skip malformed line: {line!r}")
            continue
        rows.append((parts[LOCAL], parts[HF]))
    return rows


def main() -> int:
    if not MANIFEST.exists():
        print(f"{MANIFEST.name} missing — nothing to fetch")
        return 1
    rows = parse_manifest()
    if not rows:
        print("no assets listed")
        return 1
    for local, url in rows:
        dest = ROOT / local
        if dest.exists() and dest.stat().st_size > 0:
            print(f"exists: {local}")
            continue
        # parse a resolve URL of either form (dataset URLs add a segment):
        #   ...huggingface.co/<owner>/<repo>/resolve/<rev>/<file...>
        #   ...huggingface.co/datasets/<owner>/<repo>/resolve/<rev>/<file...>
        tail = url.split("huggingface.co/", 1)[1]
        segs = tail.split("/")
        if segs[0] == "datasets":
            segs = segs[1:]
        repo_id = f"{segs[0]}/{segs[1]}"
        filename = "/".join(segs[4:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch: {local} <- {repo_id}:{filename}")
        got = hf_hub_download(repo_id=repo_id, filename=filename)
        dest.write_bytes(Path(got).read_bytes())
        print(f"  saved {dest.stat().st_size/1e6:,.1f} MB")
    print("ALL ASSETS PRESENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())