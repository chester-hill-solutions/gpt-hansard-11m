# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Retrain BPE merges on the bilingual corpus (era-sampled slice → merges-bilingual.json)."""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
corpus = ROOT / "hansard-bilingual.txt"
NUM_MERGES = 1024
T0 = time.time()

size = corpus.stat().st_size
slices, chunk = 12, 100_000
parts = []
with open(corpus, "rb") as f:
    for k in range(slices):
        pos = int((k + 0.5) * (size - chunk * 4) / slices)
        f.seek(pos)
        parts.append(f.read(chunk * 4).decode("utf-8", errors="ignore")[:chunk])
text = "".join(parts)
ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
print(f"sample: {len(text):,} chars ({len(ids):,} bytes) · training {NUM_MERGES} merges",
      flush=True)

out = []
for i in range(NUM_MERGES):
    combo = ids[:-1].astype(np.int64) * 65536 + ids[1:]
    uniq, counts = np.unique(combo, return_counts=True)
    j = int(np.argmax(counts))
    c = int(uniq[j])
    a, b = c >> 16, c & 0xFFFF
    hit = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
    if hit.size == 0:
        break
    if a == b:
        keep, last = [], -2
        for x in hit:
            if x > last + 1:
                keep.append(x)
                last = x
        keep = np.array(keep)
    else:
        keep = hit
    ids[keep] = 256 + i
    ids = np.delete(ids, keep + 1)
    out.append([a, b, 256 + i])
    if i % 256 == 0 or i == NUM_MERGES - 1:
        print(f"  {i:4d}/{NUM_MERGES} · {len(ids):,} ids · {len(text)/len(ids):.2f} c/t "
              f"(+{time.time() - T0:,.0f}s)", flush=True)

(ROOT / "merges-bilingual.json").write_text(json.dumps(out))
print(f"saved {len(out)} merges -> merges-bilingual.json · "
      f"{len(text) / len(ids):.2f} chars/token (+{time.time() - T0:,.0f}s)", flush=True)
