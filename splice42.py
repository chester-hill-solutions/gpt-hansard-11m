# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""One-off fix: 42-1 (2015–2019) hit the 400-sitting cap — its archive extends past it.

Fetches sittings 401+ (miss-streak stop), appends to input-42-1.txt, rebuilds
hansard-2006-now.txt, encodes the tail, and SPLICES the tail tokens into
tokens.bin at 42-1's end offset (uint16 stream, byte-exact insertion).
Updates tokens-info.json and MORNING.md with the corrected totals.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HANSARD = ROOT / "hansard"
CHRONO = [(39, 1), (39, 2), (40, 1), (40, 2), (40, 3), (41, 1), (41, 2),
          (42, 1), (43, 1), (43, 2), (44, 1), (45, 1)]
T0 = time.time()


def log(msg):
    print(f"[splice] {msg}   (+{time.time() - T0:,.0f}s)", flush=True)


# 1 -- fetch the real tail -----------------------------------------------------
tail_path = HANSARD / "input-42-1-tail.txt"
r = subprocess.run(
    [sys.executable, "fetch_hansard.py", "--parl", "42", "--session", "1",
     "--from-sitting", "401", "--max-sittings", "400", "--delay", "0.7",
     "--out", tail_path.name],
    cwd=HANSARD, capture_output=True, text=True)
log(r.stdout.strip().splitlines()[-1])

# 2 -- append to the per-session file, rebuild the big concat ------------------
old42 = (HANSARD / "input-42-1.txt").read_text(encoding="utf-8")
tail = tail_path.read_text(encoding="utf-8")
(HANSARD / "input-42-1.txt").write_text(old42 + tail, encoding="utf-8")
total = 0
with open(ROOT / "hansard-2006-now.txt", "w", encoding="utf-8") as w:
    for p, s in CHRONO:
        name = "input.txt" if (p, s) == (44, 1) else f"input-{p}-{s}.txt"
        t = (HANSARD / name).read_text(encoding="utf-8")
        w.write(t)
        if not t.endswith("\n"):
            w.write("\n")
        total += len(t)
log(f"concat rebuilt: {total:,} chars")

# 3 -- encode the tail ---------------------------------------------------------
merges_list = json.load(open(ROOT / "merges.json"))
mmap = {(a, b): i for a, b, i in merges_list}
tb = {i: bytes([i]) for i in range(256)}
for a, b, i in merges_list:
    tb[i] = tb[a] + tb[b]


def encode_text(text):
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    for (a, b), idx in mmap.items():
        hit = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
        if hit.size == 0:
            continue
        if a == b:
            keep, last = [], -2
            for i in hit:
                if i > last + 1:
                    keep.append(i)
                    last = i
            keep = np.array(keep)
        else:
            keep = hit
        ids[keep] = idx
        ids = np.delete(ids, keep + 1)
    return ids.astype(np.uint16)


def decode_ids(ids):
    return b"".join(tb[int(i)] for i in ids).decode("utf-8", errors="replace")


tail_ids = encode_text(tail)
for frac in (0.25, 0.5, 0.75):
    seg = tail[int(len(tail) * frac):int(len(tail) * frac) + 3000]
    if seg and decode_ids(encode_text(seg)) != seg:
        raise SystemExit("ROUND-TRIP FAILED on 42-1 tail")
log(f"tail: {len(tail):,} chars -> {len(tail_ids):,} tokens (round-trip OK)")

# 4 -- splice into tokens.bin at 42-1's END ------------------------------------
info = json.load(open(ROOT / "tokens-info.json"))
per = {e["session"]: e for e in info["sessions"]}
prefix = sum(per[f"{p}-{s}"]["tokens"] for p, s in CHRONO if (p, s) != (42, 1)
             and f"{p}-{s}" in per and CHRONO.index((p, s)) < CHRONO.index((42, 1)))
pos = prefix + per["42-1"]["tokens"]          # first token of 43-1
data = np.fromfile(ROOT / "tokens.bin", dtype=np.uint16)
log(f"splice point: token {pos:,} of {len(data):,} (inserting {len(tail_ids):,})")
new = np.concatenate([data[:pos], tail_ids, data[pos:]])
tmp = ROOT / "tokens-splice.tmp"
new.tofile(tmp)
tmp.replace(ROOT / "tokens.bin")
log(f"tokens.bin: {len(data):,} -> {len(new):,} tokens")

# 5 -- verify the seam decodes to real corpus text -----------------------------
c = sum(per[f"{p}-{s}"]["chars"] for p, s in CHRONO if CHRONO.index((p, s)) < 7)
c += per["42-1"]["chars"] + len(tail)         # start of 43-1 in rebuilt concat
corpus = (ROOT / "hansard-2006-now.txt").read_text(encoding="utf-8")
seam = decode_ids(new[pos - 100:pos + 100])
if corpus.find(seam) == -1:
    raise SystemExit("SEAM VERIFY FAILED — spliced text not found in corpus")
log(f"seam verified: 100-token window around splice found verbatim in corpus")

# 6 -- update tokens-info.json + MORNING.md ------------------------------------
per["42-1"]["chars"] += len(tail)
per["42-1"]["tokens"] += int(len(tail_ids))
info["total_chars"] = total
info["total_tokens"] = int(len(new))
info["chars_per_token"] = round(total / len(new), 3)
(ROOT / "tokens-info.json").write_text(json.dumps(info, indent=2))

vocab = info["vocab"]
upload_mb = (ROOT / "tokens.bin").stat().st_size / 1e6
(ROOT / "MORNING.md").write_text(f"""# Morning run — capstone on the 2006–now corpus

## Built overnight
- `hansard-2006-now.txt` — {total:,} chars, 12 sessions (39-1 → 45-1 + reused 44-1; 42-1 spliced past its 400-sitting cap)
- `merges.json` — 1,024 merges, **vocab {vocab}** (trained on an era-sampled slice)
- `tokens.bin` — **{len(new):,} tokens** (uint16, {upload_mb:,.0f} MB) at {total / len(new):.2f} chars/token
- `tokens-info.json` — per-session table
- Old 44-1 tokenizer backed up as `merges-441.json` / `tokens-441.bin`

## Colab (T4 runtime)
1. Upload three files from this folder: `gpt_tokens.py`, `merges.json`, `tokens.bin`
   ({upload_mb:,.0f} MB — the only big one)
2. Run:
```
!python gpt_tokens.py --preset default --sft --steps 20000 --batch 16
```
   - step-0 loss should read ≈ {np.log(vocab):.2f} (= ln {vocab})
   - 20,000 steps × 16 × 512 = ~164M tokens · ETA ~8–11 h on T4 (overnight)
   - shorter session? `--steps 12000` ≈ ~98M tokens, ~5–7 h
3. Plan B (if tokens.bin upload stalls): upload `hansard-2006-now.txt` instead and
   add `--max-encode-chars 999999999` (re-encodes ~25 min on Colab before training)

## Bring back
Loss curve + the three `[prompt]/[model]` samples + SFT Q/A answers.
We then compute **bits per character** vs the char model's 1.378 nats/char —
the fair cross-tokenization verdict.
""")
log("done — tokens-info.json and MORNING.md updated")
