"""
GPT-0 Academy — Module 6: build a BPE tokenizer on your own Hansard.
Stdlib only. Train on a subsample, then encode/decode and verify losslessness.

Run:  python bpe_hansard.py        (or: uv run bpe_hansard.py)

What you'll see:
  - the first merges learned (watch " t", "he", " the", "Speaker" crystallize)
  - compression ratio: ~chars per token on held-out text
  - a lossless round-trip assert (BPE MUST be reversible — that's why it's safe)

Knobs at the top: TRAIN_CHARS, NUM_MERGES.
"""

import gzip
import json
import os
import urllib.request

# ------------------------------------------------------------------ config --
CORPUS_URL = (
    "https://huggingface.co/datasets/NathanielArfin/canadian-hansard-44-1/"
    "resolve/main/lm/en.txt"
)
LOCAL_FALLBACK = "input.txt"        # if you're in gpt-from-scratch/, use the local copy
CORPUS_FILE = "hansard-en.txt"
SLICE = 10_000_000                  # full corpus slice (same as the GPT run)
TRAIN_CHARS = 1_000_000             # chars used to LEARN merges (pure python: keep ≤2M)
NUM_MERGES = 512                    # final vocab = 256 bytes + NUM_MERGES
VAL_CHARS = 200_000                 # held-out slice for compression measurement

# ------------------------------------------------------------------- data --
if not os.path.exists(CORPUS_FILE):
    if os.path.exists(LOCAL_FALLBACK):
        CORPUS_FILE = LOCAL_FALLBACK
    else:
        print("fetching corpus from Hugging Face ...")
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_FILE)

text = open(CORPUS_FILE, encoding="utf-8").read()[:SLICE]
print(f"corpus: {len(text):,} chars")

# ---------------------------------------------------------------- BPE core --
def get_stats(ids):
    """count every adjacent pair — the frequency table of the current token stream"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """replace every occurrence of `pair` with the new token `idx`, left to right"""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

# ----------------------------------------------------------------- training --
ids = list(text[:TRAIN_CHARS].encode("utf-8"))       # start from raw bytes
merges = {}                                          # (a, b) -> new token id
tok_bytes = {i: bytes([i]) for i in range(256)}      # id -> bytes (for pretty printing)
print(f"training {NUM_MERGES} merges on {TRAIN_CHARS:,} chars (stdlib python: ~3-5 min) ...")

for i in range(NUM_MERGES):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)                 # most frequent adjacent pair
    if stats[pair] == 1:                             # nothing left worth merging
        print(f"stopped at merge {i}: no pair occurs more than once")
        break
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx
    tok_bytes[idx] = tok_bytes[pair[0]] + tok_bytes[pair[1]]
    if i < 25:                                       # the opening moves, decoded live
        print(f"  merge {i:3d}: {pair} -> {idx}   {tok_bytes[idx]!r}   (count {stats[pair]:,})")

print(f"learned {len(merges)} merges · vocab size {256 + len(merges)}")

# ------------------------------------------------------------ encode/decode --
def encode(s):
    ids = list(s.encode("utf-8"))
    for pair, idx in merges.items():                 # apply merges in learned order
        ids = merge(ids, pair, idx)
    return ids

def decode(ids):
    tokens = []
    for i in ids:
        b = []                                       # expand token -> bytes, in order
        stack = [i]
        while stack:
            t = stack.pop()
            if t >= 256:
                a, b2 = [k for k, v in merges.items() if v == t][0]
                stack.extend([b2, a])                # pop a-branch fully, then b-branch
            else:
                b.append(t)
        tokens.extend(b)                             # b is already in the right order
    return bytes(tokens).decode("utf-8", errors="replace")

# ------------------------------------------------------------ verification --
val_text = text[-VAL_CHARS:]                         # held-out slice (never trained on)
val_ids = encode(val_text)
assert decode(val_ids) == val_text, "BPE round-trip FAILED (it must be lossless!)"
ratio = len(val_text) / len(val_ids)

print(f"\nheld-out slice: {len(val_text):,} chars -> {len(val_ids):,} tokens")
print(f"compression: {ratio:.2f} chars per token (char-level was 1.00)")
print(f"lossless round-trip: OK")

# a real Hansard passage, tokenized — '|' marks token boundaries
sample = "Mr. Speaker, pursuant to Standing Order 108(3)(h), the committee has examined the report."
tok = encode(sample)
parts = []
for t in tok:
    parts.append(decode([t]).replace("\n", "\\n"))
print("\ntoken boundaries:")
print(" | ".join(parts))

json.dump([[a, b, idx] for (a, b), idx in merges.items()], open("merges.json", "w"))
print("\nsaved merges -> merges.json (reuse these — a checkpoint's vocab is its tokenizer)")
