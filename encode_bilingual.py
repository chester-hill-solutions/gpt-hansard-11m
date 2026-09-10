# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Encode the bilingual corpus with merges-bilingual.json → tokens-bilingual.bin.

Per-session (en, fr) encode → append, round-trip verified per session.
Output: tokens-bilingual.bin (uint16) + tokens-bilingual-info.json
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
H = ROOT / "hansard"
CHRONO = [(39, 1), (39, 2), (40, 1), (40, 2), (40, 3), (41, 1), (41, 2),
          (42, 1), (43, 1), (43, 2), (44, 1), (45, 1)]
T0 = time.time()


def log(m):
    print(f"[encode-bi] {m}   (+{time.time() - T0:,.0f}s)", flush=True)


merges_list = json.load(open(ROOT / "merges-bilingual.json"))
mmap = {(a, b): i for a, b, i in merges_list}
tb = {i: bytes([i]) for i in range(256)}
for a, b, i in merges_list:
    tb[i] = tb[a] + tb[b]
vocab = 256 + len(merges_list)


def greedy_keep(hit):
    keep, last = [], -2
    for i in hit:
        if i > last + 1:
            keep.append(i)
            last = i
    return np.array(keep)


def encode_text(text):
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    for (a, b), idx in mmap.items():
        hit = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
        if hit.size == 0:
            continue
        keep = greedy_keep(hit) if a == b else hit
        ids[keep] = idx
        ids = np.delete(ids, keep + 1)
    return ids.astype(np.uint16)


def decode_ids(ids):
    return b"".join(tb[int(i)] for i in ids).decode("utf-8", errors="replace")


out_bin = ROOT / "tokens-bilingual.bin"
out_bin.unlink(missing_ok=True)
info, total_c, total_t = [], 0, 0
for parl, sess in CHRONO:
    for lang, name in (
        ("en", "input.txt" if (parl, sess) == (44, 1) else f"input-{parl}-{sess}.txt"),
        ("fr", "input-fr.txt" if (parl, sess) == (44, 1) else f"input-fr-{parl}-{sess}.txt"),
    ):
        src = H / name
        if not src.exists():
            log(f"MISSING {name} — skipped")
            continue
        text = src.read_text(encoding="utf-8")
        ids = encode_text(text)
        for frac in (0.25, 0.5, 0.75):
            seg = text[int(len(text) * frac):int(len(text) * frac) + 3000]
            if seg and decode_ids(encode_text(seg)) != seg:
                raise SystemExit(f"ROUND-TRIP FAILED in {name}")
        with open(out_bin, "ab") as f:
            ids.tofile(f)
        total_c += len(text)
        total_t += len(ids)
        info.append({"session": f"{parl}-{sess}-{lang}", "chars": len(text),
                     "tokens": int(len(ids))})
        log(f"{name}: {len(text):,} -> {len(ids):,} tokens ({len(text)/len(ids):.2f} c/t)")

(ROOT / "tokens-bilingual-info.json").write_text(json.dumps({
    "vocab": vocab, "merges": len(merges_list), "sessions": info,
    "total_chars": total_c, "total_tokens": int(total_t),
    "chars_per_token": round(total_c / total_t, 3)}, indent=2))
log(f"DONE: {total_c:,} chars -> {total_t:,} tokens ({total_c/total_t:.2f} c/t) · "
    f"vocab {vocab}")
